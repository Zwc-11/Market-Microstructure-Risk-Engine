from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from zipfile import ZipFile

import pandas as pd
import requests

from src.data.schemas import L2_LEVELS, SCHEMAS, enforce_schema
from src.utils.http import request_json


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    def _norm(name: str) -> str:
        name = name.strip().lower().replace(" ", "")
        name = re.sub(r"[^a-z0-9_]+", "", name)
        return name

    return df.rename(columns={c: _norm(c) for c in df.columns})


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _parse_ts_ms(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("int64")
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return (parsed.view("int64") // 1_000_000).astype("int64")


def _read_okx_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".zip":
        with ZipFile(path) as zf:
            csv_files = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_files:
                raise ValueError(f"No CSV found in {path}")
            with zf.open(csv_files[0]) as f:
                return pd.read_csv(f)
    if path.suffix.lower() == ".gz":
        return pd.read_csv(path, compression="gzip")
    return pd.read_csv(path)


def _pick_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _parse_okx_trades(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = _normalize_columns(df)
    rename = {
        "instrument_name": "symbol",
        "instrumentname": "symbol",
        "trade_id": "agg_id",
        "tradeid": "agg_id",
        "id": "agg_id",
        "created_time": "ts",
        "createtime": "ts",
        "timestamp": "ts",
        "time": "ts",
        "px": "price",
        "price": "price",
        "sz": "qty",
        "size": "qty",
        "qty": "qty",
        "side": "side",
    }
    df = df.rename(columns=rename)

    required = ["ts", "price", "qty"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"OKX trades missing columns: {missing}")

    if "agg_id" not in df.columns:
        df["agg_id"] = range(1, len(df) + 1)
    if "side" not in df.columns:
        df["side"] = "buy"

    df["ts"] = _parse_ts_ms(df["ts"])
    df["price"] = _coerce_numeric(df["price"])
    df["qty"] = _coerce_numeric(df["qty"])
    side = df["side"].astype(str).str.lower()
    df["is_buyer_maker"] = side == "sell"
    df["symbol"] = symbol
    return enforce_schema(df, SCHEMAS["agg_trades"])


def _parse_book_ticker(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = _normalize_columns(df)
    rename = {
        "ts": "ts",
        "timestamp": "ts",
        "bidpx": "bid_price",
        "bestbidprice": "bid_price",
        "bidprice": "bid_price",
        "bidqty": "bid_qty",
        "bidsz": "bid_qty",
        "askpx": "ask_price",
        "bestaskprice": "ask_price",
        "askprice": "ask_price",
        "askqty": "ask_qty",
        "asksz": "ask_qty",
        "updateid": "update_id",
        "seqid": "update_id",
    }
    df = df.rename(columns=rename)

    required = ["ts", "bid_price", "bid_qty", "ask_price", "ask_qty"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"OKX book ticker missing columns: {missing}")

    if "update_id" not in df.columns:
        df["update_id"] = df["ts"]

    df["ts"] = _parse_ts_ms(df["ts"])
    df["update_id"] = _coerce_numeric(df["update_id"]).astype("int64")
    for col in ["bid_price", "bid_qty", "ask_price", "ask_qty"]:
        df[col] = _coerce_numeric(df[col])
    df["symbol"] = symbol
    return enforce_schema(df, SCHEMAS["book_ticker"])


def _parse_depth_arrays(raw: str, levels: int) -> List[float]:
    try:
        arr = json.loads(raw)
    except json.JSONDecodeError:
        return [float("nan")] * (levels * 2)
    out = []
    for i in range(levels):
        if i < len(arr):
            price = float(arr[i][0])
            size = float(arr[i][1])
        else:
            price = float("nan")
            size = float("nan")
        out.extend([price, size])
    return out


def _parse_okx_book_depth(df: pd.DataFrame, symbol: str, depth_levels: int) -> pd.DataFrame:
    df = _normalize_columns(df)
    ts_col = _pick_column(df, ["timems", "ts", "timestamp", "time"])
    if ts_col is None:
        raise ValueError("OKX book depth missing timestamp column")

    update_col = _pick_column(df, ["exchtimems", "updateid", "seqid", "checksum"])
    if update_col is None:
        update_col = ts_col

    per_level_cols = []
    for lvl in range(1, depth_levels + 1):
        per_level_cols.extend(
            [
                f"bid_{lvl}_px",
                f"bid_px_{lvl}",
                f"bidprice_{lvl}",
                f"bid_price_{lvl}",
                f"bid_{lvl}_qty",
                f"bid_{lvl}_sz",
                f"bid_qty_{lvl}",
                f"bid_sz_{lvl}",
                f"bid_size_{lvl}",
                f"ask_{lvl}_px",
                f"ask_px_{lvl}",
                f"askprice_{lvl}",
                f"ask_price_{lvl}",
                f"ask_{lvl}_qty",
                f"ask_{lvl}_sz",
                f"ask_qty_{lvl}",
                f"ask_sz_{lvl}",
                f"ask_size_{lvl}",
            ]
        )

    has_level_cols = any(col in df.columns for col in per_level_cols)

    out = pd.DataFrame({"ts": _parse_ts_ms(df[ts_col])})
    out["update_id"] = _coerce_numeric(df[update_col]).astype("int64")

    if has_level_cols:
        for lvl in range(1, depth_levels + 1):
            bid_px = _pick_column(df, [f"bid_{lvl}_px", f"bid_px_{lvl}", f"bidprice_{lvl}", f"bid_price_{lvl}"])
            bid_sz = _pick_column(
                df, [f"bid_{lvl}_qty", f"bid_{lvl}_sz", f"bid_qty_{lvl}", f"bid_sz_{lvl}", f"bid_size_{lvl}"]
            )
            ask_px = _pick_column(df, [f"ask_{lvl}_px", f"ask_px_{lvl}", f"askprice_{lvl}", f"ask_price_{lvl}"])
            ask_sz = _pick_column(
                df, [f"ask_{lvl}_qty", f"ask_{lvl}_sz", f"ask_qty_{lvl}", f"ask_sz_{lvl}", f"ask_size_{lvl}"]
            )

            out[f"bid_price_{lvl}"] = _coerce_numeric(df[bid_px]) if bid_px else float("nan")
            out[f"bid_size_{lvl}"] = _coerce_numeric(df[bid_sz]) if bid_sz else float("nan")
            out[f"ask_price_{lvl}"] = _coerce_numeric(df[ask_px]) if ask_px else float("nan")
            out[f"ask_size_{lvl}"] = _coerce_numeric(df[ask_sz]) if ask_sz else float("nan")
    else:
        bids_col = _pick_column(df, ["bids"])
        asks_col = _pick_column(df, ["asks"])
        if bids_col is None or asks_col is None:
            raise ValueError("OKX book depth missing bids/asks arrays or per-level columns")

        bid_vals = df[bids_col].astype(str).apply(lambda x: _parse_depth_arrays(x, depth_levels))
        ask_vals = df[asks_col].astype(str).apply(lambda x: _parse_depth_arrays(x, depth_levels))
        bid_matrix = pd.DataFrame(bid_vals.tolist(), columns=[f"bid_{i}" for i in range(depth_levels * 2)])
        ask_matrix = pd.DataFrame(ask_vals.tolist(), columns=[f"ask_{i}" for i in range(depth_levels * 2)])

        for lvl in range(1, depth_levels + 1):
            out[f"bid_price_{lvl}"] = bid_matrix[f"bid_{(lvl - 1) * 2}"].astype(float)
            out[f"bid_size_{lvl}"] = bid_matrix[f"bid_{(lvl - 1) * 2 + 1}"].astype(float)
            out[f"ask_price_{lvl}"] = ask_matrix[f"ask_{(lvl - 1) * 2}"].astype(float)
            out[f"ask_size_{lvl}"] = ask_matrix[f"ask_{(lvl - 1) * 2 + 1}"].astype(float)

    if depth_levels < L2_LEVELS:
        for lvl in range(depth_levels + 1, L2_LEVELS + 1):
            out[f"bid_price_{lvl}"] = float("nan")
            out[f"bid_size_{lvl}"] = float("nan")
            out[f"ask_price_{lvl}"] = float("nan")
            out[f"ask_size_{lvl}"] = float("nan")

    out["symbol"] = symbol
    return enforce_schema(out, SCHEMAS["book_depth"])


def _parse_funding_rate(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = _normalize_columns(df)
    rename = {"ts": "ts", "timestamp": "ts", "fundingrate": "funding_rate", "rate": "funding_rate"}
    df = df.rename(columns=rename)

    required = ["ts", "funding_rate"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"OKX funding rate missing columns: {missing}")

    df["ts"] = _parse_ts_ms(df["ts"])
    df["funding_rate"] = _coerce_numeric(df["funding_rate"])
    df["symbol"] = symbol
    return enforce_schema(df, SCHEMAS["funding_rate"])


def _hash_file(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_checksum(path: Path, expected: str) -> None:
    algo = "sha256" if len(expected) == 64 else "md5"
    actual = _hash_file(path, algo)
    if actual.lower() != expected.lower():
        raise ValueError(f"Checksum mismatch for {path}")


def _download_with_retries(url: str, dest: Path, max_retries: int = 5) -> None:
    backoff = 1.0
    for attempt in range(max_retries):
        resp = requests.get(url, stream=True, timeout=30)
        if resp.status_code == 200:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            return
        if resp.status_code in {418, 429, 500, 502, 503, 504}:
            retry_after = resp.headers.get("Retry-After")
            sleep_s = float(retry_after) if retry_after else backoff
            time.sleep(sleep_s)
            backoff *= 2.0
            continue
        resp.raise_for_status()
    resp.raise_for_status()


@dataclass(frozen=True)
class FileDescriptor:
    module: str
    dataset: str
    inst_id: str
    date: str
    url: str
    filename: str
    local_path: Path
    checksum: Optional[str] = None


class OKXHistoricalClient:
    def __init__(
        self,
        endpoint: str = "https://www.okx.com/api/v5/public/market-data-history",
        manifest_dir: Optional[Path] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.endpoint = endpoint
        self.session = session or requests.Session()
        self.manifest_dir = manifest_dir

    def list_files(
        self,
        dataset: str,
        inst_id: str,
        start_ms: int,
        end_ms: int,
        date_aggr_type: str,
        level: Optional[int],
        inst_type: str,
        allow_remote: bool,
        local_dir: Path,
    ) -> List[FileDescriptor]:
        module = self._dataset_module(dataset, level)
        begin, end = self._format_date_range(start_ms, end_ms, date_aggr_type)
        payload = self._fetch_manifest(
            module,
            inst_id,
            date_aggr_type,
            begin,
            end,
            level,
            inst_type,
            allow_remote,
        )
        if payload is None:
            return []
        files = self._parse_manifest(payload, module, dataset, inst_id, local_dir)
        start_dt = pd.to_datetime(start_ms, unit="ms", utc=True).normalize()
        end_dt = pd.to_datetime(end_ms, unit="ms", utc=True).normalize()
        if end_dt <= start_dt:
            end_dt = start_dt + pd.Timedelta(days=1)

        filtered = []
        for desc in files:
            if not desc.date:
                filtered.append(desc)
                continue
            try:
                date_dt = pd.to_datetime(desc.date, utc=True)
            except ValueError:
                filtered.append(desc)
                continue
            if start_dt <= date_dt < end_dt:
                filtered.append(desc)
        return filtered

    def _dataset_module(self, dataset: str, level: Optional[int]) -> str:
        mapping = {
            "trades": "1",
            "agg_trades": "1",
            "klines_1m": "2",
            "funding_rate": "3",
            "book_depth": None,
        }
        if dataset not in mapping:
            raise ValueError(f"Unsupported OKX dataset: {dataset}")
        if dataset != "book_depth":
            return mapping[dataset]
        depth_level = int(level or 50)
        if depth_level <= 50:
            return "6"
        if depth_level <= 400:
            return "4"
        return "5"

    def _format_date_range(self, start_ms: int, end_ms: int, date_aggr_type: str) -> tuple[str, str]:
        start_dt = pd.to_datetime(start_ms, unit="ms", utc=True).normalize()
        end_dt = pd.to_datetime(end_ms, unit="ms", utc=True).normalize()
        if end_dt <= start_dt:
            end_dt = start_dt
        else:
            end_dt = end_dt - pd.Timedelta(days=1)

        if date_aggr_type == "monthly":
            begin = start_dt.strftime("%Y%m")
            end = end_dt.strftime("%Y%m")
        else:
            begin = start_dt.strftime("%Y%m%d")
            end = end_dt.strftime("%Y%m%d")
        return begin, end

    def _manifest_cache_path(self, module: str, inst_id: str, date_aggr_type: str, level: Optional[int]) -> Optional[Path]:
        if self.manifest_dir is None:
            return None
        safe_inst = inst_id.replace("/", "_")
        level_tag = f"level{level}" if level is not None else "levelna"
        filename = f"manifest_{module}_{safe_inst}_{date_aggr_type}_{level_tag}.json"
        return self.manifest_dir / filename

    def _fetch_manifest(
        self,
        module: str,
        inst_id: str,
        date_aggr_type: str,
        begin: str,
        end: str,
        level: Optional[int],
        inst_type: str,
        allow_remote: bool,
    ) -> Optional[Dict]:
        cache_path = self._manifest_cache_path(module, inst_id, date_aggr_type, level)
        if cache_path is not None and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        if not allow_remote:
            return None

        params = {
            "module": module,
            "instType": inst_type,
            "instFamilyList": self._inst_family(inst_id),
            "dateAggrType": date_aggr_type,
            "begin": begin,
            "end": end,
        }
        if level is not None:
            params["level"] = str(level)

        payload = request_json(self.session, self.endpoint, params=params, timeout=15, max_retries=5)
        if payload is None:
            return None
        if payload.get("code") not in {"0", 0, None}:
            raise ValueError(f"OKX history endpoint error: {payload}")

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    def _parse_manifest(
        self,
        payload: Dict,
        module: str,
        dataset: str,
        inst_id: str,
        local_dir: Path,
    ) -> List[FileDescriptor]:
        out: List[FileDescriptor] = []
        data = payload.get("data", [])
        for item in data:
            for detail in item.get("details", []):
                group_details = detail.get("groupDetails") or []
                for group in group_details:
                    url = group.get("url") or group.get("link")
                    filename = group.get("fileName") or group.get("filename")
                    if not filename and url:
                        filename = Path(url).name
                    if not url or not filename:
                        continue
                    date_value = group.get("dateTs") or detail.get("dateTs") or group.get("date") or detail.get("date")
                    date_dt = _parse_date_value(date_value)
                    date_str = date_dt.strftime("%Y-%m-%d") if date_dt is not None else ""
                    checksum = group.get("checksum") or group.get("md5") or group.get("sha256")
                    local_path = local_dir / dataset / inst_id / filename
                    out.append(
                        FileDescriptor(
                            module=str(module),
                            dataset=dataset,
                            inst_id=inst_id,
                            date=date_str,
                            url=url,
                            filename=filename,
                            local_path=local_path,
                            checksum=checksum,
                        )
                    )
        return out

    @staticmethod
    def _inst_family(inst_id: str) -> str:
        if inst_id.endswith("-SWAP"):
            return inst_id.rsplit("-", 1)[0]
        parts = inst_id.split("-")
        if len(parts) >= 2:
            return "-".join(parts[:2])
        return inst_id


def _parse_date_value(value: Optional[object]) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        ts = int(value)
        unit = "ms" if ts > 10**12 else "s"
        return pd.to_datetime(ts, unit=unit, utc=True)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            if len(stripped) >= 13:
                return pd.to_datetime(int(stripped), unit="ms", utc=True)
            if len(stripped) == 10:
                return pd.to_datetime(int(stripped), unit="s", utc=True)
            if len(stripped) == 8:
                return pd.to_datetime(stripped, format="%Y%m%d", utc=True)
            if len(stripped) == 6:
                return pd.to_datetime(stripped + "01", format="%Y%m%d", utc=True)
        try:
            return pd.to_datetime(stripped, utc=True)
        except ValueError:
            return None
    try:
        return pd.to_datetime(value, utc=True)
    except ValueError:
        return None


class OKXHistoricalDownloader:
    def __init__(
        self,
        local_dir: Optional[Path] = None,
        auto_download: bool = False,
        history_endpoint: str = "https://www.okx.com/api/v5/public/market-data-history",
        date_aggr_type: str = "daily",
        orderbook_level: int = 50,
        depth_levels: int = L2_LEVELS,
        manifest_dir: Optional[Path] = None,
        inst_type: str = "SWAP",
    ) -> None:
        self.local_dir = Path(local_dir) if local_dir else None
        self.auto_download = auto_download
        self.date_aggr_type = date_aggr_type
        self.orderbook_level = orderbook_level
        self.depth_levels = min(depth_levels, L2_LEVELS)
        self.inst_type = inst_type
        self.client = OKXHistoricalClient(
            endpoint=history_endpoint,
            manifest_dir=manifest_dir or (self.local_dir / "_manifests" if self.local_dir else None),
        )

    def fetch_dataset(
        self,
        dataset: str,
        symbol: str,
        start_ms: int,
        end_ms: int,
    ) -> pd.DataFrame:
        if self.local_dir is None:
            return pd.DataFrame()

        files = self.client.list_files(
            dataset,
            symbol,
            start_ms,
            end_ms,
            date_aggr_type=self.date_aggr_type,
            level=self.orderbook_level if dataset == "book_depth" else None,
            inst_type=self.inst_type,
            allow_remote=self.auto_download,
            local_dir=self.local_dir,
        )

        frames: List[pd.DataFrame] = []
        run_manifest = {
            "dataset": dataset,
            "symbol": symbol,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "auto_download": self.auto_download,
            "files": [],
        }

        for desc in files:
            path = self._ensure_local_file(desc)
            status = "missing"
            if path and path.exists():
                df = _read_okx_file(path)
                parsed = self._parse_dataset(dataset, df, symbol)
                frames.append(parsed)
                status = "loaded"
            run_manifest["files"].append(
                {
                    "url": desc.url,
                    "filename": desc.filename,
                    "local_path": str(desc.local_path),
                    "checksum": desc.checksum,
                    "status": status,
                }
            )

        if self.local_dir is not None:
            manifest_dir = self.local_dir / "_manifests"
            manifest_dir.mkdir(parents=True, exist_ok=True)
            stamp = pd.Timestamp.utcnow().strftime("%Y%m%d%H%M%S")
            (manifest_dir / f"run_{dataset}_{symbol}_{stamp}.json").write_text(
                json.dumps(run_manifest, indent=2), encoding="utf-8"
            )

        if not frames:
            return pd.DataFrame()

        out = pd.concat(frames, ignore_index=True).sort_values("ts").drop_duplicates()
        out = out[(out["ts"] >= start_ms) & (out["ts"] < end_ms)]
        return out

    def _ensure_local_file(self, desc: FileDescriptor) -> Optional[Path]:
        if desc.local_path.exists():
            if desc.checksum:
                _verify_checksum(desc.local_path, desc.checksum)
            return desc.local_path
        if not self.auto_download:
            return None
        if not desc.url:
            return None
        _download_with_retries(desc.url, desc.local_path)
        if desc.checksum:
            _verify_checksum(desc.local_path, desc.checksum)
        return desc.local_path

    def _parse_dataset(self, dataset: str, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if dataset in {"trades", "agg_trades"}:
            return _parse_okx_trades(df, symbol)
        if dataset == "book_depth":
            return _parse_okx_book_depth(df, symbol, depth_levels=self.depth_levels)
        if dataset == "book_ticker":
            return _parse_book_ticker(df, symbol)
        if dataset == "funding_rate":
            return _parse_funding_rate(df, symbol)
        if dataset == "klines_1m":
            raise ValueError("OKX klines parsing not implemented; derive from trades or add parser.")
        raise ValueError(f"Unsupported OKX dataset: {dataset}")
