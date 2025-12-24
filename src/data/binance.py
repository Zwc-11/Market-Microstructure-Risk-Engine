from __future__ import annotations

import hashlib
from pathlib import Path
import re
import time
from typing import Dict, Iterable, List, Optional

import requests
from zipfile import ZipFile

import pandas as pd

from src.data.exchange_base import ExchangeBase
from src.data.schemas import L2_LEVELS, SCHEMAS, enforce_schema


class BinanceFuturesRest(ExchangeBase):
    def __init__(self, rate_limit_per_sec: float = 5.0) -> None:
        super().__init__(base_url="https://fapi.binance.com", name="binance", rate_limit_per_sec=rate_limit_per_sec)

    def fetch_klines_1m(self, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        interval_ms = 60_000
        rows: List[Dict] = []
        current = start_ms
        limit = 1500

        while current < end_ms:
            params = {
                "symbol": symbol,
                "interval": "1m",
                "startTime": current,
                "endTime": end_ms,
                "limit": limit,
            }
            data = self._get_json("/fapi/v1/klines", params=params)
            if not data:
                break

            for k in data:
                rows.append(
                    {
                        "ts": int(k[0]),
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5]),
                        "close_ts": int(k[6]),
                        "quote_volume": float(k[7]),
                        "trade_count": int(k[8]),
                        "taker_buy_base": float(k[9]),
                        "taker_buy_quote": float(k[10]),
                        "symbol": symbol,
                    }
                )

            last_open = int(data[-1][0])
            next_start = last_open + interval_ms
            if next_start <= current:
                break
            current = next_start

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df = df.sort_values("ts").drop_duplicates(subset=["symbol", "ts"], keep="last")
        return enforce_schema(df, SCHEMAS["klines_1m"])

    def fetch_agg_trades(self, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        rows: List[Dict] = []
        limit = 1000
        window_ms = 55 * 60 * 1000
        current = start_ms

        while current < end_ms:
            window_end = min(current + window_ms, end_ms)
            page_start = current

            while True:
                params = {
                    "symbol": symbol,
                    "startTime": page_start,
                    "endTime": window_end,
                    "limit": limit,
                }
                data = self._get_json("/fapi/v1/aggTrades", params=params)
                if not data:
                    break

                for t in data:
                    rows.append(
                        {
                            "agg_id": int(t["a"]),
                            "ts": int(t["T"]),
                            "price": float(t["p"]),
                            "qty": float(t["q"]),
                            "is_buyer_maker": bool(t["m"]),
                            "symbol": symbol,
                        }
                    )

                last_ts = int(data[-1]["T"])
                if last_ts < window_end and len(data) == limit:
                    next_start = last_ts + 1
                    if next_start <= page_start:
                        break
                    page_start = next_start
                    continue
                break

            current = window_end + 1

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df = df.sort_values(["ts", "agg_id"]).drop_duplicates(subset=["symbol", "agg_id"], keep="last")
        return enforce_schema(df, SCHEMAS["agg_trades"])

    def fetch_depth_snapshot(self, symbol: str, limit: int = 1000) -> Dict:
        params = {"symbol": symbol, "limit": limit}
        return self._get_json("/fapi/v1/depth", params=params)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    def _norm(name: str) -> str:
        name = name.strip().lower().replace(" ", "")
        name = re.sub(r"[^a-z0-9_]+", "", name)
        return name

    return df.rename(columns={c: _norm(c) for c in df.columns})


def _pick_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _coerce_bool(series: pd.Series) -> pd.Series:
    if series.dtype == "bool":
        return series
    return series.astype(str).str.lower().isin({"true", "1", "t"})


def _parse_klines_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = _normalize_columns(df)
    rename = {
        "opentime": "ts",
        "open_time": "ts",
        "close_time": "close_ts",
        "closetime": "close_ts",
        "quoteassetvolume": "quote_volume",
        "quote_asset_volume": "quote_volume",
        "numberoftrades": "trade_count",
        "number_of_trades": "trade_count",
        "takerbuybaseassetvolume": "taker_buy_base",
        "taker_buy_base_asset_volume": "taker_buy_base",
        "takerbuyquoteassetvolume": "taker_buy_quote",
        "taker_buy_quote_asset_volume": "taker_buy_quote",
    }
    df = df.rename(columns=rename)

    required = ["ts", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Klines CSV missing columns: {missing}")

    if "close_ts" not in df.columns:
        df["close_ts"] = _coerce_numeric(df["ts"]) + 60_000
    if "volume" not in df.columns:
        df["volume"] = 0.0
    if "quote_volume" not in df.columns:
        df["quote_volume"] = 0.0
    if "trade_count" not in df.columns:
        df["trade_count"] = 0
    if "taker_buy_base" not in df.columns:
        df["taker_buy_base"] = 0.0
    if "taker_buy_quote" not in df.columns:
        df["taker_buy_quote"] = 0.0

    df["symbol"] = symbol

    df["ts"] = _coerce_numeric(df["ts"]).astype("int64")
    df["close_ts"] = _coerce_numeric(df["close_ts"]).astype("int64")
    for col in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]:
        df[col] = _coerce_numeric(df[col])
    df["trade_count"] = _coerce_numeric(df["trade_count"]).fillna(0).astype("int64")
    return enforce_schema(df, SCHEMAS["klines_1m"])


def _parse_agg_trades_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = _normalize_columns(df)
    rename = {
        "aggtradeid": "agg_id",
        "aggregate_trade_id": "agg_id",
        "agg_trade_id": "agg_id",
        "timestamp": "ts",
        "time": "ts",
        "transact_time": "ts",
        "transacttime": "ts",
        "quantity": "qty",
    }
    df = df.rename(columns=rename)

    required = ["agg_id", "ts", "price", "qty"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"AggTrades CSV missing columns: {missing}")

    if "is_buyer_maker" not in df.columns:
        col = _pick_column(df, ["isbuyermaker", "buyer_maker"])
        if col:
            df["is_buyer_maker"] = df[col]
        else:
            df["is_buyer_maker"] = False

    df["symbol"] = symbol
    df["agg_id"] = _coerce_numeric(df["agg_id"]).astype("int64")
    df["ts"] = _coerce_numeric(df["ts"]).astype("int64")
    df["price"] = _coerce_numeric(df["price"])
    df["qty"] = _coerce_numeric(df["qty"])
    df["is_buyer_maker"] = _coerce_bool(df["is_buyer_maker"])
    return enforce_schema(df, SCHEMAS["agg_trades"])


def _parse_book_ticker_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = _normalize_columns(df)
    rename = {
        "eventtime": "ts",
        "event_time": "ts",
        "updateid": "update_id",
        "bidprice": "bid_price",
        "bid_price": "bid_price",
        "bestbidprice": "bid_price",
        "best_bid_price": "bid_price",
        "bidqty": "bid_qty",
        "bid_qty": "bid_qty",
        "bestbidqty": "bid_qty",
        "best_bid_qty": "bid_qty",
        "askprice": "ask_price",
        "ask_price": "ask_price",
        "bestaskprice": "ask_price",
        "best_ask_price": "ask_price",
        "askqty": "ask_qty",
        "ask_qty": "ask_qty",
        "bestaskqty": "ask_qty",
        "best_ask_qty": "ask_qty",
    }
    df = df.rename(columns=rename)

    required = ["ts", "bid_price", "bid_qty", "ask_price", "ask_qty"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"BookTicker CSV missing columns: {missing}")

    if "update_id" not in df.columns:
        df["update_id"] = df["ts"]

    df["symbol"] = symbol
    df["ts"] = _coerce_numeric(df["ts"]).astype("int64")
    df["update_id"] = _coerce_numeric(df["update_id"]).astype("int64")
    for col in ["bid_price", "bid_qty", "ask_price", "ask_qty"]:
        df[col] = _coerce_numeric(df[col])
    return enforce_schema(df, SCHEMAS["book_ticker"])


def _depth_candidates(side: str, field: str, level: int) -> List[str]:
    if field == "price":
        field_candidates = ["price", "px"]
    else:
        field_candidates = ["qty", "size", "quantity"]

    candidates = []
    for f in field_candidates:
        candidates.extend(
            [
                f"{side}_{f}_{level}",
                f"{side}{f}{level}",
                f"{side}{f}_{level}",
            ]
        )
    return candidates


def _parse_book_depth_df(df: pd.DataFrame, symbol: str, depth_levels: int) -> pd.DataFrame:
    df = _normalize_columns(df)
    ts_col = _pick_column(df, ["eventtime", "event_time", "timestamp", "time", "ts"])
    if ts_col is None:
        raise ValueError("BookDepth CSV missing event time column")
    update_col = _pick_column(df, ["update_id", "updateid", "u"])
    if update_col is None:
        df["update_id"] = df[ts_col]
        update_col = "update_id"

    out = pd.DataFrame({"ts": _coerce_numeric(df[ts_col]).astype("int64")})
    out["update_id"] = _coerce_numeric(df[update_col]).astype("int64")

    nan_value = float("nan")
    for lvl in range(1, depth_levels + 1):
        for side in ("bid", "ask"):
            price_col = _pick_column(df, _depth_candidates(side, "price", lvl))
            size_col = _pick_column(df, _depth_candidates(side, "size", lvl))
            out[f"{side}_price_{lvl}"] = _coerce_numeric(df[price_col]) if price_col else nan_value
            out[f"{side}_size_{lvl}"] = _coerce_numeric(df[size_col]) if size_col else nan_value

    if depth_levels < L2_LEVELS:
        for lvl in range(depth_levels + 1, L2_LEVELS + 1):
            for side in ("bid", "ask"):
                out[f"{side}_price_{lvl}"] = nan_value
                out[f"{side}_size_{lvl}"] = nan_value

    out["symbol"] = symbol
    return enforce_schema(out, SCHEMAS["book_depth"])


def _read_vision_csv(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".zip":
        with ZipFile(path) as zf:
            csv_files = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_files:
                raise ValueError(f"No CSV found in {path}")
            with zf.open(csv_files[0]) as f:
                return pd.read_csv(f)
    return pd.read_csv(path)


def _hash_file(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_checksum(file_path: Path) -> None:
    checksum_path = Path(f"{file_path}.CHECKSUM")
    if not checksum_path.exists():
        return
    content = checksum_path.read_text(encoding="utf-8").strip().split()
    if not content:
        raise ValueError(f"Empty checksum file: {checksum_path}")
    expected = content[0].strip()
    algo = "sha256" if len(expected) == 64 else "md5"
    actual = _hash_file(file_path, algo)
    if actual.lower() != expected.lower():
        raise ValueError(f"Checksum mismatch for {file_path}")


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


class BinanceVision:
    def __init__(
        self,
        local_dir: Optional[Path] = None,
        auto_download: bool = True,
        base_url: str = "https://data.binance.vision",
        market: str = "um",
    ) -> None:
        self.local_dir = Path(local_dir) if local_dir else None
        self.auto_download = auto_download
        self.base_url = base_url.rstrip("/")
        self.market = market

    def fetch_dataset(
        self,
        dataset: str,
        symbol: str,
        start_ms: int,
        end_ms: int,
        depth_levels: int = L2_LEVELS,
    ) -> pd.DataFrame:
        date_list = self._date_range_from_ms(start_ms, end_ms)
        frames: List[pd.DataFrame] = []

        for dt in date_list:
            file_path = self._resolve_local_file(dataset, symbol, dt)
            if file_path is None and self.auto_download:
                file_path = self._download_vision_file(dataset, symbol, dt)
            if file_path is None or not file_path.exists():
                continue
            _verify_checksum(file_path)
            df = _read_vision_csv(file_path)
            parsed = self._parse_dataset(dataset, df, symbol, depth_levels)
            frames.append(parsed)

        if not frames:
            return pd.DataFrame()

        out = pd.concat(frames, ignore_index=True)
        out = out.sort_values("ts").drop_duplicates()
        out = out[(out["ts"] >= start_ms) & (out["ts"] < end_ms)]
        return out

    def _date_range_from_ms(self, start_ms: int, end_ms: int) -> List[pd.Timestamp]:
        start_dt = pd.to_datetime(start_ms, unit="ms", utc=True).normalize()
        end_dt = pd.to_datetime(end_ms, unit="ms", utc=True).normalize()
        if end_dt <= start_dt:
            return [start_dt]
        return list(pd.date_range(start_dt, end_dt - pd.Timedelta(days=1), freq="D"))

    def _dataset_config(self, dataset: str) -> Dict[str, Optional[str]]:
        mapping = {
            "klines_1m": {"category": "klines", "interval": "1m", "suffix": "1m"},
            "agg_trades": {"category": "aggTrades", "interval": None, "suffix": "aggTrades"},
            "book_ticker": {"category": "bookTicker", "interval": None, "suffix": "bookTicker"},
            "book_depth": {"category": "bookDepth", "interval": None, "suffix": "bookDepth"},
            "premium_kline": {"category": "premiumIndexKlines", "interval": "1m", "suffix": "premiumIndexKline"},
            "mark_kline": {"category": "markPriceKlines", "interval": "1m", "suffix": "markPriceKline"},
        }
        if dataset not in mapping:
            raise ValueError(f"Unsupported vision dataset: {dataset}")
        return mapping[dataset]

    def _resolve_local_file(self, dataset: str, symbol: str, date: pd.Timestamp) -> Optional[Path]:
        if self.local_dir is None:
            return None
        cfg = self._dataset_config(dataset)
        date_str = date.strftime("%Y-%m-%d")
        filename = f"{symbol}-{cfg['suffix']}-{date_str}.zip"

        candidates = []
        root = self.local_dir
        candidates.append(
            root
            / "data"
            / "futures"
            / self.market
            / "daily"
            / cfg["category"]
            / symbol
            / (cfg["interval"] or "")
            / filename
        )
        candidates.append(root / "futures" / self.market / "daily" / cfg["category"] / symbol / (cfg["interval"] or "") / filename)
        candidates.append(root / cfg["category"] / symbol / (cfg["interval"] or "") / filename)
        candidates.append(root / filename)

        for path in candidates:
            if path.exists():
                return path
        return None

    def _download_vision_file(self, dataset: str, symbol: str, date: pd.Timestamp) -> Optional[Path]:
        if self.local_dir is None:
            return None
        cfg = self._dataset_config(dataset)
        date_str = date.strftime("%Y-%m-%d")
        filename = f"{symbol}-{cfg['suffix']}-{date_str}.zip"
        relative = Path("data") / "futures" / self.market / "daily" / cfg["category"] / symbol
        if cfg["interval"]:
            relative = relative / cfg["interval"]
        dest = self.local_dir / relative / filename
        url = f"{self.base_url}/{relative.as_posix()}/{filename}"
        _download_with_retries(url, dest)
        checksum_url = f"{url}.CHECKSUM"
        checksum_dest = Path(f"{dest}.CHECKSUM")
        _download_with_retries(checksum_url, checksum_dest)
        return dest

    def _parse_dataset(self, dataset: str, df: pd.DataFrame, symbol: str, depth_levels: int) -> pd.DataFrame:
        if dataset == "klines_1m":
            return _parse_klines_df(df, symbol)
        if dataset == "agg_trades":
            return _parse_agg_trades_df(df, symbol)
        if dataset == "book_ticker":
            return _parse_book_ticker_df(df, symbol)
        if dataset == "book_depth":
            return _parse_book_depth_df(df, symbol, depth_levels)
        if dataset in {"premium_kline", "mark_kline"}:
            parsed = _parse_klines_df(df, symbol)
            schema = SCHEMAS[dataset]
            return enforce_schema(parsed, schema)
        raise ValueError(f"Unsupported vision dataset: {dataset}")
