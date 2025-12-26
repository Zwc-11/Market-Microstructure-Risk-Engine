from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.bars.time_bars import build_time_bars, resample_time_bars
from src.data.binance import BinanceFuturesRest, BinanceVision
from src.data.deepcoin import DeepcoinRest
from src.data.okx import OKXHistoricalDownloader
from src.data.okx_manual import load_manual_datasets
from src.data.bybit_manual import (
    detect_bybit_trades_gz,
    iter_bybit_trades_gz_strict,
    load_bybit_manual_datasets,
    scan_orderbook_payloads,
)
from src.data.schemas import SCHEMAS, enforce_schema
from src.data.storage import write_partitioned_parquet


def _load_config(path: str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_date_to_ms(date_str: str) -> int:
    ts = pd.Timestamp(date_str)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.value // 1_000_000)


def _parse_end_to_ms(date_str: str, end_inclusive_date: bool) -> int:
    ts = pd.Timestamp(date_str)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    if end_inclusive_date:
        ts = ts + pd.Timedelta(days=1)
    return int(ts.value // 1_000_000)


def _require_pyarrow() -> None:
    try:
        import pyarrow  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "pyarrow is required for parquet writes. Install with `pip install pyarrow`."
        ) from exc


def _partition_manifest_path(part_dir: Path) -> Path:
    return part_dir / "manifest.json"


def _source_fingerprint(paths: List[Path]) -> List[Dict[str, object]]:
    entries = []
    for path in sorted(paths, key=lambda p: str(p).lower()):
        try:
            stat = path.stat()
        except OSError:
            stat = None
        entries.append(
            {
                "path": str(path.resolve()),
                "size_bytes": int(stat.st_size) if stat else None,
                "mtime": int(stat.st_mtime) if stat else None,
            }
        )
    return entries


def _partition_stats(part_dir: Path, time_col: str = "ts") -> Dict[str, object]:
    files = sorted(part_dir.glob("part-*.parquet"))
    if not files:
        return {"rows": 0, "min_ts": None, "max_ts": None}
    rows = 0
    min_ts = None
    max_ts = None
    for path in files:
        df = pd.read_parquet(path, columns=[time_col])
        if df.empty:
            continue
        rows += len(df)
        ts_min = int(df[time_col].min())
        ts_max = int(df[time_col].max())
        min_ts = ts_min if min_ts is None else min(min_ts, ts_min)
        max_ts = ts_max if max_ts is None else max(max_ts, ts_max)
    return {"rows": int(rows), "min_ts": min_ts, "max_ts": max_ts}


def _load_partition_manifest(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_partition_manifest(
    part_dir: Path, dataset: str, source_files: List[Path], stats: Dict[str, object]
) -> None:
    manifest = {
        "dataset": dataset,
        "source_files": _source_fingerprint(source_files),
        "rows": stats.get("rows"),
        "min_ts": stats.get("min_ts"),
        "max_ts": stats.get("max_ts"),
        "created_at": pd.Timestamp.utcnow().isoformat(),
    }
    path = _partition_manifest_path(part_dir)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _manifest_matches(manifest: Dict, source_files: List[Path]) -> bool:
    if not manifest:
        return False
    expected = manifest.get("source_files", [])
    current = _source_fingerprint(source_files)
    return expected == current


def _partition_ready(
    part_dir: Path,
    dataset: str,
    source_files: List[Path],
    force_rebuild: bool,
    time_col: str = "ts",
) -> bool:
    if force_rebuild:
        if part_dir.exists():
            shutil.rmtree(part_dir, ignore_errors=True)
        return False
    manifest = _load_partition_manifest(_partition_manifest_path(part_dir))
    if _manifest_matches(manifest or {}, source_files):
        return True
    if part_dir.exists() and list(part_dir.glob("part-*.parquet")):
        if manifest is None:
            stats = _partition_stats(part_dir, time_col=time_col)
            _write_partition_manifest(part_dir, dataset, source_files, stats)
            return True
        raise ValueError(
            f"Cached partition mismatch for {dataset} at {part_dir}. "
            "Use --force-rebuild to refresh."
        )
    return False


def _build_bars_from_klines(klines: pd.DataFrame) -> pd.DataFrame:
    bars = klines.copy()
    bars["timestamp"] = pd.to_datetime(bars["ts"], unit="ms", utc=True)
    bars = bars.rename(
        columns={"open": "mid_open", "high": "mid_high", "low": "mid_low", "close": "mid_close"}
    )
    vwap = bars["quote_volume"] / bars["volume"]
    vwap = vwap.where(bars["volume"] > 0, bars["mid_close"])
    bars["vwap"] = vwap
    return bars[["timestamp", "mid_open", "mid_high", "mid_low", "mid_close", "volume", "vwap"]]


def _parse_datasets_arg(datasets: str | None) -> list[str]:
    if not datasets:
        return []
    return [item.strip() for item in datasets.split(",") if item.strip()]


def _init_bybit_trades_debug(start_ms: int, end_ms: int, start_str: str, end_str: str) -> Dict:
    return {
        "files_found": [],
        "files_skipped_cached": [],
        "files_used": [],
        "rows_seen_total": 0,
        "rows_kept_total": 0,
        "min_ts_seen": None,
        "max_ts_seen": None,
        "min_ts_kept": None,
        "max_ts_kept": None,
        "invalid_rows_dropped": {
            "bad_ts": 0,
            "bad_price": 0,
            "bad_size": 0,
            "bad_side": 0,
            "bad_symbol": 0,
        },
        "sample_rows": [],
        "requested_start": start_str,
        "requested_end": end_str,
        "requested_start_ms": start_ms,
        "requested_end_ms": end_ms,
    }


def _finalize_bybit_trades_debug(debug: Dict) -> Dict:
    def _to_utc(value: Optional[int]) -> Optional[str]:
        if value is None:
            return None
        return pd.to_datetime(value, unit="ms", utc=True).isoformat()

    debug["min_ts_seen_utc"] = _to_utc(debug.get("min_ts_seen"))
    debug["max_ts_seen_utc"] = _to_utc(debug.get("max_ts_seen"))
    debug["min_ts_kept_utc"] = _to_utc(debug.get("min_ts_kept"))
    debug["max_ts_kept_utc"] = _to_utc(debug.get("max_ts_kept"))
    return debug


def _write_bybit_trades_debug(debug: Dict, diagnostics_dir: Path) -> None:
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    _finalize_bybit_trades_debug(debug)
    path = diagnostics_dir / "bybit_trades_debug.json"
    path.write_text(json.dumps(debug, indent=2), encoding="utf-8")


def _bybit_trades_to_agg_trades(trades: pd.DataFrame, symbol: str) -> pd.DataFrame:
    payload = trades[["ts_ms", "side", "price", "size", "trade_id"]].copy()
    agg_id = pd.util.hash_pandas_object(payload, index=False).astype("int64")
    out = pd.DataFrame(
        {
            "agg_id": agg_id,
            "ts": trades["ts_ms"].astype("int64"),
            "price": trades["price"].astype("float64"),
            "qty": trades["size"].astype("float64"),
            "is_buyer_maker": trades["side"].astype(str).str.lower().eq("sell"),
            "symbol": symbol,
        }
    )
    return enforce_schema(out, SCHEMAS["agg_trades"])


def _write_bybit_trades_partitions(
    trades: pd.DataFrame,
    processed_dir: Path,
    exchange: str,
    symbol: str,
    part_counters: Dict[str, int],
    partitions: List[Dict],
) -> int:
    if trades.empty:
        return 0
    ts = pd.to_datetime(trades["ts"], unit="ms", utc=True)
    trades = trades.assign(_date=ts.dt.strftime("%Y-%m-%d"))
    rows_written = 0
    for date_str, part in trades.groupby("_date"):
        part = part.drop(columns=["_date"])
        date_path = (
            processed_dir
            / "agg_trades"
            / f"exchange={exchange}"
            / f"symbol={symbol}"
            / f"date={date_str}"
        )
        date_path.mkdir(parents=True, exist_ok=True)
        idx = part_counters.get(date_str, 0)
        file_path = date_path / f"part-{idx:03d}.parquet"
        part.to_parquet(file_path, index=False, engine="pyarrow")
        part_counters[date_str] = idx + 1
        partitions.append(
            {
                "date": date_str,
                "path": str(file_path.relative_to(processed_dir)),
                "rows": int(len(part)),
            }
        )
        rows_written += int(len(part))
    return rows_written


def _get_depth_levels(cfg: dict, exchange: str, symbol: str) -> int:
    for ex in cfg.get("data", {}).get("exchanges", []):
        if ex.get("name") == exchange and (not ex.get("symbols") or symbol in ex.get("symbols", [])):
            return int(ex.get("depth_levels", 10))
    return 10


def _extract_date_from_path(path: Path) -> Optional[str]:
    import re

    for part in (path.name, path.parent.name):
        match = re.search(r"\d{4}-\d{2}-\d{2}", part)
        if match:
            return match.group(0)
    return None


def _minute_coverage(df: pd.DataFrame, start_ms: int, end_ms: int) -> float:
    if df.empty or "ts" not in df.columns:
        return 0.0
    ts = pd.to_datetime(df["ts"], unit="ms", utc=True, errors="coerce")
    minutes = ts.dt.floor("min").dropna().unique()
    expected = max(1, int((end_ms - start_ms) / 60_000))
    return float(len(minutes) / expected)


def build_dataset(
    cfg: dict,
    exchange: str,
    symbol: str,
    start: str,
    end: str,
    source: str = "rest",
    datasets: str | None = None,
    mode: str = "rest",
    build_bars: bool = False,
    vision_dir: str | None = None,
    vision_auto_download: bool = True,
    okx_dir: str | None = None,
    okx_cache_dir: str | None = None,
    okx_auto_download: bool = False,
    okx_modules: str | None = None,
    okx_level: int | None = None,
    okx_agg: str | None = None,
    okx_manual_candles_dir: str | None = None,
    okx_manual_trades_dir: str | None = None,
    okx_manual_book_dir: str | None = None,
    okx_store_top_levels: int | None = None,
    bybit_manual_root: str | None = None,
    bybit_manual_book_root: str | None = None,
    bybit_manual_trades_root: str | None = None,
    bybit_store_top_levels: int | None = None,
    bybit_book_sample_ms: int | None = None,
    end_inclusive_date: bool = False,
    force_rebuild: bool = False,
) -> None:
    raw_dir = Path(cfg["paths"]["raw_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    start_ms = _parse_date_to_ms(start)
    end_ms = _parse_end_to_ms(end, end_inclusive_date=end_inclusive_date)
    max_gap_seconds = cfg.get("data", {}).get("quality", {}).get("max_gap_seconds", 120)
    max_gap_ms = int(max_gap_seconds) * 1000

    dataset_list = _parse_datasets_arg(datasets)
    if exchange == "okx" and not dataset_list and okx_modules:
        dataset_list = _parse_datasets_arg(okx_modules)
    if not dataset_list:
        if exchange == "okx":
            dataset_list = ["trades", "book_depth"]
        elif exchange == "bybit":
            dataset_list = ["agg_trades", "book_depth"]
        else:
            dataset_list = ["klines_1m", "agg_trades"]

    klines = None
    trades_df = None
    book_depth_df = None

    if exchange == "binance":
        market = "usdm"
        depth_levels = _get_depth_levels(cfg, exchange, symbol)
        if source == "vision":
            local_dir = Path(vision_dir) if vision_dir else Path("data/vision")
            client = BinanceVision(local_dir=local_dir, auto_download=vision_auto_download)
            for dataset in dataset_list:
                df = client.fetch_dataset(dataset, symbol, start_ms, end_ms, depth_levels=depth_levels)
                if df.empty:
                    continue
                if dataset == "klines_1m":
                    klines = df
                expected_interval = 60_000 if dataset in {"klines_1m", "premium_kline", "mark_kline"} else None
                write_partitioned_parquet(
                    df,
                    raw_dir,
                    exchange,
                    market,
                    symbol,
                    dataset,
                    expected_interval_ms=expected_interval,
                    max_gap_ms=max_gap_ms,
                )
        elif source == "rest":
            client = BinanceFuturesRest()
            klines = None
            if "klines_1m" in dataset_list:
                klines = client.fetch_klines_1m(symbol, start_ms, end_ms)
                write_partitioned_parquet(
                    klines,
                    raw_dir,
                    exchange,
                    market,
                    symbol,
                    "klines_1m",
                    expected_interval_ms=60_000,
                    max_gap_ms=max_gap_ms,
                )
            if "agg_trades" in dataset_list:
                trades = client.fetch_agg_trades(symbol, start_ms, end_ms)
                write_partitioned_parquet(
                    trades,
                    raw_dir,
                    exchange,
                    market,
                    symbol,
                    "agg_trades",
                )
        else:
            raise ValueError("source must be 'vision' or 'rest'")
    elif exchange == "deepcoin":
        market = "perp"
        client = DeepcoinRest()
        if source != "rest":
            raise ValueError("deepcoin supports only rest source")
        klines = None
        if "klines_1m" in dataset_list:
            klines = client.fetch_klines_1m(symbol, start_ms, end_ms)
            write_partitioned_parquet(
                klines,
                raw_dir,
                exchange,
                market,
                symbol,
                "klines_1m",
                expected_interval_ms=60_000,
                max_gap_ms=max_gap_ms,
            )
        if "agg_trades" in dataset_list:
            trades = client.fetch_trades(symbol, start_ms, end_ms)
            write_partitioned_parquet(trades, raw_dir, exchange, market, symbol, "agg_trades")
    elif exchange == "okx":
        market = "swap"
        if source not in {"okx_hist", "okx_api", "okx_manual"}:
            raise ValueError("okx source must be 'okx_hist', 'okx_api', or 'okx_manual'")
        okx_cfg = cfg.get("data", {}).get("okx", {})
        depth_levels = int(okx_cfg.get("depth_levels", _get_depth_levels(cfg, exchange, symbol)))
        if source == "okx_manual":
            manual_cfg = okx_cfg.get("manual", {})
            candles_dir = Path(okx_manual_candles_dir or manual_cfg.get("candles_dir", ""))
            trades_dir = Path(okx_manual_trades_dir or manual_cfg.get("trades_dir", ""))
            book_dir = Path(okx_manual_book_dir or manual_cfg.get("book_dir", ""))
            store_levels = int(okx_store_top_levels or manual_cfg.get("store_top_levels", 50))
            allow_baseline_only = bool(manual_cfg.get("allow_baseline_only", False))

            def _book_writer(df: pd.DataFrame) -> None:
                write_partitioned_parquet(
                    df,
                    raw_dir,
                    exchange,
                    market,
                    symbol,
                    "book_depth",
                )

            datasets_map, run_manifest = load_manual_datasets(
                candles_dir=candles_dir,
                trades_dir=trades_dir,
                book_dir=book_dir,
                symbol=symbol,
                start_ms=start_ms,
                end_ms=end_ms,
                store_levels=store_levels,
                allow_missing_book=allow_baseline_only,
                book_writer=_book_writer,
                diagnostics_dir=Path(cfg["paths"].get("artifacts_dir", "artifacts")) / "diagnostics",
                start_str=start,
                end_str=end,
            )

            for dataset, parsed in datasets_map.items():
                if parsed.empty:
                    continue
                if dataset == "agg_trades":
                    trades_df = parsed
                if dataset == "book_depth":
                    book_depth_df = parsed
                if dataset == "klines_1m":
                    klines = parsed
                write_partitioned_parquet(
                    parsed,
                    raw_dir,
                    exchange,
                    market,
                    symbol,
                    dataset,
                )

            if run_manifest is not None:
                manifest_path = Path(cfg["paths"].get("artifacts_dir", "artifacts")) / "run_manifest.json"
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
        else:
            history_endpoint = okx_cfg.get(
                "history_endpoint", "https://www.okx.com/api/v5/public/market-data-history"
            )
            cache_dir = (
                Path(okx_cache_dir)
                if okx_cache_dir
                else (Path(okx_dir) if okx_dir else Path(okx_cfg.get("cache_dir", "data/okx")))
            )
            date_aggr_type = okx_agg or okx_cfg.get("date_aggr_type", "daily")
            orderbook_level = int(okx_level or okx_cfg.get("orderbook_level", 50))
            client = OKXHistoricalDownloader(
                local_dir=cache_dir,
                auto_download=okx_auto_download,
                history_endpoint=history_endpoint,
                date_aggr_type=date_aggr_type,
                orderbook_level=orderbook_level,
                depth_levels=depth_levels,
            )

            for dataset in dataset_list:
                okx_dataset = "agg_trades" if dataset == "trades" else dataset
                parsed = client.fetch_dataset(dataset, symbol, start_ms, end_ms)
                if parsed.empty:
                    continue
                if dataset in {"trades", "agg_trades"}:
                    trades_df = parsed
                if dataset == "book_depth":
                    book_depth_df = parsed
                write_partitioned_parquet(
                    parsed,
                    raw_dir,
                    exchange,
                    market,
                    symbol,
                    okx_dataset,
                )
    elif exchange == "bybit":
        market = "perp"
        if source != "bybit_manual":
            raise ValueError("bybit supports only bybit_manual source")
        bybit_cfg = cfg.get("data", {}).get("bybit", {})
        manual_cfg = bybit_cfg.get("manual", {})
        manual_root = Path(bybit_manual_root or manual_cfg.get("root", ""))
        book_root = Path(bybit_manual_book_root or manual_cfg.get("book_root", "") or manual_root)
        trades_root = Path(bybit_manual_trades_root or manual_cfg.get("trades_root", "") or manual_root)
        store_levels = int(bybit_store_top_levels or manual_cfg.get("store_top_levels", 50))
        book_sample_ms = int(bybit_book_sample_ms or manual_cfg.get("book_sample_ms", 1000))
        emit_every_delta = bool(manual_cfg.get("emit_every_delta", False))

        run_manifest: Dict = {
            "source": "bybit_manual",
            "symbol": symbol,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "files": [],
            "datasets": {},
        }

        diagnostics_dir = Path(cfg["paths"].get("artifacts_dir", "artifacts")) / "diagnostics"

        if "book_depth" in dataset_list:
            skip_book_dates: set[str] = set()
            payloads = scan_orderbook_payloads(book_root, symbol)
            for payload in payloads:
                path = Path(payload["selected"])
                date_str = _extract_date_from_path(path)
                if not date_str:
                    continue
                part_dir = (
                    raw_dir / exchange / market / symbol / "book_depth" / f"date={date_str}"
                )
                if _partition_ready(part_dir, "book_depth", [path], force_rebuild, time_col="ts"):
                    skip_book_dates.add(date_str)

            datasets_map, book_manifest = load_bybit_manual_datasets(
                book_root=book_root,
                symbol=symbol,
                start_ms=start_ms,
                end_ms=end_ms,
                store_levels=store_levels,
                book_sample_ms=book_sample_ms,
                diagnostics_dir=diagnostics_dir,
                emit_every_delta=emit_every_delta,
                start_str=start,
                end_str=end,
                candles_dir=None,
                trades_dir=None,
                skip_book_dates=skip_book_dates,
            )
            if book_manifest:
                run_manifest = book_manifest
            book_df = datasets_map.get("book_depth", pd.DataFrame())
            if not book_df.empty:
                book_depth_df = book_df
                write_partitioned_parquet(
                    book_df,
                    raw_dir,
                    exchange,
                    market,
                    symbol,
                    "book_depth",
                )
                for date_str in book_df["ts"].pipe(pd.to_datetime, unit="ms", utc=True).dt.strftime("%Y-%m-%d").unique():
                    part_dir = (
                        raw_dir
                        / exchange
                        / market
                        / symbol
                        / "book_depth"
                        / f"date={date_str}"
                    )
                    if part_dir.exists():
                        stats = _partition_stats(part_dir, time_col="ts")
                        source_files = [
                            Path(payload["selected"])
                            for payload in payloads
                            if _extract_date_from_path(Path(payload["selected"])) == date_str
                        ]
                        if source_files:
                            _write_partition_manifest(part_dir, "book_depth", source_files, stats)
            if skip_book_dates:
                run_manifest.setdefault("cached", {}).setdefault("book_depth", []).extend(
                    sorted(skip_book_dates)
                )

        if "agg_trades" in dataset_list:
            _require_pyarrow()
            debug = _init_bybit_trades_debug(start_ms, end_ms, start, end)
            files = detect_bybit_trades_gz(trades_root, symbol)
            debug["files_found"] = [
                {"date": date_str, "path": str(path.resolve())} for date_str, path in files
            ]
            files_to_process: List[tuple[str, Path]] = []
            for date_str, path in files:
                part_dir = (
                    processed_dir
                    / "agg_trades"
                    / f"exchange={exchange}"
                    / f"symbol={symbol}"
                    / f"date={date_str}"
                )
                if _partition_ready(part_dir, "agg_trades", [path], force_rebuild, time_col="ts"):
                    debug.setdefault("files_skipped_cached", []).append(
                        {"date": date_str, "path": str(path.resolve())}
                    )
                    run_manifest.setdefault("cached", {}).setdefault("agg_trades", []).append(date_str)
                    continue
                files_to_process.append((date_str, path))

            partitions: List[Dict] = []
            part_counters: Dict[str, int] = {}
            rows_written = 0
            for chunk in iter_bybit_trades_gz_strict(
                trades_root,
                symbol,
                start_ms=start_ms,
                end_ms=end_ms,
                chunksize=2_000_000,
                debug=debug,
                files=files_to_process,
            ):
                agg_trades = _bybit_trades_to_agg_trades(chunk, symbol)
                rows_written += _write_bybit_trades_partitions(
                    agg_trades,
                    processed_dir,
                    exchange,
                    symbol,
                    part_counters,
                    partitions,
                )

            _write_bybit_trades_debug(debug, diagnostics_dir)
            if debug["rows_seen_total"] > 0 and debug["rows_kept_total"] == 0:
                min_utc = debug.get("min_ts_seen_utc", "unknown")
                max_utc = debug.get("max_ts_seen_utc", "unknown")
                raise ValueError(
                    f"All bybit trades filtered out. Observed ts range: {min_utc}..{max_utc}. "
                    f"Requested: [{start},{end}). Check end-exclusive boundary."
                )

            run_manifest.setdefault("datasets", {}).setdefault("agg_trades", {})
            run_manifest["datasets"]["agg_trades"].update(
                {"rows": int(rows_written), "partitions": partitions}
            )
            run_manifest.setdefault("files", []).extend(
                [
                    {"dataset": "agg_trades", "path": str(path.resolve())}
                    for _, path in files_to_process
                ]
            )
            for date_str in {item["date"] for item in partitions}:
                part_dir = (
                    processed_dir
                    / "agg_trades"
                    / f"exchange={exchange}"
                    / f"symbol={symbol}"
                    / f"date={date_str}"
                )
                if part_dir.exists():
                    stats = _partition_stats(part_dir, time_col="ts")
                    source_files = [
                        path for d, path in files_to_process if d == date_str
                    ]
                    if source_files:
                        _write_partition_manifest(part_dir, "agg_trades", source_files, stats)

        if run_manifest is not None:
            manifest_path = Path(cfg["paths"].get("artifacts_dir", "artifacts")) / "run_manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    else:
        raise ValueError("exchange must be 'binance', 'deepcoin', 'okx', or 'bybit'")

    if build_bars:
        if klines is not None and not klines.empty:
            bars_1m = _build_bars_from_klines(klines)
        elif exchange in {"okx", "bybit"} and trades_df is not None and not trades_df.empty:
            trades = trades_df.copy()
            trades["timestamp"] = pd.to_datetime(trades["ts"], unit="ms", utc=True)
            l2 = None
            if book_depth_df is not None and not book_depth_df.empty:
                coverage = _minute_coverage(book_depth_df, start_ms, end_ms)
                if coverage >= 0.5:
                    l2 = book_depth_df.copy()
                    l2["timestamp"] = pd.to_datetime(l2["ts"], unit="ms", utc=True)
                    l2 = l2.rename(
                        columns={
                            "bid_price_1": "bid",
                            "ask_price_1": "ask",
                            "bid_size_1": "bid_size",
                            "ask_size_1": "ask_size",
                        }
                    )
            bars_1m = build_time_bars(
                trades,
                l2=l2,
                time_col="timestamp",
                price_col="price",
                qty_col="qty",
                l2_time_col="timestamp",
                bid_col="bid",
                ask_col="ask",
                bid_size_col="bid_size",
                ask_size_col="ask_size",
            )
            bars_1m = bars_1m.reset_index().rename(columns={"index": "timestamp"})
        else:
            if exchange == "bybit":
                return
            raise ValueError("klines_1m or trades required to build bars")
        bars_1m.to_parquet(processed_dir / "bars_1m.parquet", index=False)
        bars_5m = resample_time_bars(bars_1m.set_index("timestamp"), freq="5min")
        bars_5m.to_parquet(processed_dir / "bars_5m.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and build datasets.")
    parser.add_argument("--config", required=True, help="Path to config yaml")
    parser.add_argument("--exchange", required=True, choices=["binance", "deepcoin", "okx", "bybit"])
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--source",
        default="rest",
        choices=["rest", "vision", "okx_hist", "okx_api", "okx_manual", "bybit_manual"],
    )
    parser.add_argument("--datasets", default=None, help="Comma-separated datasets")
    parser.add_argument("--mode", default="rest", choices=["rest", "ws_l2"])
    parser.add_argument("--vision-dir", default=None, help="Local directory for Binance Vision data")
    parser.add_argument("--vision-auto-download", action="store_true", help="Download missing Binance Vision files")
    parser.add_argument("--okx-dir", default=None, help="Local directory for OKX historical data (deprecated)")
    parser.add_argument("--okx-cache-dir", default=None, help="Local cache directory for OKX historical data")
    parser.add_argument("--okx-auto-download", action="store_true", help="Download missing OKX files")
    parser.add_argument("--okx-modules", default=None, help="Comma-separated OKX datasets (overrides --datasets)")
    parser.add_argument("--okx-level", type=int, default=None, help="Order book depth level for OKX (default 50)")
    parser.add_argument("--okx-agg", default=None, help="OKX date aggregation (daily|monthly)")
    parser.add_argument("--okx-manual-candles-dir", default=None, help="Local OKX manual candles directory")
    parser.add_argument("--okx-manual-trades-dir", default=None, help="Local OKX manual trades directory")
    parser.add_argument("--okx-manual-book-dir", default=None, help="Local OKX manual book directory")
    parser.add_argument(
        "--okx-store-top-levels",
        type=int,
        default=None,
        help="Store top N levels for OKX orderbook (default 50)",
    )
    parser.add_argument(
        "--bybit-manual-root",
        default=None,
        help="Local BYBIT manual root directory (trades + book defaults)",
    )
    parser.add_argument("--bybit-manual-book-root", default=None, help="Local BYBIT manual book root directory")
    parser.add_argument(
        "--bybit-manual-trades-root",
        default=None,
        help="Local BYBIT manual trades root directory",
    )
    parser.add_argument(
        "--bybit-book-sample-ms",
        type=int,
        default=None,
        help="Emit BYBIT book snapshots every N ms (default 1000)",
    )
    parser.add_argument(
        "--bybit-store-top-levels",
        type=int,
        default=None,
        help="Store top N levels for BYBIT orderbook (default 50)",
    )
    parser.add_argument(
        "--end-inclusive-date",
        action="store_true",
        help="Treat end date as inclusive (internally adds 1 day to end).",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild datasets even if cached partitions exist.",
    )
    parser.add_argument("--build-bars", action="store_true", help="Build 1m/5m bars after download.")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    build_dataset(
        cfg,
        args.exchange,
        args.symbol,
        args.start,
        args.end,
        source=args.source,
        datasets=args.datasets,
        mode=args.mode,
        build_bars=args.build_bars,
        vision_dir=args.vision_dir,
        vision_auto_download=args.vision_auto_download,
        okx_dir=args.okx_dir,
        okx_cache_dir=args.okx_cache_dir,
        okx_auto_download=args.okx_auto_download,
        okx_modules=args.okx_modules,
        okx_level=args.okx_level,
        okx_agg=args.okx_agg,
        okx_manual_candles_dir=args.okx_manual_candles_dir,
        okx_manual_trades_dir=args.okx_manual_trades_dir,
        okx_manual_book_dir=args.okx_manual_book_dir,
        okx_store_top_levels=args.okx_store_top_levels,
        bybit_manual_root=args.bybit_manual_root,
        bybit_manual_book_root=args.bybit_manual_book_root,
        bybit_manual_trades_root=args.bybit_manual_trades_root,
        bybit_store_top_levels=args.bybit_store_top_levels,
        bybit_book_sample_ms=args.bybit_book_sample_ms,
        end_inclusive_date=args.end_inclusive_date,
        force_rebuild=args.force_rebuild,
    )


if __name__ == "__main__":
    main()
