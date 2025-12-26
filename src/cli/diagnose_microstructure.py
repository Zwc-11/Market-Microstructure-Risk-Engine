from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from src.alpha.micro_features import build_micro_features
from src.bars.time_bars import build_time_bars


def _load_config(path: str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _config_hash(cfg: Dict) -> str:
    payload = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


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


def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        out = df.copy()
        out.index = idx
        return out.sort_index()
    for col in ("timestamp", "ts"):
        if col in df.columns:
            out = df.copy()
            out[col] = pd.to_datetime(out[col], utc=True)
            return out.set_index(col).sort_index()
    raise ValueError("DataFrame must include a DatetimeIndex or timestamp/ts column.")


def _date_range(start: str, end: str) -> List[str]:
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    if end_ts <= start_ts:
        return [start_ts.strftime("%Y-%m-%d")]
    dates = pd.date_range(start_ts, end_ts - pd.Timedelta(days=1), freq="D")
    return [d.strftime("%Y-%m-%d") for d in dates]


def _load_raw_dataset(
    raw_dir: Path,
    exchange: str,
    market: str,
    symbol: str,
    dataset: str,
    start_ms: int,
    end_ms: int,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    base = raw_dir / exchange / market / symbol / dataset
    if not base.exists():
        return pd.DataFrame()
    files = sorted(base.glob("date=*/part-000.parquet"))
    if not files:
        return pd.DataFrame()
    dates = set(_date_range(pd.Timestamp(start_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d"),
                            pd.Timestamp(end_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d")))
    filtered = []
    for path in files:
        date_part = path.parent.name.replace("date=", "")
        if date_part in dates:
            filtered.append(path)
    files = filtered or files

    cols = None
    if columns is not None:
        cols = list(dict.fromkeys(columns))
        if "ts" not in cols:
            cols = ["ts"] + cols

    frames = [pd.read_parquet(path, columns=cols) for path in files]
    df = pd.concat(frames, ignore_index=True)
    if "ts" in df.columns and not df.empty:
        df = df[(df["ts"] >= start_ms) & (df["ts"] < end_ms)]
    return df.reset_index(drop=True)


def _load_bybit_trades_processed(
    processed_dir: Path,
    exchange: str,
    symbol: str,
    start_ms: int,
    end_ms: int,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    base = processed_dir / "agg_trades" / f"exchange={exchange}" / f"symbol={symbol}"
    if not base.exists():
        return pd.DataFrame()
    files = sorted(base.glob("date=*/part-*.parquet"))
    if files:
        dates = set(_date_range(pd.Timestamp(start_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d"),
                                pd.Timestamp(end_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d")))
        filtered = []
        for path in files:
            date_part = path.parent.name.replace("date=", "")
            if date_part in dates:
                filtered.append(path)
        files = filtered
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(path, columns=columns) if columns else pd.read_parquet(path) for path in files]
    df = pd.concat(frames, ignore_index=True)
    if "ts" in df.columns:
        df = df[(df["ts"] >= start_ms) & (df["ts"] < end_ms)]
    return df.reset_index(drop=True)


def _required_l2_columns(cfg: Dict) -> List[str]:
    levels: set[int] = set()
    ofi_cfg = cfg.get("ofi", {})
    if ofi_cfg.get("enabled", True):
        levels.update(int(x) for x in ofi_cfg.get("levels", []))
    obi_cfg = cfg.get("obi", {})
    if obi_cfg.get("enabled", True):
        levels.update(int(x) for x in obi_cfg.get("levels", []))
    extras = cfg.get("extras", {})
    levels.update(int(x) for x in extras.get("depth_levels", []) or [])
    if extras.get("spread", False):
        levels.add(1)

    levels = {lvl for lvl in levels if lvl > 0}
    if not levels:
        levels = {1}

    cols = ["ts"]
    for lvl in sorted(levels):
        cols.extend(
            [
                f"bid_price_{lvl}",
                f"ask_price_{lvl}",
                f"bid_size_{lvl}",
                f"ask_size_{lvl}",
            ]
        )
    return cols


def _default_micro_config() -> Dict:
    return {
        "ofi": {
            "enabled": True,
            "levels": [1, 5, 10],
            "decay": 0.7,
            "zscore_window": 60,
            "include_raw": False,
            "include_zscore": True,
        },
        "obi": {"enabled": True, "levels": [1, 5, 10], "decay": 0.7},
        "kyle_lambda": {
            "enabled": True,
            "window_minutes": 5,
            "zscore_window_minutes": 60,
            "bucket_seconds": 60,
            "outputs": ["lambda_z", "illiquidity_flag"],
        },
        "vpin": {
            "enabled": True,
            "bucket_mult": 1.0,
            "volume_window_minutes": 60,
            "window_buckets": 20,
            "cdf_window_buckets": 60,
        },
        "extras": {"spread": True, "mid_change": True, "depth_levels": [1, 5, 10]},
        "diagnostics": {
            "ic_method": "spearman",
            "ic_horizons_minutes": [1, 5, 15, 30],
            "quantiles": [0.01, 0.05, 0.5, 0.95, 0.99],
        },
    }


def _feature_missingness(features: pd.DataFrame) -> Dict[str, object]:
    overall = features.isna().mean().to_dict()
    by_day = {}
    if not features.empty:
        for day, df in features.groupby(features.index.date):
            by_day[str(day)] = df.isna().mean().to_dict()
    return {"overall": overall, "by_day": by_day}


def _feature_quantiles(features: pd.DataFrame, quantiles: Iterable[float]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for col in features.columns:
        series = pd.to_numeric(features[col], errors="coerce").astype(float).dropna()
        if series.empty:
            out[col] = {}
            continue
        qvals = series.quantile(list(quantiles))
        out[col] = {f"p{int(q * 100):02d}": float(val) for q, val in qvals.items()}
    return out


def _forward_returns(bars: pd.DataFrame, horizons: Iterable[int], price_col: str) -> Dict[int, pd.Series]:
    if price_col not in bars.columns:
        raise ValueError(f"bars missing price column '{price_col}'")
    price = bars[price_col].astype(float)
    returns = {}
    for h in horizons:
        returns[int(h)] = price.shift(-int(h)) / price - 1.0
    return returns


def _ic_table(
    features: pd.DataFrame,
    forward_returns: Dict[int, pd.Series],
    method: str = "spearman",
) -> Dict[str, Dict[str, Optional[float]]]:
    results: Dict[str, Dict[str, Optional[float]]] = {}
    for horizon, fwd in forward_returns.items():
        ic_vals: Dict[str, Optional[float]] = {}
        for col in features.columns:
            df = pd.concat([features[col], fwd.rename("ret")], axis=1).dropna()
            if df.empty:
                ic_vals[col] = None
            else:
                ic_vals[col] = float(df[col].corr(df["ret"], method=method))
        results[str(horizon)] = ic_vals
    return results


def _top_ic(ic_table: Dict[str, Dict[str, Optional[float]]], top_n: int = 5) -> Dict[str, List[Dict[str, float]]]:
    summary: Dict[str, List[Dict[str, float]]] = {}
    for horizon, vals in ic_table.items():
        pairs = [(feat, val) for feat, val in vals.items() if val is not None and np.isfinite(val)]
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        summary[horizon] = [{"feature": feat, "ic": float(val)} for feat, val in pairs[:top_n]]
    return summary


def _build_markdown(report: Dict[str, object]) -> str:
    lines = [
        "# Microstructure Diagnostics",
        "",
        f"- Exchange: {report.get('exchange')}",
        f"- Symbol: {report.get('symbol')}",
        f"- Window: {report.get('start')} to {report.get('end')}",
        f"- Rows (1m bars): {report.get('rows')}",
        f"- Features: {report.get('features_count')}",
        "",
        "## Missingness (overall)",
    ]
    missing = report.get("missingness", {}).get("overall", {})
    for key, val in sorted(missing.items()):
        lines.append(f"- {key}: {val:.4f}")

    lines.append("")
    lines.append("## IC Summary (Spearman)")
    ic_summary = report.get("ic_summary", {})
    for horizon, rows in ic_summary.items():
        lines.append(f"- Horizon {horizon}m:")
        if not rows:
            lines.append("  - (no data)")
        else:
            for row in rows:
                lines.append(f"  - {row['feature']}: {row['ic']:.4f}")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose microstructure feature quality and IC.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config yaml")
    parser.add_argument("--exchange", required=True, choices=["binance", "deepcoin", "okx", "bybit"])
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--processed-dir", default=None, help="Override processed dir")
    parser.add_argument("--raw-dir", default=None, help="Override raw dir")
    parser.add_argument("--artifacts-dir", default=None, help="Override artifacts dir")
    parser.add_argument("--variant", default="micro_diag", help="Output variant suffix")
    parser.add_argument("--end-inclusive-date", action="store_true", help="Treat end date as inclusive")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    micro_cfg = cfg.get("microstructure") or _default_micro_config()
    cfg_hash = _config_hash(micro_cfg)

    processed_dir = Path(args.processed_dir or cfg["paths"]["processed_dir"])
    raw_dir = Path(args.raw_dir or cfg["paths"]["raw_dir"])
    artifacts_dir = Path(args.artifacts_dir or cfg["paths"]["artifacts_dir"])

    exchange_cfg = None
    for entry in cfg.get("data", {}).get("exchanges", []):
        if entry.get("name") == args.exchange:
            exchange_cfg = entry
            break
    if exchange_cfg is None:
        raise ValueError(f"Exchange '{args.exchange}' not found in config")
    market = exchange_cfg.get("market", "perp")

    start_ms = _parse_date_to_ms(args.start)
    end_ms = _parse_end_to_ms(args.end, args.end_inclusive_date)

    bars_path = processed_dir / "bars_1m.parquet"
    bars_1m = pd.DataFrame()
    if bars_path.exists():
        bars_1m = pd.read_parquet(bars_path)
        bars_1m = _ensure_dt_index(bars_1m)
        bars_1m = bars_1m[(bars_1m.index >= pd.Timestamp(start_ms, unit="ms", tz="UTC")) & (bars_1m.index < pd.Timestamp(end_ms, unit="ms", tz="UTC"))]

    trade_cols = ["ts", "price", "qty", "is_buyer_maker"]
    if args.exchange == "bybit":
        trades = _load_bybit_trades_processed(processed_dir, args.exchange, args.symbol, start_ms, end_ms, columns=trade_cols)
    else:
        trades = _load_raw_dataset(raw_dir, args.exchange, market, args.symbol, "agg_trades", start_ms, end_ms, columns=trade_cols)

    l2_cols = _required_l2_columns(micro_cfg)
    l2 = _load_raw_dataset(raw_dir, args.exchange, market, args.symbol, "book_depth", start_ms, end_ms, columns=l2_cols)

    if bars_1m.empty:
        if trades.empty:
            raise ValueError("bars_1m missing and trades are empty; cannot build 1m bars")
        trades_tmp = trades.copy()
        trades_tmp["timestamp"] = pd.to_datetime(trades_tmp["ts"], unit="ms", utc=True)
        bars_1m = build_time_bars(trades_tmp, l2=None, time_col="timestamp", price_col="price", qty_col="qty")
        bars_1m = _ensure_dt_index(bars_1m)
        bars_1m = bars_1m[(bars_1m.index >= pd.Timestamp(start_ms, unit="ms", tz="UTC")) & (bars_1m.index < pd.Timestamp(end_ms, unit="ms", tz="UTC"))]

    if bars_1m.empty:
        raise ValueError("bars_1m is empty after filtering; check date range")

    micro_features = build_micro_features(bars_1m, l2, trades, micro_cfg, time_col=None, l2_time_col="ts")

    diag_cfg = micro_cfg.get("diagnostics", {})
    price_col = "mid_close" if "mid_close" in bars_1m.columns else "close"
    horizons = diag_cfg.get("ic_horizons_minutes", [1, 5, 15, 30])
    ic_method = str(diag_cfg.get("ic_method", "spearman"))
    quantiles = diag_cfg.get("quantiles", [0.01, 0.05, 0.5, 0.95, 0.99])

    missingness = _feature_missingness(micro_features)
    quantile_table = _feature_quantiles(micro_features, quantiles)
    fwd_returns = _forward_returns(bars_1m, horizons, price_col=price_col)
    ic_table = _ic_table(micro_features, fwd_returns, method=ic_method)
    ic_summary = _top_ic(ic_table, top_n=5)

    timestamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = artifacts_dir / "stage7" / f"{timestamp}_{args.variant}"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "config_hash": cfg_hash,
        "micro_config": micro_cfg,
        "exchange": args.exchange,
        "symbol": args.symbol,
        "start": args.start,
        "end": args.end,
        "rows": int(len(micro_features)),
        "features_count": int(micro_features.shape[1]),
        "data_counts": {
            "bars_1m": int(len(bars_1m)),
            "l2_rows": int(len(l2)),
            "trade_rows": int(len(trades)),
        },
        "missingness": missingness,
        "quantiles": quantile_table,
        "ic_method": ic_method,
        "ic_table": ic_table,
        "ic_summary": ic_summary,
    }
    (out_dir / "micro_diag.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "micro_diag.md").write_text(_build_markdown(report), encoding="utf-8")

    micro_features.reset_index().rename(columns={"index": "timestamp"}).to_parquet(
        out_dir / "micro_features.parquet", index=False
    )

    print(f"microstructure diagnostics written to: {out_dir}")


if __name__ == "__main__":
    main()
