from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.bars.time_bars import resample_time_bars
from src.data.binance import BinanceFuturesRest, BinanceVision
from src.data.deepcoin import DeepcoinRest
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


def _get_depth_levels(cfg: dict, exchange: str, symbol: str) -> int:
    for ex in cfg.get("data", {}).get("exchanges", []):
        if ex.get("name") == exchange and (not ex.get("symbols") or symbol in ex.get("symbols", [])):
            return int(ex.get("depth_levels", 10))
    return 10


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
) -> None:
    raw_dir = Path(cfg["paths"]["raw_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    start_ms = _parse_date_to_ms(start)
    end_ms = _parse_date_to_ms(end)
    max_gap_ms = int(cfg["data"]["quality"]["max_gap_seconds"]) * 1000

    dataset_list = _parse_datasets_arg(datasets)
    if not dataset_list:
        dataset_list = ["klines_1m", "agg_trades"]

    klines = None

    if exchange == "binance":
        market = "usdm"
        depth_levels = _get_depth_levels(cfg, exchange, symbol)
        if source == "vision":
            local_dir = Path(vision_dir) if vision_dir else Path("data/vision")
            client = BinanceVision(local_dir=local_dir, auto_download=True)
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
    else:
        raise ValueError("exchange must be 'binance' or 'deepcoin'")

    if build_bars:
        if "klines_1m" not in dataset_list or klines is None or klines.empty:
            raise ValueError("klines_1m required to build bars")
        bars_1m = _build_bars_from_klines(klines)
        bars_1m.to_parquet(processed_dir / "bars_1m.parquet", index=False)
        bars_5m = resample_time_bars(bars_1m.set_index("timestamp"), freq="5min")
        bars_5m.to_parquet(processed_dir / "bars_5m.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and build datasets.")
    parser.add_argument("--config", required=True, help="Path to config yaml")
    parser.add_argument("--exchange", required=True, choices=["binance", "deepcoin"])
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--source", default="rest", choices=["rest", "vision"])
    parser.add_argument("--datasets", default=None, help="Comma-separated datasets")
    parser.add_argument("--mode", default="rest", choices=["rest", "ws_l2"])
    parser.add_argument("--vision-dir", default=None, help="Local directory for Binance Vision data")
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
    )


if __name__ == "__main__":
    main()
