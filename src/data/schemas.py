from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


@dataclass(frozen=True)
class Schema:
    name: str
    columns: Dict[str, str]
    primary_key: List[str]


L2_LEVELS = 10


def _depth_columns(levels: int) -> Dict[str, str]:
    cols: Dict[str, str] = {}
    for lvl in range(1, levels + 1):
        cols[f"bid_price_{lvl}"] = "float64"
        cols[f"bid_size_{lvl}"] = "float64"
        cols[f"ask_price_{lvl}"] = "float64"
        cols[f"ask_size_{lvl}"] = "float64"
    return cols


_KLINE_COLUMNS = {
    "ts": "int64",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
    "close_ts": "int64",
    "quote_volume": "float64",
    "trade_count": "int64",
    "taker_buy_base": "float64",
    "taker_buy_quote": "float64",
    "symbol": "string",
}


SCHEMAS: Dict[str, Schema] = {
    "klines_1m": Schema(
        name="klines_1m",
        columns=_KLINE_COLUMNS,
        primary_key=["symbol", "ts"],
    ),
    "agg_trades": Schema(
        name="agg_trades",
        columns={
            "agg_id": "int64",
            "ts": "int64",
            "price": "float64",
            "qty": "float64",
            "is_buyer_maker": "bool",
            "symbol": "string",
        },
        primary_key=["symbol", "agg_id"],
    ),
    "book_ticker": Schema(
        name="book_ticker",
        columns={
            "ts": "int64",
            "update_id": "int64",
            "bid_price": "float64",
            "bid_qty": "float64",
            "ask_price": "float64",
            "ask_qty": "float64",
            "symbol": "string",
        },
        primary_key=["symbol", "update_id"],
    ),
    "book_depth": Schema(
        name="book_depth",
        columns={
            "ts": "int64",
            "update_id": "int64",
            **_depth_columns(L2_LEVELS),
            "symbol": "string",
        },
        primary_key=["symbol", "update_id"],
    ),
    "premium_kline": Schema(
        name="premium_kline",
        columns=_KLINE_COLUMNS,
        primary_key=["symbol", "ts"],
    ),
    "mark_kline": Schema(
        name="mark_kline",
        columns=_KLINE_COLUMNS,
        primary_key=["symbol", "ts"],
    ),
    "l2_snapshots": Schema(
        name="l2_snapshots",
        columns={
            "ts": "int64",
            **_depth_columns(L2_LEVELS),
            "symbol": "string",
        },
        primary_key=["symbol", "ts"],
    ),
}


def enforce_schema(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    missing = [c for c in schema.columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {schema.name}: {missing}")

    out = df.copy()
    for col, dtype in schema.columns.items():
        if dtype == "string":
            out[col] = out[col].astype("string")
        else:
            out[col] = out[col].astype(dtype)

    return out[list(schema.columns.keys())]
