from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Awaitable, Callable, Optional, Union

import websockets


@dataclass
class TradeEvent:
    ts: datetime
    price: float
    qty: float


@dataclass
class BookEvent:
    ts: datetime
    bid: float
    ask: float
    bid_size: float
    ask_size: float


Event = Union[TradeEvent, BookEvent]


def _to_ts_ms(value: str | int) -> datetime:
    return datetime.fromtimestamp(int(value) / 1000.0, tz=timezone.utc)


async def _dispatch(handler: Callable[[Event], Awaitable[None]] | Callable[[Event], None], event: Event) -> None:
    if asyncio.iscoroutinefunction(handler):
        await handler(event)  # type: ignore[arg-type]
    else:
        handler(event)


async def stream_binance(
    symbol: str,
    handler: Callable[[Event], Awaitable[None]] | Callable[[Event], None],
    stop_event: asyncio.Event,
    reconnect_delay: float = 5.0,
) -> None:
    stream_symbol = symbol.lower()
    stream = f"{stream_symbol}@trade/{stream_symbol}@bookTicker"
    url = f"wss://fstream.binance.com/stream?streams={stream}"

    while not stop_event.is_set():
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                async for message in ws:
                    if stop_event.is_set():
                        return
                    payload = json.loads(message)
                    data = payload.get("data", {})
                    stream_name = payload.get("stream", "")
                    if stream_name.endswith("@trade"):
                        ts = _to_ts_ms(data.get("T", 0))
                        price = float(data.get("p", 0.0))
                        qty = float(data.get("q", 0.0))
                        await _dispatch(handler, TradeEvent(ts=ts, price=price, qty=qty))
                    elif stream_name.endswith("@bookTicker"):
                        ts = _to_ts_ms(data.get("E", 0))
                        bid = float(data.get("b", 0.0))
                        ask = float(data.get("a", 0.0))
                        bid_size = float(data.get("B", 0.0))
                        ask_size = float(data.get("A", 0.0))
                        await _dispatch(handler, BookEvent(ts=ts, bid=bid, ask=ask, bid_size=bid_size, ask_size=ask_size))
        except Exception:
            if stop_event.is_set():
                return
            await asyncio.sleep(reconnect_delay)


async def stream_okx(
    symbol: str,
    handler: Callable[[Event], Awaitable[None]] | Callable[[Event], None],
    stop_event: asyncio.Event,
    reconnect_delay: float = 5.0,
) -> None:
    url = "wss://ws.okx.com:8443/ws/v5/public"
    subscribe = {
        "op": "subscribe",
        "args": [
            {"channel": "trades", "instId": symbol},
            {"channel": "books5", "instId": symbol},
        ],
    }

    while not stop_event.is_set():
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                await ws.send(json.dumps(subscribe))
                async for message in ws:
                    if stop_event.is_set():
                        return
                    payload = json.loads(message)
                    if "event" in payload:
                        continue
                    arg = payload.get("arg", {})
                    channel = arg.get("channel")
                    data = payload.get("data", [])
                    if not data:
                        continue
                    if channel == "trades":
                        for trade in data:
                            ts = _to_ts_ms(trade.get("ts", 0))
                            price = float(trade.get("px", 0.0))
                            qty = float(trade.get("sz", 0.0))
                            await _dispatch(handler, TradeEvent(ts=ts, price=price, qty=qty))
                    elif channel == "books5":
                        book = data[0]
                        ts = _to_ts_ms(book.get("ts", 0))
                        bids = book.get("bids", [])
                        asks = book.get("asks", [])
                        if bids and asks:
                            bid = float(bids[0][0])
                            bid_size = float(bids[0][1])
                            ask = float(asks[0][0])
                            ask_size = float(asks[0][1])
                            await _dispatch(
                                handler,
                                BookEvent(ts=ts, bid=bid, ask=ask, bid_size=bid_size, ask_size=ask_size),
                            )
        except Exception:
            if stop_event.is_set():
                return
            await asyncio.sleep(reconnect_delay)
