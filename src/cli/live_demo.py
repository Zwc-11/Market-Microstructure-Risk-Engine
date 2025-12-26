from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.live.paper_trader import PaperTrader, TradeResult
from src.live.stream import BookEvent, TradeEvent, stream_binance, stream_okx


class LiveLogger:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.out_dir / "events.jsonl"
        self.trades_path = self.out_dir / "trades.csv"
        self.status_path = self.out_dir / "status.json"
        self._csv_header_written = self.trades_path.exists()

    def log_event(self, payload: dict) -> None:
        payload["logged_at"] = datetime.now(timezone.utc).isoformat()
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def log_entry(self, trader: PaperTrader) -> None:
        pos = trader.position
        if pos is None:
            return
        payload = {
            "type": "entry",
            "ts": pos.entry_ts.isoformat(),
            "side": pos.side,
            "entry_price": pos.entry_price,
            "qty": pos.qty,
            "notional": pos.notional,
            "reasons": pos.reasons,
        }
        self.log_event(payload)

    def log_exit(self, result: TradeResult) -> None:
        payload = {"type": "exit", **asdict(result)}
        payload["entry_ts"] = result.entry_ts.isoformat()
        payload["exit_ts"] = result.exit_ts.isoformat()
        self.log_event(payload)
        self._write_trade_csv(result)

    def _write_trade_csv(self, result: TradeResult) -> None:
        row = {
            "trade_id": result.trade_id,
            "side": result.side,
            "entry_ts": result.entry_ts.isoformat(),
            "entry_price": result.entry_price,
            "exit_ts": result.exit_ts.isoformat(),
            "exit_price": result.exit_price,
            "qty": result.qty,
            "notional": result.notional,
            "pnl_gross": result.pnl_gross,
            "pnl_net": result.pnl_net,
            "fees": result.fees,
            "hold_minutes": result.hold_minutes,
            "entry_reasons": "|".join(result.entry_reasons),
            "exit_reason": result.exit_reason,
            "equity_after": result.equity_after,
        }
        with self.trades_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not self._csv_header_written:
                writer.writeheader()
                self._csv_header_written = True
            writer.writerow(row)

    def write_status(self, status: dict) -> None:
        self.status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")


async def run_live(args: argparse.Namespace) -> None:
    cfg = {
        "initial_balance": args.initial_balance,
        "ret_1m_thr": args.ret_1m_thr,
        "obi_thr": args.obi_thr,
        "take_profit_pct": args.take_profit_pct,
        "stop_loss_pct": args.stop_loss_pct,
        "max_hold_minutes": args.max_hold_minutes,
        "position_pct": args.position_pct,
        "fee_bps": args.fee_bps,
        "cooldown_minutes": args.cooldown_minutes,
        "allow_long": not args.disable_long,
        "allow_short": not args.disable_short,
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / f"{timestamp}_{args.exchange}_{args.symbol}"
    logger = LiveLogger(out_dir)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    trader = PaperTrader(cfg)
    stop_event = asyncio.Event()
    prev_has_position = False
    last_status = 0.0
    start_time = time.monotonic()

    def _maybe_stop() -> None:
        if args.run_minutes <= 0:
            return
        if (time.monotonic() - start_time) >= args.run_minutes * 60.0:
            stop_event.set()

    async def handler(event) -> None:
        nonlocal prev_has_position, last_status
        if isinstance(event, BookEvent):
            trader.update_book(event.ts, event.bid, event.ask, event.bid_size, event.ask_size)
        elif isinstance(event, TradeEvent):
            result = trader.update_trade(event.ts, event.price, event.qty)
            if not prev_has_position and trader.position is not None:
                logger.log_entry(trader)
                print(f"[ENTRY] {trader.position.side} @ {trader.position.entry_price:.4f} {trader.position.reasons}")
            if result is not None:
                logger.log_exit(result)
                print(f"[EXIT] {result.exit_reason} pnl_net={result.pnl_net:.4f} hold={result.hold_minutes:.2f}m")
            prev_has_position = trader.position is not None

        now = time.monotonic()
        if now - last_status >= args.status_interval:
            status = trader.status()
            status["timestamp"] = datetime.now(timezone.utc).isoformat()
            logger.write_status(status)
            last_status = now
        _maybe_stop()

    try:
        if args.exchange == "binance":
            await stream_binance(args.symbol, handler, stop_event)
        else:
            await stream_okx(args.symbol, handler, stop_event)
    except asyncio.CancelledError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Live demo paper-trading with WS streams.")
    parser.add_argument("--exchange", choices=["binance", "okx"], required=True)
    parser.add_argument("--symbol", required=True, help="Binance: BTCUSDT | OKX: BTC-USDT-SWAP")
    parser.add_argument("--out-dir", default="artifacts/live_demo", help="Output directory")
    parser.add_argument("--initial-balance", type=float, default=1000.0)
    parser.add_argument("--position-pct", type=float, default=0.25)
    parser.add_argument("--ret-1m-thr", type=float, default=0.001)
    parser.add_argument("--obi-thr", type=float, default=0.1)
    parser.add_argument("--take-profit-pct", type=float, default=0.003)
    parser.add_argument("--stop-loss-pct", type=float, default=0.003)
    parser.add_argument("--max-hold-minutes", type=int, default=30)
    parser.add_argument("--cooldown-minutes", type=int, default=2)
    parser.add_argument("--fee-bps", type=float, default=0.0)
    parser.add_argument("--disable-long", action="store_true")
    parser.add_argument("--disable-short", action="store_true")
    parser.add_argument("--run-minutes", type=int, default=0, help="Stop after N minutes (0 = run forever)")
    parser.add_argument("--status-interval", type=int, default=60)
    args = parser.parse_args()

    asyncio.run(run_live(args))


if __name__ == "__main__":
    main()
