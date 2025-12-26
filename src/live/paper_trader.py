from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional


@dataclass
class Position:
    side: str
    entry_ts: datetime
    entry_price: float
    qty: float
    notional: float
    reasons: List[str] = field(default_factory=list)


@dataclass
class TradeResult:
    trade_id: int
    side: str
    entry_ts: datetime
    entry_price: float
    exit_ts: datetime
    exit_price: float
    qty: float
    notional: float
    pnl_gross: float
    pnl_net: float
    fees: float
    hold_minutes: float
    entry_reasons: List[str]
    exit_reason: str
    equity_after: float


def _floor_minute(ts: datetime) -> datetime:
    return ts.replace(second=0, microsecond=0, tzinfo=timezone.utc)


class PaperTrader:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.cash = float(cfg.get("initial_balance", 1000.0))
        self.position: Optional[Position] = None
        self.trade_id = 0

        self.last_minute: Optional[datetime] = None
        self.prev_minute_mid: Optional[float] = None
        self.last_mid: Optional[float] = None
        self.last_obi: Optional[float] = None
        self.last_trade_minute: Optional[datetime] = None

    def update_book(self, ts: datetime, bid: float, ask: float, bid_size: float, ask_size: float) -> None:
        mid = (bid + ask) / 2.0
        self.last_mid = mid
        denom = bid_size + ask_size
        if denom > 0:
            self.last_obi = (bid_size - ask_size) / denom

    def update_trade(self, ts: datetime, price: float, qty: float) -> Optional[TradeResult]:
        ts = ts.astimezone(timezone.utc)
        minute = _floor_minute(ts)
        if self.last_minute is None:
            self.last_minute = minute
            if self.last_mid is None:
                self.last_mid = price
            return None

        trade_result = None
        if minute > self.last_minute:
            trade_result = self._on_minute_close(self.last_minute + timedelta(minutes=1))
            self.last_minute = minute

        if self.last_mid is None:
            self.last_mid = price
        return trade_result

    def _on_minute_close(self, close_ts: datetime) -> Optional[TradeResult]:
        if self.last_mid is None:
            return None
        if self.prev_minute_mid is None:
            self.prev_minute_mid = self.last_mid
            return None

        ret_1m = (self.last_mid / self.prev_minute_mid) - 1.0
        self.prev_minute_mid = self.last_mid

        trade_result = self._maybe_exit(close_ts)
        if trade_result is not None:
            return trade_result
        return self._maybe_entry(close_ts, ret_1m)

    def _maybe_entry(self, ts: datetime, ret_1m: float) -> Optional[TradeResult]:
        if self.position is not None:
            return None

        cooldown = int(self.cfg.get("cooldown_minutes", 0))
        if self.last_trade_minute is not None and cooldown > 0:
            if ts < self.last_trade_minute + timedelta(minutes=cooldown):
                return None

        ret_thr = float(self.cfg.get("ret_1m_thr", 0.001))
        obi_thr = float(self.cfg.get("obi_thr", 0.1))
        allow_long = bool(self.cfg.get("allow_long", True))
        allow_short = bool(self.cfg.get("allow_short", True))

        reasons: List[str] = []
        side: Optional[str] = None

        if allow_long and ret_1m > ret_thr:
            reasons.append("ret_1m_gt_thr")
        if allow_short and ret_1m < -ret_thr:
            reasons.append("ret_1m_lt_neg_thr")

        obi = self.last_obi
        if obi is not None and obi > obi_thr and allow_long:
            reasons.append("obi_gt_thr")
        if obi is not None and obi < -obi_thr and allow_short:
            reasons.append("obi_lt_neg_thr")

        if allow_long and "ret_1m_gt_thr" in reasons and "obi_gt_thr" in reasons:
            side = "long"
        if allow_short and "ret_1m_lt_neg_thr" in reasons and "obi_lt_neg_thr" in reasons:
            side = "short"

        if side is None:
            return None

        entry_price = float(self.last_mid)
        notional = self.cash * float(self.cfg.get("position_pct", 0.1))
        notional = max(0.0, min(notional, self.cash))
        if notional <= 0:
            return None

        qty = notional / entry_price if entry_price > 0 else 0.0
        if qty <= 0:
            return None

        fee_bps = float(self.cfg.get("fee_bps", 0.0))
        fees = notional * fee_bps / 1.0e4
        self.cash -= fees

        self.position = Position(
            side=side,
            entry_ts=ts,
            entry_price=entry_price,
            qty=qty,
            notional=notional,
            reasons=reasons,
        )
        self.last_trade_minute = ts
        return None

    def _maybe_exit(self, ts: datetime) -> Optional[TradeResult]:
        if self.position is None:
            return None
        if self.last_mid is None:
            return None

        side = self.position.side
        entry_price = self.position.entry_price
        price = float(self.last_mid)

        tp_pct = float(self.cfg.get("take_profit_pct", 0.003))
        sl_pct = float(self.cfg.get("stop_loss_pct", 0.003))
        max_hold = int(self.cfg.get("max_hold_minutes", 30))

        exit_reason = None
        if side == "long":
            if price >= entry_price * (1.0 + tp_pct):
                exit_reason = "take_profit"
            elif price <= entry_price * (1.0 - sl_pct):
                exit_reason = "stop_loss"
        else:
            if price <= entry_price * (1.0 - tp_pct):
                exit_reason = "take_profit"
            elif price >= entry_price * (1.0 + sl_pct):
                exit_reason = "stop_loss"

        hold_minutes = (ts - self.position.entry_ts).total_seconds() / 60.0
        if exit_reason is None and hold_minutes >= max_hold:
            exit_reason = "time_exit"

        if exit_reason is None:
            return None

        trade = self._close_position(ts, price, exit_reason)
        self.last_trade_minute = ts
        return trade

    def _close_position(self, ts: datetime, price: float, reason: str) -> TradeResult:
        pos = self.position
        if pos is None:
            raise RuntimeError("No position to close")

        self.trade_id += 1
        if pos.side == "long":
            pnl_gross = (price - pos.entry_price) * pos.qty
        else:
            pnl_gross = (pos.entry_price - price) * pos.qty

        notional_exit = pos.qty * price
        fee_bps = float(self.cfg.get("fee_bps", 0.0))
        fees = (pos.notional + notional_exit) * fee_bps / 1.0e4
        pnl_net = pnl_gross - fees

        self.cash += pos.notional + pnl_net
        hold_minutes = (ts - pos.entry_ts).total_seconds() / 60.0

        result = TradeResult(
            trade_id=self.trade_id,
            side=pos.side,
            entry_ts=pos.entry_ts,
            entry_price=pos.entry_price,
            exit_ts=ts,
            exit_price=price,
            qty=pos.qty,
            notional=pos.notional,
            pnl_gross=pnl_gross,
            pnl_net=pnl_net,
            fees=fees,
            hold_minutes=hold_minutes,
            entry_reasons=pos.reasons,
            exit_reason=reason,
            equity_after=self.cash,
        )
        self.position = None
        return result

    def status(self) -> Dict[str, object]:
        equity = self.cash
        if self.position is not None and self.last_mid is not None:
            if self.position.side == "long":
                equity += self.position.qty * self.last_mid
            else:
                equity += self.position.notional + (self.position.entry_price - self.last_mid) * self.position.qty

        return {
            "cash": self.cash,
            "equity": equity,
            "position": None
            if self.position is None
            else {
                "side": self.position.side,
                "entry_ts": self.position.entry_ts.isoformat(),
                "entry_price": self.position.entry_price,
                "qty": self.position.qty,
                "notional": self.position.notional,
                "reasons": self.position.reasons,
            },
            "last_mid": self.last_mid,
            "last_obi": self.last_obi,
        }
