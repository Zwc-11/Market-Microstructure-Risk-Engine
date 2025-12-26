from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live.paper_trader import PaperTrader


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def test_paper_trader_entry_exit():
    cfg = {
        "initial_balance": 1000.0,
        "ret_1m_thr": 0.005,
        "obi_thr": 0.1,
        "take_profit_pct": 0.005,
        "stop_loss_pct": 0.02,
        "max_hold_minutes": 5,
        "position_pct": 0.5,
        "fee_bps": 0.0,
    }
    trader = PaperTrader(cfg)

    trader.update_book(_ts("2025-01-01T00:00:10"), 99.0, 101.0, 10.0, 5.0)
    trader.update_trade(_ts("2025-01-01T00:00:10"), 100.0, 1.0)
    trader.update_book(_ts("2025-01-01T00:00:50"), 100.0, 102.0, 10.0, 5.0)

    trader.update_trade(_ts("2025-01-01T00:01:05"), 101.0, 1.0)
    assert trader.position is None

    trader.update_book(_ts("2025-01-01T00:01:50"), 101.0, 103.0, 10.0, 5.0)
    trader.update_trade(_ts("2025-01-01T00:02:05"), 102.0, 1.0)
    assert trader.position is not None
    assert trader.position.side == "long"
    assert "ret_1m_gt_thr" in trader.position.reasons
    assert "obi_gt_thr" in trader.position.reasons

    trader.update_book(_ts("2025-01-01T00:02:50"), 102.0, 104.0, 10.0, 5.0)
    result = trader.update_trade(_ts("2025-01-01T00:03:05"), 103.0, 1.0)

    assert result is not None
    assert result.exit_reason == "take_profit"
    assert np.isclose(result.hold_minutes, 1.0, atol=0.01)
