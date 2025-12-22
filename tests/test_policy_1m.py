from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.strategy.policy_1m import evaluate_hazard_policy


def _config():
    return {
        "hazard": {"feature_window_minutes": 3},
        "policy": {
            "exit": {"enabled": True, "hazard_threshold": 0.7, "consecutive_minutes": 2, "require_rising": True},
            "fail_fast": {"enabled": True, "hazard_threshold": 0.85, "confirm_signals": {}},
            "add_risk": {"enabled": True, "hazard_max_to_add": 0.2},
        },
    }


def test_consecutive_and_require_rising():
    cfg = _config()
    times = pd.date_range("2025-01-01 00:00", periods=3, freq="1min")
    p_series = pd.Series([0.6, 0.8, 0.85], index=times)

    exit_ts, reason, _ = evaluate_hazard_policy(p_series, None, cfg, mode="hazard_exit_only")
    assert exit_ts == times[2]
    assert reason == "hazard_exit"


def test_require_rising_blocks_exit():
    cfg = _config()
    times = pd.date_range("2025-01-01 00:00", periods=3, freq="1min")
    p_series = pd.Series([0.8, 0.8, 0.8], index=times)

    exit_ts, reason, _ = evaluate_hazard_policy(p_series, None, cfg, mode="hazard_exit_only")
    assert exit_ts is None
    assert reason is None


def test_fail_fast_triggers_immediately():
    cfg = _config()
    times = pd.date_range("2025-01-01 00:00", periods=3, freq="1min")
    p_series = pd.Series([0.6, 0.9, 0.7], index=times)

    exit_ts, reason, _ = evaluate_hazard_policy(p_series, None, cfg, mode="full_policy")
    assert exit_ts == times[1]
    assert reason == "hazard_fail_fast"
