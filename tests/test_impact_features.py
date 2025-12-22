from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.impact import compute_kyle_lambda, kyle_lambda_features


def _bars():
    times = pd.date_range("2025-01-01 00:00", periods=6, freq="1min")
    return pd.DataFrame({"timestamp": times, "mid_close": [100, 101, 103, 106, 110, 115]})


def test_kyle_lambda_positive_and_deterministic():
    bars = _bars()
    flow = pd.Series([1, 2, 3, 4, 5, 6], index=pd.to_datetime(bars["timestamp"]))

    feats = compute_kyle_lambda(bars, flow, window=3)
    assert feats["lambda"].iloc[-1] > 0
    assert feats["r2"].iloc[-1] >= 0

    feats2 = compute_kyle_lambda(bars, flow, window=3)
    pd.testing.assert_frame_equal(feats, feats2)


def test_kyle_lambda_separate_up_down_and_no_lookahead():
    bars = _bars()
    flow = pd.Series([1, 2, 3, 4, 5, 6], index=pd.to_datetime(bars["timestamp"]))
    cfg = {"window_minutes": 3, "separate_up_down": True, "outputs": ["lambda", "r2", "resid_std", "lambda_up", "lambda_down"]}

    full = kyle_lambda_features(bars, flow, cfg)
    trunc = kyle_lambda_features(bars.iloc[:4], flow.iloc[:4], cfg)

    pd.testing.assert_frame_equal(full.iloc[:4], trunc)
    assert full["lambda_up"].iloc[-1] > 0
