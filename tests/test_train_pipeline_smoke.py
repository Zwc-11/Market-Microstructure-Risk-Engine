from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modeling.train_hazard import train_hazard_model


def _roc_auc_score(y_true, p_hat):
    y_true = np.asarray(y_true, dtype=int)
    pos = y_true.sum()
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return 0.0
    order = np.argsort(p_hat)
    y_sorted = y_true[order]
    cum_neg = np.cumsum(1 - y_sorted)
    return float(cum_neg[y_sorted == 1].sum() / (pos * neg))


def _config():
    return {
        "paths": {"artifacts_dir": "artifacts"},
        "hazard": {"horizon_minutes": 5},
        "model": {
            "random_state": 42,
            "preprocessing": {"standardize": True, "clip_zscore": 10.0},
            "training": {
                "class_weight": "balanced",
                "calibration": {"enabled": True, "method": "sigmoid"},
                "validation": {
                    "embargo_minutes": 0,
                    "splits": {"train_days": 1, "test_days": 1, "step_days": 1},
                },
            },
        },
    }


def _hazard_data():
    times = pd.date_range("2025-01-01", periods=96, freq="1h")
    y = (np.arange(len(times)) % 2).astype(int)
    hazard = pd.DataFrame(
        {
            "event_id": [f"evt_{i}" for i in range(len(times))],
            "t": times,
            "y": y,
            "horizon_end_ts": times + pd.Timedelta(minutes=5),
        }
    )
    features = pd.DataFrame({"timestamp": times, "signal": y.astype(float)})
    return hazard, features


def test_train_pipeline_smoke(tmp_path):
    hazard, features = _hazard_data()
    cfg = _config()

    oof, report = train_hazard_model(hazard, features, cfg, output_dir=tmp_path)

    assert (tmp_path / "hazard_oof_predictions.parquet").exists()
    assert (tmp_path / "hazard_report.json").exists()
    assert (tmp_path / "models" / "hazard_model.pkl").exists()

    oof_disk = pd.read_parquet(tmp_path / "hazard_oof_predictions.parquet")
    assert not oof_disk.empty
    assert oof_disk["p_hat"].between(0.0, 1.0).all()

    auc = _roc_auc_score(oof_disk["y"], oof_disk["p_hat"])
    assert auc > 0.9

    oof2, report2 = train_hazard_model(hazard, features, cfg, output_dir=tmp_path / "run2")
    pd.testing.assert_frame_equal(oof, oof2)
    for key in ("auc", "pr_auc", "brier", "lift_top_decile"):
        assert report["overall"][key] == report2["overall"][key]
