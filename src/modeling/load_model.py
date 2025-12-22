from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


def load_hazard_model(path: Path) -> Dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_hazard_proba(model_pack: Dict, features: pd.DataFrame) -> pd.Series:
    feature_names = model_pack.get("features", [])
    missing = [c for c in feature_names if c not in features.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    x = features[feature_names].astype(float).to_numpy()
    mean = model_pack.get("mean")
    std = model_pack.get("std")
    clip = model_pack.get("clip_zscore")

    if mean is not None and std is not None:
        mean = np.asarray(mean)
        std = np.asarray(std)
        std = np.where(std == 0, 1.0, std)
        x = (x - mean) / std
        if clip is not None:
            x = np.clip(x, -float(clip), float(clip))

    w = model_pack.get("model")
    if isinstance(w, np.ndarray):
        xb = np.hstack([np.ones((x.shape[0], 1)), x])
        p = 1.0 / (1.0 + np.exp(-(xb @ w)))
    elif hasattr(w, "predict_proba"):
        p = w.predict_proba(x)[:, 1]
    else:
        raise ValueError("Unsupported model type in hazard model pack")

    calibrator = model_pack.get("calibrator")
    if calibrator is not None:
        p = calibrator.predict(p)

    return pd.Series(p, index=features.index)
