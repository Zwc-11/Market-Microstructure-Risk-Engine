from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.backtest.walkforward import generate_walkforward_folds


def _ensure_timestamp_column(df: pd.DataFrame, column: str = "t") -> pd.DataFrame:
    if column in df.columns:
        out = df.copy()
        out[column] = pd.to_datetime(out[column])
        return out
    if isinstance(df.index, pd.DatetimeIndex):
        out = df.copy()
        out[column] = df.index
        return out
    if "timestamp" in df.columns:
        out = df.copy()
        out[column] = pd.to_datetime(out["timestamp"])
        return out
    if "ts" in df.columns:
        out = df.copy()
        out[column] = pd.to_datetime(out["ts"])
        return out
    raise ValueError("Data must include a datetime column ('t', 'timestamp', or 'ts') or a DatetimeIndex.")


def _prepare_features(
    hazard_df: pd.DataFrame, features_df: pd.DataFrame
) -> Tuple[pd.DataFrame, List[str], int]:
    hazard = _ensure_timestamp_column(hazard_df, "t").sort_values("t")
    features = _ensure_timestamp_column(features_df, "t").sort_values("t")

    feature_cols = [c for c in features.columns if c not in {"t", "timestamp", "ts"}]
    merged = pd.merge_asof(hazard, features[["t"] + feature_cols], on="t", direction="backward")

    before = len(merged)
    merged = merged.dropna(subset=["y"])
    dropped_y = before - len(merged)

    merged = merged.dropna(subset=feature_cols)
    return merged, feature_cols, dropped_y


def _standardize(
    x_train: np.ndarray,
    x_test: np.ndarray,
    clip_zscore: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0, ddof=0)
    std = np.where(std == 0, 1.0, std)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    if clip_zscore is not None:
        clip = float(clip_zscore)
        x_train = np.clip(x_train, -clip, clip)
        x_test = np.clip(x_test, -clip, clip)

    return x_train, x_test, mean, std


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _roc_auc_score(y_true: np.ndarray, p_hat: np.ndarray) -> float:
    y_true = y_true.astype(int)
    pos = y_true.sum()
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(p_hat)
    y_sorted = y_true[order]
    cum_neg = np.cumsum(1 - y_sorted)
    auc = (cum_neg[y_sorted == 1].sum()) / float(pos * neg)
    return float(auc)


def _average_precision(y_true: np.ndarray, p_hat: np.ndarray) -> float:
    y_true = y_true.astype(int)
    pos = y_true.sum()
    if pos == 0:
        return float("nan")
    order = np.argsort(-p_hat)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / pos
    return float(np.sum(precision * np.diff(np.concatenate(([0.0], recall)))))


def _brier_score(y_true: np.ndarray, p_hat: np.ndarray) -> float:
    return float(np.mean((p_hat - y_true) ** 2))


def _reliability_bins(y_true: np.ndarray, p_hat: np.ndarray, bins: int = 10) -> List[Dict]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.digitize(p_hat, edges, right=True) - 1
    rows = []
    for i in range(bins):
        mask = bin_ids == i
        if mask.any():
            rows.append(
                {
                    "bin": i,
                    "count": int(mask.sum()),
                    "mean_pred": float(np.mean(p_hat[mask])),
                    "mean_obs": float(np.mean(y_true[mask])),
                }
            )
        else:
            rows.append({"bin": i, "count": 0, "mean_pred": float("nan"), "mean_obs": float("nan")})
    return rows


def _lift_top_decile(y_true: np.ndarray, p_hat: np.ndarray) -> float:
    if len(p_hat) == 0:
        return 0.0
    overall = float(np.mean(y_true))
    if overall == 0:
        return 0.0
    threshold = float(np.quantile(p_hat, 0.9))
    top = y_true[p_hat >= threshold]
    if len(top) == 0:
        return 0.0
    return float(np.mean(top) / overall)


def _class_weights(y_true: np.ndarray) -> np.ndarray:
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    if pos == 0 or neg == 0:
        return np.ones_like(y_true, dtype=float)
    w_pos = len(y_true) / (2.0 * pos)
    w_neg = len(y_true) / (2.0 * neg)
    return np.where(y_true == 1, w_pos, w_neg)


def _fit_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    cfg: Dict,
    max_iter: int = 500,
    lr: float = 0.1,
) -> np.ndarray:
    x = np.hstack([np.ones((x_train.shape[0], 1)), x_train])
    w = np.zeros(x.shape[1])
    use_balanced = cfg["model"]["training"].get("class_weight") == "balanced"
    weights = _class_weights(y_train) if use_balanced else np.ones_like(y_train, dtype=float)

    for _ in range(max_iter):
        p = _sigmoid(x @ w)
        grad = (x.T @ ((p - y_train) * weights)) / x.shape[0]
        w = w - lr * grad
    return w


class _SigmoidCalibrator:
    def __init__(self, coef: np.ndarray):
        self.coef = coef

    def predict(self, scores: np.ndarray) -> np.ndarray:
        x = np.vstack([np.ones(len(scores)), scores]).T
        return _sigmoid(x @ self.coef)


class _IsotonicCalibrator:
    def __init__(self, thresholds: np.ndarray, values: np.ndarray):
        self.thresholds = thresholds
        self.values = values

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return np.interp(scores, self.thresholds, self.values, left=self.values[0], right=self.values[-1])


def _fit_sigmoid_calibrator(scores: np.ndarray, y_true: np.ndarray) -> _SigmoidCalibrator:
    x = scores.reshape(-1, 1)
    coef = _fit_logistic(x, y_true, {"model": {"training": {"class_weight": "balanced"}}}, max_iter=300, lr=0.2)
    return _SigmoidCalibrator(coef)


def _fit_isotonic_calibrator(scores: np.ndarray, y_true: np.ndarray) -> _IsotonicCalibrator:
    order = np.argsort(scores)
    s = scores[order]
    y = y_true[order].astype(float)

    weights = np.ones_like(y)
    v = y.copy()
    w = weights.copy()
    idx = list(range(len(v)))

    i = 0
    while i < len(v) - 1:
        if v[i] > v[i + 1]:
            total_w = w[i] + w[i + 1]
            avg = (v[i] * w[i] + v[i + 1] * w[i + 1]) / total_w
            v[i] = avg
            w[i] = total_w
            del v[i + 1]
            del w[i + 1]
            del idx[i + 1]
            if i > 0:
                i -= 1
        else:
            i += 1

    thresholds = []
    values = []
    start = 0
    for block_idx, block_value in zip(idx, v):
        end = block_idx + 1
        thresholds.append(s[start:end].max())
        values.append(block_value)
        start = end

    thresholds = np.array(thresholds, dtype=float)
    values = np.array(values, dtype=float)
    return _IsotonicCalibrator(thresholds, values)


def _predict_with_model(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    xb = np.hstack([np.ones((x.shape[0], 1)), x])
    return _sigmoid(xb @ w)


def train_hazard_model(
    hazard_df: pd.DataFrame,
    features_df: pd.DataFrame,
    config: Dict,
    output_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict]:
    merged, feature_cols, dropped_y = _prepare_features(hazard_df, features_df)

    merged["t"] = pd.to_datetime(merged["t"])
    merged = merged.sort_values("t")
    if "horizon_end_ts" not in merged.columns:
        horizon = int(config["hazard"]["horizon_minutes"])
        merged["horizon_end_ts"] = merged["t"] + pd.Timedelta(minutes=horizon)
    else:
        merged["horizon_end_ts"] = pd.to_datetime(merged["horizon_end_ts"])

    val_cfg = config["model"]["training"]["validation"]
    splits = val_cfg["splits"]
    folds = generate_walkforward_folds(
        merged["t"],
        train_days=int(splits["train_days"]),
        test_days=int(splits["test_days"]),
        step_days=int(splits["step_days"]),
        embargo_minutes=int(val_cfg.get("embargo_minutes", 0)),
    )

    oof_rows = []
    fold_reports = []

    standardize = bool(config["model"]["preprocessing"].get("standardize", True))
    clip_zscore = config["model"]["preprocessing"].get("clip_zscore") if standardize else None

    for fold in folds:
        train_mask = (merged["t"] >= fold["train_start"]) & (merged["t"] <= fold["train_end"])
        test_mask = (merged["t"] >= fold["test_start"]) & (merged["t"] <= fold["test_end"])

        overlap = (merged["t"] <= fold["test_end"]) & (merged["horizon_end_ts"] >= fold["test_start"])
        train_mask = train_mask & ~overlap

        embargo_minutes = int(val_cfg.get("embargo_minutes", 0))
        if embargo_minutes > 0:
            embargo_end = fold["test_end"] + pd.Timedelta(minutes=embargo_minutes)
            train_mask = train_mask & ~((merged["t"] > fold["test_end"]) & (merged["t"] <= embargo_end))

        train = merged.loc[train_mask]
        test = merged.loc[test_mask]

        if train.empty or test.empty:
            continue

        y_train = train["y"].astype(int).to_numpy()
        y_test = test["y"].astype(int).to_numpy()

        if len(np.unique(y_train)) < 2:
            continue

        x_train = train[feature_cols].astype(float).to_numpy()
        x_test = test[feature_cols].astype(float).to_numpy()

        if standardize:
            x_train, x_test, mean, std = _standardize(x_train, x_test, clip_zscore)
        else:
            mean = np.zeros(x_train.shape[1], dtype=float)
            std = np.ones(x_train.shape[1], dtype=float)

        model_w = _fit_logistic(x_train, y_train, config)
        p_raw = _predict_with_model(model_w, x_train)

        calib_cfg = config["model"]["training"].get("calibration", {})
        if calib_cfg.get("enabled", True):
            method = calib_cfg.get("method", "isotonic")
            if method == "isotonic":
                calibrator = _fit_isotonic_calibrator(p_raw, y_train)
            else:
                calibrator = _fit_sigmoid_calibrator(p_raw, y_train)
        else:
            calibrator = None

        p_hat = _predict_with_model(model_w, x_test)
        if calibrator is not None:
            p_hat = calibrator.predict(p_hat)

        oof_rows.append(
            pd.DataFrame(
                {
                    "event_id": test["event_id"].values,
                    "t": test["t"].values,
                    "y": y_test,
                    "p_hat": p_hat,
                    "fold_id": fold["fold_id"],
                }
            )
        )

        fold_report = {
            "fold_id": fold["fold_id"],
            "test_start": fold["test_start"].isoformat(),
            "test_end": fold["test_end"].isoformat(),
            "auc": _roc_auc_score(y_test, p_hat),
            "pr_auc": _average_precision(y_test, p_hat),
            "brier": _brier_score(y_test, p_hat),
            "lift_top_decile": _lift_top_decile(y_test, p_hat),
            "reliability": _reliability_bins(y_test, p_hat),
        }
        fold_reports.append(fold_report)

    oof = pd.concat(oof_rows, ignore_index=True) if oof_rows else pd.DataFrame(
        columns=["event_id", "t", "y", "p_hat", "fold_id"]
    )

    overall = {}
    if not oof.empty:
        y_all = oof["y"].astype(int).to_numpy()
        p_all = oof["p_hat"].astype(float).to_numpy()
        overall = {
            "auc": _roc_auc_score(y_all, p_all),
            "pr_auc": _average_precision(y_all, p_all),
            "brier": _brier_score(y_all, p_all),
            "lift_top_decile": _lift_top_decile(y_all, p_all),
            "reliability": _reliability_bins(y_all, p_all),
        }

    report = {
        "dropped_y": int(dropped_y),
        "folds": fold_reports,
        "overall": overall,
        "features": feature_cols,
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        models_dir = output_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        model_path = models_dir / "hazard_model.pkl"
        if not merged.empty:
            full_y = merged["y"].astype(int).to_numpy()
            full_x = merged[feature_cols].astype(float).to_numpy()
            if standardize:
                full_x, _, mean, std = _standardize(full_x, full_x, clip_zscore)
            else:
                mean = np.zeros(full_x.shape[1], dtype=float)
                std = np.ones(full_x.shape[1], dtype=float)
            final_model = _fit_logistic(full_x, full_y, config)
            p_raw_full = _predict_with_model(final_model, full_x)
            calib_cfg = config["model"]["training"].get("calibration", {})
            if calib_cfg.get("enabled", True):
                method = calib_cfg.get("method", "isotonic")
                if method == "isotonic":
                    calibrator = _fit_isotonic_calibrator(p_raw_full, full_y)
                else:
                    calibrator = _fit_sigmoid_calibrator(p_raw_full, full_y)
            else:
                calibrator = None
            with open(model_path, "wb") as f:
                pickle.dump(
                    {
                        "model": final_model,
                        "calibrator": calibrator,
                        "features": feature_cols,
                        "mean": mean,
                        "std": std,
                        "clip_zscore": clip_zscore,
                    },
                    f,
                )

        oof_path = output_dir / "hazard_oof_predictions.parquet"
        oof.to_parquet(oof_path, index=False)

        report_path = output_dir / "hazard_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return oof, report
