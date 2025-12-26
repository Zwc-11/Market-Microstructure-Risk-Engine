from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.backtest.walkforward import generate_walkforward_folds
from src.labeling.triple_barrier import triple_barrier_labels_by_regime


def _ensure_datetime_index(bars: pd.DataFrame, time_col: Optional[str]) -> pd.DataFrame:
    if time_col is not None:
        if time_col not in bars.columns:
            raise ValueError(f"time_col '{time_col}' not found in bars")
        bars = bars.copy()
        bars[time_col] = pd.to_datetime(bars[time_col])
        bars = bars.set_index(time_col)

    if not isinstance(bars.index, pd.DatetimeIndex):
        if "timestamp" in bars.columns:
            bars = bars.copy()
            bars["timestamp"] = pd.to_datetime(bars["timestamp"])
            bars = bars.set_index("timestamp")
        elif "ts" in bars.columns:
            bars = bars.copy()
            bars["ts"] = pd.to_datetime(bars["ts"])
            bars = bars.set_index("ts")
        else:
            raise ValueError("bars must have a DatetimeIndex or timestamp/ts column")
    return bars.sort_index()


def _get_price_col(bars: pd.DataFrame) -> str:
    if "mid_close" in bars.columns:
        return "mid_close"
    if "close" in bars.columns:
        return "close"
    raise ValueError("bars must include mid_close or close")


def _prepare_tb_bars(bars: pd.DataFrame, price_col: str) -> pd.DataFrame:
    price = bars[price_col].astype(float)
    if {"mid_open", "mid_high", "mid_low"}.issubset(bars.columns):
        open_px = bars["mid_open"].astype(float)
        high_px = bars["mid_high"].astype(float)
        low_px = bars["mid_low"].astype(float)
    elif {"open", "high", "low"}.issubset(bars.columns):
        open_px = bars["open"].astype(float)
        high_px = bars["high"].astype(float)
        low_px = bars["low"].astype(float)
    else:
        open_px = price
        high_px = price
        low_px = price
    return pd.DataFrame({"open": open_px, "high": high_px, "low": low_px, price_col: price}, index=bars.index)


def _asof_price(bars: pd.DataFrame, ts: pd.Timestamp, price_col: str) -> float:
    idx = bars.index
    pos = idx.searchsorted(ts, side="right") - 1
    if pos < 0:
        raise ValueError("timestamp precedes available bars")
    return float(bars.iloc[pos][price_col])


def _apply_cost_label(events: pd.DataFrame, bars_5m: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    events = events.copy()
    events["entry_ts"] = pd.to_datetime(events["entry_ts"])
    events["t0"] = pd.to_datetime(events.get("t0", events["entry_ts"]))

    bars = _ensure_datetime_index(bars_5m, None)
    price_col = _get_price_col(bars)
    bars_tb = _prepare_tb_bars(bars, price_col)

    need_labels = not {"t1", "event_type", "pt_price", "sl_price"}.issubset(events.columns)
    if need_labels:
        labels = triple_barrier_labels_by_regime(events, bars_tb, cfg, price_col=price_col, time_col=None)
        events = events.join(labels[["t1", "event_type", "pt_price", "sl_price", "label"]])
        events = events.rename(columns={"label": "tb_label"})
    else:
        if "tb_label" not in events.columns and "label" in events.columns:
            events = events.rename(columns={"label": "tb_label"})

    backtest_cfg = cfg["backtest"]
    initial_capital = float(backtest_cfg["initial_capital"])
    leverage = float(backtest_cfg.get("leverage", 1.0))
    max_notional_pct = float(backtest_cfg["sizing"]["max_position_notional_pct"])
    fee_rate = float(backtest_cfg["fees_bps"]["taker"]) / 10000.0
    slippage_rate = float(backtest_cfg.get("slippage_bps", 0.0)) / 10000.0

    gross_list = []
    net_list = []
    label_list = []

    for _, row in events.iterrows():
        side = int(row.get("side", 1))
        entry_ts = pd.Timestamp(row["entry_ts"])
        entry_price = row.get("entry_price")
        if entry_price is None or not np.isfinite(entry_price):
            entry_price = _asof_price(bars_tb, entry_ts, price_col)
        else:
            entry_price = float(entry_price)

        event_type = row.get("event_type", "timeout")
        t1 = pd.Timestamp(row.get("t1", entry_ts))

        if event_type == "pt":
            exit_price = float(row.get("pt_price", entry_price))
        elif event_type == "sl":
            exit_price = float(row.get("sl_price", entry_price))
        else:
            exit_price = _asof_price(bars_tb, t1, price_col)

        notional = initial_capital * leverage * max_notional_pct
        qty = notional / entry_price
        gross_pnl = side * (exit_price - entry_price) * qty

        fees = 2.0 * notional * fee_rate
        slippage = 2.0 * notional * slippage_rate
        net_pnl = gross_pnl - fees - slippage

        gross_list.append(gross_pnl)
        net_list.append(net_pnl)
        label_list.append(1 if net_pnl > 0 else 0)

    events["gross_pnl_est"] = gross_list
    events["net_pnl_est"] = net_list
    events["label"] = label_list
    return events


def apply_cost_label(events: pd.DataFrame, bars_5m: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    return _apply_cost_label(events, bars_5m, cfg)


def _select_micro_cols(features: pd.DataFrame, max_cols: int) -> List[str]:
    if features.empty:
        return []
    cols = []
    for col in features.columns:
        name = str(col).lower()
        if any(key in name for key in ("ofi", "lambda", "repl", "depth", "microprice", "spread")):
            cols.append(col)
    cols = cols[:max_cols]
    return cols


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


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
    class_weight: str = "balanced",
    max_iter: int = 500,
    lr: float = 0.1,
) -> np.ndarray:
    x = np.hstack([np.ones((x_train.shape[0], 1)), x_train])
    w = np.zeros(x.shape[1])
    weights = _class_weights(y_train) if class_weight == "balanced" else np.ones_like(y_train, dtype=float)

    for _ in range(max_iter):
        p = _sigmoid(x @ w)
        grad = (x.T @ ((p - y_train) * weights)) / x.shape[0]
        w = w - lr * grad
    return w


def _predict_with_model(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    xb = np.hstack([np.ones((x.shape[0], 1)), x])
    return _sigmoid(xb @ w)


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


def _precision_recall(y_true: np.ndarray, p_hat: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    pred = (p_hat >= threshold).astype(int)
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    return {"precision": precision, "recall": recall}


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
    coef = _fit_logistic(x, y_true, class_weight="balanced", max_iter=300, lr=0.2)
    return _SigmoidCalibrator(coef)


def _fit_isotonic_calibrator(scores: np.ndarray, y_true: np.ndarray) -> _IsotonicCalibrator:
    order = np.argsort(scores)
    s = scores[order]
    y = y_true[order].astype(float)

    weights = np.ones_like(y)
    v = y.tolist()
    w = weights.tolist()

    i = 0
    while i < len(v) - 1:
        if v[i] > v[i + 1]:
            total_w = w[i] + w[i + 1]
            avg = (v[i] * w[i] + v[i + 1] * w[i + 1]) / total_w
            v[i] = avg
            w[i] = total_w
            del v[i + 1]
            del w[i + 1]
            if i > 0:
                i -= 1
        else:
            i += 1

    thresholds = []
    values = []
    start = 0
    for block_val, block_w in zip(v, w):
        end = start + int(block_w)
        thresholds.append(s[min(end - 1, len(s) - 1)])
        values.append(block_val)
        start = end

    return _IsotonicCalibrator(np.array(thresholds), np.array(values))


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


def prepare_meta_frame(
    events: pd.DataFrame,
    bars_5m: pd.DataFrame,
    cfg: Dict,
    hazard_features: Optional[pd.DataFrame] = None,
    intrabar_features: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    if events.empty:
        return pd.DataFrame(), {"used_features": [], "dropped_features": []}
    if "entry_ts" not in events.columns:
        raise ValueError("events must include entry_ts for meta-labeling")
    if "label" not in events.columns:
        raise ValueError("events must include label for meta-labeling")

    events = events.copy()
    events["entry_ts"] = pd.to_datetime(events["entry_ts"])
    events["y"] = (events["label"].astype(int) == 1).astype(int)

    bars = _ensure_datetime_index(bars_5m, None)
    price_col = _get_price_col(bars)
    price = bars[price_col].astype(float)

    ret_1 = price.pct_change()
    ret_3 = price.pct_change(3)
    rv_5 = ret_1.rolling(5, min_periods=5).std(ddof=0)
    rv_10 = ret_1.rolling(10, min_periods=10).std(ddof=0)

    bar_feat = pd.DataFrame(
        {
            "ts": bars.index,
            "ret_1": ret_1,
            "ret_3": ret_3,
            "rv_5": rv_5,
            "rv_10": rv_10,
        }
    ).sort_values("ts")

    events_sorted = events.sort_values("entry_ts")
    merged = pd.merge_asof(events_sorted, bar_feat, left_on="entry_ts", right_on="ts", direction="backward")

    merged["regime_trend"] = (merged.get("regime") == "TREND").astype(int)
    merged["regime_range"] = (merged.get("regime") == "RANGE").astype(int)

    micro_cols: List[str] = []
    if hazard_features is not None and not hazard_features.empty:
        feats = hazard_features.copy()
        if "t" in feats.columns:
            feats["ts"] = pd.to_datetime(feats["t"])
        elif "timestamp" in feats.columns:
            feats["ts"] = pd.to_datetime(feats["timestamp"])
        elif "ts" in feats.columns:
            feats["ts"] = pd.to_datetime(feats["ts"])
        else:
            feats["ts"] = pd.to_datetime(feats.index)

        max_cols = int(cfg.get("model", {}).get("meta", {}).get("max_micro_features", 20))
        micro_cols = _select_micro_cols(feats, max_cols)
        if micro_cols:
            feats = feats.sort_values("ts")
            merged = pd.merge_asof(
                merged.sort_values("entry_ts"),
                feats[["ts"] + micro_cols],
                left_on="entry_ts",
                right_on="ts",
                direction="backward",
                suffixes=("", "_micro"),
            )

    intrabar_cols: List[str] = []
    if intrabar_features is not None and not intrabar_features.empty:
        ib = intrabar_features.copy()
        if "entry_ts" not in ib.columns:
            raise ValueError("intrabar_features must include entry_ts column")
        ib["entry_ts"] = pd.to_datetime(ib["entry_ts"])
        intrabar_cols = [c for c in ib.columns if c != "entry_ts"]
        merged = merged.merge(ib, on="entry_ts", how="left")

    feature_cols = [
        "regime_trend",
        "regime_range",
        "ret_1",
        "ret_3",
        "rv_5",
        "rv_10",
    ] + micro_cols + intrabar_cols

    missing = merged[feature_cols].isna().mean()
    used_features = [c for c in feature_cols if float(missing.get(c, 1.0)) <= 0.8]
    dropped = [c for c in feature_cols if c not in used_features]

    if "event_id" not in merged.columns:
        merged = merged.copy()
        merged["event_id"] = merged.index.astype(str)
    base_cols = ["event_id", "entry_ts", "y"]
    if "regime" in merged.columns:
        base_cols.append("regime")
    if "gross_pnl_est" in merged.columns:
        base_cols.append("gross_pnl_est")
    if "net_pnl_est" in merged.columns:
        base_cols.append("net_pnl_est")
    meta = merged[base_cols + used_features].copy()
    meta = meta.dropna(subset=["y"])
    if used_features:
        meta[used_features] = meta[used_features].fillna(0.0)
    meta = meta.set_index("event_id", drop=False)

    missing_any = merged[feature_cols].isna()
    missing_rows = missing_any.any(axis=1)
    missing_examples = []
    if missing_rows.any():
        sample = merged.loc[missing_rows, ["entry_ts"] + feature_cols].head(5)
        for _, row in sample.iterrows():
            missing_cols = [c for c in feature_cols if pd.isna(row.get(c))]
            missing_examples.append(
                {
                    "entry_ts": pd.to_datetime(row["entry_ts"]).isoformat(),
                    "missing_cols": missing_cols,
                }
            )

    missing_ratio = {c: float(missing.get(c, 0.0)) for c in feature_cols}
    missing_features = [c for c, ratio in missing_ratio.items() if ratio > 0.0]

    return meta, {
        "used_features": used_features,
        "dropped_features": dropped,
        "missing_ratio": missing_ratio,
        "missing_features": missing_features,
        "missing_examples": missing_examples,
    }


def fit_meta_model(meta_frame: pd.DataFrame, cfg: Dict) -> Dict[str, object]:
    if meta_frame.empty:
        raise ValueError("Meta-label frame is empty.")
    feature_cols = [
        c
        for c in meta_frame.columns
        if c not in {"event_id", "entry_ts", "y", "regime", "gross_pnl_est", "net_pnl_est"}
    ]
    if not feature_cols:
        raise ValueError("No usable features for meta-labeling.")

    x = meta_frame[feature_cols].astype(float).to_numpy()
    y = meta_frame["y"].astype(int).to_numpy()

    meta_cfg = cfg.get("model", {}).get("meta", {})
    train_cfg = meta_cfg.get("training", cfg["model"]["training"])
    class_weight = train_cfg.get("class_weight", "balanced")
    standardize = bool(meta_cfg.get("preprocessing", {}).get("standardize", cfg["model"]["preprocessing"].get("standardize", True)))
    clip = meta_cfg.get("preprocessing", {}).get("clip_zscore", cfg["model"]["preprocessing"].get("clip_zscore"))

    mean = None
    std = None
    if standardize:
        x, _, mean, std = _standardize(x, x, clip)

    model_w = _fit_logistic(x, y, class_weight=str(class_weight))

    calibrator = None
    calib_cfg = train_cfg.get("calibration", {})
    if calib_cfg.get("enabled", True):
        raw = _predict_with_model(model_w, x)
        method = str(calib_cfg.get("method", "isotonic"))
        if method == "sigmoid":
            calibrator = _fit_sigmoid_calibrator(raw, y)
        else:
            calibrator = _fit_isotonic_calibrator(raw, y)

    return {
        "model": model_w,
        "features": feature_cols,
        "mean": mean.tolist() if mean is not None else None,
        "std": std.tolist() if std is not None else None,
        "clip_zscore": clip if standardize else None,
        "calibrator": calibrator,
    }


def predict_meta(model_pack: Dict[str, object], meta_frame: pd.DataFrame) -> np.ndarray:
    feature_cols = model_pack["features"]
    x = meta_frame[feature_cols].astype(float).to_numpy()
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
    scores = _predict_with_model(model_pack["model"], x)
    calibrator = model_pack.get("calibrator")
    if calibrator is not None:
        scores = calibrator.predict(scores)
    return scores


def train_meta_model(
    events: pd.DataFrame,
    bars_5m: pd.DataFrame,
    cfg: Dict,
    hazard_features: Optional[pd.DataFrame] = None,
    intrabar_features: Optional[pd.DataFrame] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[pd.Series, Dict[str, object], Dict[str, object], pd.DataFrame]:
    labeled_events = _apply_cost_label(events, bars_5m, cfg)
    meta_frame, meta_info = prepare_meta_frame(
        labeled_events, bars_5m, cfg, hazard_features, intrabar_features
    )
    if meta_frame.empty:
        raise ValueError("Meta-label frame is empty after feature alignment.")

    meta_cfg = cfg.get("model", {}).get("meta", {})
    train_cfg = meta_cfg.get("training", cfg["model"]["training"])
    val_cfg = train_cfg.get("validation", cfg["model"]["training"]["validation"])
    splits = val_cfg.get("splits", cfg["model"]["training"]["validation"]["splits"])

    folds = generate_walkforward_folds(
        pd.to_datetime(meta_frame["entry_ts"]),
        train_days=int(splits["train_days"]),
        test_days=int(splits["test_days"]),
        step_days=int(splits["step_days"]),
        embargo_minutes=int(val_cfg.get("embargo_minutes", 0)),
    )

    if not folds:
        idx = pd.to_datetime(meta_frame["entry_ts"]).sort_values()
        split = max(1, int(len(idx) * 0.7))
        if split >= len(idx):
            split = len(idx) - 1
        folds = [
            {
                "fold_id": 0,
                "train_start": idx.iloc[0],
                "train_end": idx.iloc[split - 1],
                "test_start": idx.iloc[split],
                "test_end": idx.iloc[-1],
            }
        ]

    oof = pd.Series(index=meta_frame.index, dtype="float64")
    fold_rows: List[Dict[str, object]] = []

    for fold in folds:
        train_mask = (meta_frame["entry_ts"] >= fold["train_start"]) & (
            meta_frame["entry_ts"] <= fold["train_end"]
        )
        test_mask = (meta_frame["entry_ts"] >= fold["test_start"]) & (
            meta_frame["entry_ts"] <= fold["test_end"]
        )
        train_df = meta_frame.loc[train_mask]
        test_df = meta_frame.loc[test_mask]
        if train_df.empty or test_df.empty:
            continue

        model_pack = fit_meta_model(train_df, cfg)
        p_hat = predict_meta(model_pack, test_df)
        oof.loc[test_df.index] = p_hat

        y_true = test_df["y"].astype(int).to_numpy()
        auc = _roc_auc_score(y_true, p_hat)
        pr = _precision_recall(y_true, p_hat, threshold=0.5)
        net_est = float(test_df.get("net_pnl_est", pd.Series(dtype=float)).sum())

        fold_rows.append(
            {
                "fold_id": fold["fold_id"],
                "train_start": fold["train_start"].isoformat(),
                "train_end": fold["train_end"].isoformat(),
                "test_start": fold["test_start"].isoformat(),
                "test_end": fold["test_end"].isoformat(),
                "auc": auc,
                "precision": pr["precision"],
                "recall": pr["recall"],
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "net_pnl_est": net_est,
            }
        )

    full_model = fit_meta_model(meta_frame, cfg)
    full_scores = predict_meta(full_model, meta_frame)
    y_all = meta_frame["y"].astype(int).to_numpy()
    auc_all = _roc_auc_score(y_all, full_scores)
    pr_all = _precision_recall(y_all, full_scores, threshold=0.5)

    thresholds = [0.1, 0.25, 0.5, 0.75]
    threshold_rows = []
    for thr in thresholds:
        accept = full_scores >= thr
        accept_rate = float(accept.mean()) if len(full_scores) else 0.0
        net_est = float(meta_frame.loc[accept, "net_pnl_est"].sum()) if "net_pnl_est" in meta_frame else 0.0
        threshold_rows.append({"threshold": float(thr), "accept_rate": accept_rate, "net_pnl_est": net_est})

    report = {
        "used_features": meta_info["used_features"],
        "dropped_features": meta_info["dropped_features"],
        "missing_ratio": meta_info.get("missing_ratio", {}),
        "missing_features": meta_info.get("missing_features", []),
        "missing_examples": meta_info.get("missing_examples", []),
        "folds": fold_rows,
        "thresholds": threshold_rows,
        "overall": {
            "auc": auc_all,
            "precision": pr_all["precision"],
            "recall": pr_all["recall"],
            "rows": int(len(meta_frame)),
        },
        "oof_coverage": float(oof.notna().mean()),
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        (output_dir / "models").mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "models" / "meta_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(full_model, f)
        (output_dir / "meta_eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    return oof, full_model, report, meta_frame
