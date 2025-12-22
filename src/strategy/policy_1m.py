from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _consecutive_counts(p: pd.Series, threshold: float) -> pd.Series:
    counts = []
    current = 0
    for val in p.fillna(0.0):
        if val >= threshold:
            current += 1
        else:
            current = 0
        counts.append(current)
    return pd.Series(counts, index=p.index)


def _confirm_absorption_spike(features_window: pd.DataFrame) -> Optional[bool]:
    if features_window.empty:
        return None

    ofi_cols = [c for c in features_window.columns if c.startswith("ofi_L")]
    has_lambda = "lambda" in features_window.columns
    has_resid = "resid_std" in features_window.columns

    if not ofi_cols or (not has_lambda and not has_resid):
        return None

    ofi_mag = features_window[ofi_cols].abs().max(axis=1)
    if ofi_mag.empty:
        return None
    threshold = float(ofi_mag.quantile(0.9))
    spike = float(ofi_mag.iloc[-1]) >= threshold

    lambda_drop = False
    resid_rise = False
    if len(features_window) > 1:
        if has_lambda:
            lambda_drop = float(features_window["lambda"].iloc[-1]) < float(features_window["lambda"].iloc[-2])
        if has_resid:
            resid_rise = float(features_window["resid_std"].iloc[-1]) > float(features_window["resid_std"].iloc[-2])

    return spike and (lambda_drop or resid_rise)


def _confirm_replenishment_drop(features_window: pd.DataFrame) -> Optional[bool]:
    if features_window.empty:
        return None

    repl_cols = [c for c in features_window.columns if c.startswith("repl_ratio")]
    has_spread = "spread_mean" in features_window.columns

    if not repl_cols and not has_spread:
        return None

    repl_drop = False
    spread_widen = False
    if len(features_window) > 1:
        if repl_cols:
            repl_series = features_window[repl_cols].mean(axis=1)
            repl_drop = float(repl_series.iloc[-1]) < float(repl_series.iloc[-2])
        if has_spread:
            spread_widen = float(features_window["spread_mean"].iloc[-1]) > float(
                features_window["spread_mean"].iloc[-2]
            )

    return repl_drop or spread_widen


def evaluate_hazard_policy(
    p_series: pd.Series,
    features: Optional[pd.DataFrame],
    config: Dict,
    mode: str = "full_policy",
) -> Tuple[Optional[pd.Timestamp], Optional[str], Dict[str, float]]:
    """
    Evaluate hazard policy over a series of probabilities and return first exit signal.

    Returns:
      exit_ts, exit_reason, diagnostics
    """
    policy_cfg = config["policy"]
    exit_cfg = policy_cfg["exit"]
    ff_cfg = policy_cfg["fail_fast"]
    add_cfg = policy_cfg["add_risk"]

    p_series = p_series.dropna()
    if p_series.empty:
        return None, None, {"hazard_exits": 0, "fail_fast_exits": 0}

    consecutive = _consecutive_counts(p_series, float(exit_cfg["hazard_threshold"]))
    require_rising = bool(exit_cfg.get("require_rising", False))
    consecutive_needed = int(exit_cfg.get("consecutive_minutes", 1))

    feature_window = int(config["hazard"].get("feature_window_minutes", 3))

    prev_p = None
    for ts, p_val in p_series.items():
        dp = p_val - prev_p if prev_p is not None else 0.0
        prev_p = p_val

        if mode in {"full_policy", "fail_fast_only"} and bool(ff_cfg.get("enabled", True)):
            if p_val >= float(ff_cfg["hazard_threshold"]):
                confirm = True
                confirm_cfg = ff_cfg.get("confirm_signals", {})
                window = features.loc[:ts].tail(feature_window) if features is not None else pd.DataFrame()

                if confirm_cfg.get("require_absorption_spike", False):
                    absorption = _confirm_absorption_spike(window)
                    confirm = confirm and (absorption if absorption is not None else True)
                if confirm_cfg.get("require_replenishment_drop", False):
                    repl = _confirm_replenishment_drop(window)
                    confirm = confirm and (repl if repl is not None else True)

                if confirm:
                    return ts, "hazard_fail_fast", {"hazard_exits": 0, "fail_fast_exits": 1}

        if mode in {"full_policy", "hazard_exit_only"} and bool(exit_cfg.get("enabled", True)):
            if p_val >= float(exit_cfg["hazard_threshold"]):
                rising_ok = True if not require_rising else dp > 0
                if rising_ok and consecutive.loc[ts] >= consecutive_needed:
                    return ts, "hazard_exit", {"hazard_exits": 1, "fail_fast_exits": 0}

    return None, None, {"hazard_exits": 0, "fail_fast_exits": 0}


def allow_add_risk(p_val: float, config: Dict) -> bool:
    add_cfg = config["policy"]["add_risk"]
    if not bool(add_cfg.get("enabled", True)):
        return True
    return p_val <= float(add_cfg["hazard_max_to_add"])
