from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def _find_latest_artifacts(root: Path) -> Path:
    root = Path(root)
    direct = root / "compare_summary.parquet"
    if direct.exists():
        return root
    candidates = list(root.rglob("compare_summary.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No compare_summary.parquet found under {root}")
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest.parent


def _require_paths(paths: List[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required artifacts:\n" + "\n".join(missing))


def _load_trade_logs(artifacts_dir: Path) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame], str]:
    candidates = [
        "trades_enhanced.parquet",
        "trades_baseline.parquet",
        "trade_log_hazard.parquet",
        "trade_log_baseline.parquet",
        "trades.parquet",
    ]
    trade_path = None
    for name in candidates:
        path = artifacts_dir / name
        if path.exists():
            trade_path = path
            break
    if trade_path is None:
        raise FileNotFoundError(
            "Missing required trade log. Checked:\n"
            + "\n".join(str(artifacts_dir / name) for name in candidates)
        )

    trades_main = pd.read_parquet(trade_path)
    trades_baseline = (artifacts_dir / "trades_baseline.parquet")
    trades_enhanced = (artifacts_dir / "trades_enhanced.parquet")
    base_df = pd.read_parquet(trades_baseline) if trades_baseline.exists() else None
    enh_df = pd.read_parquet(trades_enhanced) if trades_enhanced.exists() else None
    return trades_main, base_df, enh_df, trade_path.name


def _group_pnl(trades: pd.DataFrame, key: str) -> Dict[str, float]:
    if key not in trades.columns:
        return {}
    grouped = trades.groupby(trades[key].fillna("UNKNOWN"))["net_pnl"].sum()
    return {str(k): float(v) for k, v in grouped.items()}


def _pnl_by_day(trades: pd.DataFrame) -> Dict[str, float]:
    if "exit_ts" in trades.columns:
        ts = pd.to_datetime(trades["exit_ts"])
    elif "entry_ts" in trades.columns:
        ts = pd.to_datetime(trades["entry_ts"])
    else:
        return {}
    grouped = trades.assign(day=ts.dt.strftime("%Y-%m-%d")).groupby("day")["net_pnl"].sum()
    return {str(k): float(v) for k, v in grouped.items()}


def _holding_stats(trades: pd.DataFrame) -> Dict[str, float]:
    if "entry_ts" not in trades.columns or "exit_ts" not in trades.columns:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0}
    entry = pd.to_datetime(trades["entry_ts"])
    exit_ts = pd.to_datetime(trades["exit_ts"])
    minutes = (exit_ts - entry).dt.total_seconds() / 60.0
    minutes = minutes.dropna()
    if minutes.empty:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0}
    return {
        "mean": float(minutes.mean()),
        "median": float(minutes.median()),
        "p95": float(minutes.quantile(0.95)),
    }


def _turnover_stats(trades: pd.DataFrame, initial_capital: float) -> Dict[str, float]:
    if "notional" not in trades.columns:
        return {"notional_sum": 0.0, "turnover": 0.0}
    notional_sum = float(trades["notional"].sum())
    turnover = float(notional_sum / initial_capital) if initial_capital > 0 else 0.0
    return {"notional_sum": notional_sum, "turnover": turnover}


def _hazard_summary(signal_df: Optional[pd.DataFrame]) -> Dict[str, object]:
    if signal_df is None or signal_df.empty or "P_end" not in signal_df.columns:
        return {"p_hat_summary": {}, "trigger_counts": {}, "action_counts": {}}
    p_hat = pd.to_numeric(signal_df["P_end"], errors="coerce").dropna()
    if p_hat.empty:
        summary = {}
    else:
        summary = {
            "min": float(p_hat.min()),
            "mean": float(p_hat.mean()),
            "p50": float(p_hat.quantile(0.5)),
            "p90": float(p_hat.quantile(0.9)),
            "p99": float(p_hat.quantile(0.99)),
        }
    trigger_counts = {}
    if "hazard_state" in signal_df.columns:
        trigger_counts = signal_df["hazard_state"].value_counts().to_dict()
    action_counts = {}
    if "recommended_action" in signal_df.columns:
        action_counts = signal_df["recommended_action"].value_counts().to_dict()
    return {
        "p_hat_summary": summary,
        "trigger_counts": {str(k): int(v) for k, v in trigger_counts.items()},
        "action_counts": {str(k): int(v) for k, v in action_counts.items()},
    }


def _exit_saved_estimate(base: Optional[pd.DataFrame], enh: Optional[pd.DataFrame]) -> Dict[str, object]:
    if base is None or enh is None:
        return {"matched_trades": 0, "hazard_exit_count": 0, "delta_net_pnl_sum": 0.0}
    if "event_id" not in base.columns or "event_id" not in enh.columns:
        return {"matched_trades": 0, "hazard_exit_count": 0, "delta_net_pnl_sum": 0.0}
    merged = base.merge(
        enh,
        on="event_id",
        suffixes=("_base", "_enh"),
        how="inner",
    )
    if merged.empty:
        return {"matched_trades": 0, "hazard_exit_count": 0, "delta_net_pnl_sum": 0.0}
    hazard_mask = merged.get("exit_reason_enh", "").astype(str).str.startswith("hazard")
    delta = merged["net_pnl_enh"] - merged["net_pnl_base"]
    return {
        "matched_trades": int(len(merged)),
        "hazard_exit_count": int(hazard_mask.sum()),
        "delta_net_pnl_sum": float(delta.sum()),
        "delta_net_pnl_mean": float(delta.mean()) if len(delta) else 0.0,
    }


def _driver_table(attribution: Dict[str, Dict[str, float]]) -> List[Dict[str, object]]:
    drivers = []
    for key, group in attribution.items():
        for name, value in group.items():
            drivers.append({"driver": f"{key}:{name}", "net_pnl": float(value)})
    return drivers


def analyze_run(artifacts: Optional[Path] = None) -> Dict[str, object]:
    artifacts_dir = _find_latest_artifacts(Path(artifacts or "artifacts"))
    required = [
        artifacts_dir / "compare_summary.parquet",
        artifacts_dir / "pipeline_health.json",
        artifacts_dir / "hazard_report.json",
    ]
    _require_paths(required)

    trades, baseline, enhanced, trade_log_name = _load_trade_logs(artifacts_dir)
    compare = pd.read_parquet(artifacts_dir / "compare_summary.parquet")
    pipeline_health = json.loads((artifacts_dir / "pipeline_health.json").read_text(encoding="utf-8"))
    hazard_report = json.loads((artifacts_dir / "hazard_report.json").read_text(encoding="utf-8"))
    signal_path = artifacts_dir / "signal_1m.parquet"
    signal_df = pd.read_parquet(signal_path) if signal_path.exists() else None

    gross_pnl = float(trades["gross_pnl"].sum()) if "gross_pnl" in trades.columns else 0.0
    fees = float(trades["fees"].sum()) if "fees" in trades.columns else 0.0
    slippage = float(trades["slippage"].sum()) if "slippage" in trades.columns else 0.0
    net_pnl = float(trades["net_pnl"].sum()) if "net_pnl" in trades.columns else 0.0

    initial_capital = 10000.0
    turnover = _turnover_stats(trades, initial_capital)

    attribution = {
        "gross_pnl": gross_pnl,
        "fees": fees,
        "slippage": slippage,
        "net_pnl": net_pnl,
        "pnl_by_day": _pnl_by_day(trades),
        "pnl_by_regime": _group_pnl(trades, "regime"),
        "pnl_by_entry_type": _group_pnl(trades, "reason" if "reason" in trades.columns else "entry_type"),
        "pnl_by_exit_reason": _group_pnl(trades, "exit_reason"),
        "holding_time_min": _holding_stats(trades),
        "turnover_stats": turnover,
        "hazard": _hazard_summary(signal_df),
        "exits_saved_estimate": _exit_saved_estimate(baseline, enhanced),
        "trade_log_used": trade_log_name,
        "compare_summary_rows": int(len(compare)),
        "pipeline_health": pipeline_health,
        "hazard_report": hazard_report,
        "assumed_initial_capital": initial_capital,
    }

    output_json = artifacts_dir / "attribution.json"
    output_md = artifacts_dir / "attribution.md"
    output_json.write_text(json.dumps(attribution, indent=2), encoding="utf-8")

    driver_groups = {
        "day": attribution["pnl_by_day"],
        "regime": attribution["pnl_by_regime"],
        "entry_type": attribution["pnl_by_entry_type"],
        "exit_reason": attribution["pnl_by_exit_reason"],
    }
    drivers = _driver_table(driver_groups)
    top_losses = sorted(drivers, key=lambda d: d["net_pnl"])[:5]
    top_wins = sorted(drivers, key=lambda d: d["net_pnl"], reverse=True)[:5]

    lines = [
        "# Attribution Summary",
        "",
        f"- Trade log: {trade_log_name}",
        f"- Net PnL: {net_pnl:.4f}",
        f"- Gross PnL: {gross_pnl:.4f}",
        f"- Fees: {fees:.4f}",
        f"- Slippage: {slippage:.4f}",
        "",
        "## Top 5 Loss Drivers",
    ]
    for item in top_losses:
        lines.append(f"- {item['driver']}: {item['net_pnl']:.4f}")
    lines.append("")
    lines.append("## Top 5 Profit Drivers")
    for item in top_wins:
        lines.append(f"- {item['driver']}: {item['net_pnl']:.4f}")
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return attribution


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze run artifacts for attribution.")
    parser.add_argument("--artifacts", default=None, help="Artifacts directory (default: latest under ./artifacts).")
    args = parser.parse_args()
    analyze_run(Path(args.artifacts) if args.artifacts else None)


if __name__ == "__main__":
    main()
