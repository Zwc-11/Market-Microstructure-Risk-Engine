from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.backtest.metrics import compute_tail_metrics


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


def _parse_thresholds(raw: str) -> List[float]:
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if not tokens:
        raise ValueError("No thresholds provided.")
    if "..." in tokens:
        idx = tokens.index("...")
        if idx == 0 or idx == len(tokens) - 1:
            raise ValueError("Ellipsis must be between two numeric thresholds.")
        start = float(tokens[idx - 1])
        end = float(tokens[idx + 1])
        if idx >= 2:
            prev = float(tokens[idx - 2])
            step = start - prev
        else:
            step = 0.05
        if step <= 0:
            raise ValueError("Threshold step must be positive.")
        values = []
        current = start
        while current <= end + 1e-9:
            values.append(round(current, 6))
            current += step
        tokens = tokens[: idx - 1] + [str(v) for v in values] + tokens[idx + 2 :]
    return sorted({float(t) for t in tokens})


def _win_rate(trades: pd.DataFrame) -> float:
    if trades.empty or "net_pnl" not in trades.columns:
        return 0.0
    wins = trades[trades["net_pnl"] > 0]
    return float(len(wins) / len(trades)) if len(trades) else 0.0


def sweep_meta_thresholds(
    artifacts: Optional[Path] = None,
    thresholds: Optional[str] = None,
) -> pd.DataFrame:
    artifacts_dir = _find_latest_artifacts(Path(artifacts or "artifacts"))
    meta_scores_path = artifacts_dir / "meta_scores.parquet"
    if not meta_scores_path.exists():
        raise FileNotFoundError(f"Missing required meta scores: {meta_scores_path}")
    meta_scores = pd.read_parquet(meta_scores_path)
    if meta_scores.empty:
        raise ValueError("meta_scores.parquet is empty.")
    if "score" in meta_scores.columns:
        score_col = "score"
    elif "score_oof" in meta_scores.columns:
        score_col = "score_oof"
    else:
        score_col = "score"

    baseline_path = artifacts_dir / "trades_baseline_all.parquet"
    if baseline_path.exists():
        baseline = pd.read_parquet(baseline_path)
    else:
        fallback = artifacts_dir / "trades_baseline.parquet"
        if not fallback.exists():
            raise FileNotFoundError(
                f"Missing baseline trades: {baseline_path} or {fallback}"
            )
        baseline = pd.read_parquet(fallback)

    if baseline.empty:
        raise ValueError("Baseline trades are empty.")
    if "event_id" not in baseline.columns:
        raise ValueError("Baseline trades missing event_id column.")

    threshold_list = _parse_thresholds(thresholds or "0.05,0.10,...,0.95")
    baseline_trade_count = int(len(baseline))
    min_required = max(20, int(0.2 * baseline_trade_count))

    rows: List[Dict[str, object]] = []
    for thr in threshold_list:
        accepted_ids = set(meta_scores.loc[meta_scores[score_col] >= thr, "event_id"])
        accepted_trades = baseline[baseline["event_id"].isin(accepted_ids)]

        accepted_trade_count = int(len(accepted_trades))
        accept_rate = float(accepted_trade_count / baseline_trade_count) if baseline_trade_count else 0.0
        pnl_net = float(accepted_trades["net_pnl"].sum()) if not accepted_trades.empty else 0.0
        total_fees = float(accepted_trades["fees"].sum()) if "fees" in accepted_trades.columns else 0.0
        total_slippage = (
            float(accepted_trades["slippage"].sum()) if "slippage" in accepted_trades.columns else 0.0
        )
        win_rate = _win_rate(accepted_trades)
        tail = compute_tail_metrics(accepted_trades)
        expected_net = None
        if "net_pnl_est" in meta_scores.columns:
            expected_net = float(
                meta_scores.loc[meta_scores[score_col] >= thr, "net_pnl_est"].sum()
            )

        per_regime = {}
        if "regime" in meta_scores.columns:
            for regime, group in meta_scores.groupby(meta_scores["regime"].fillna("UNKNOWN")):
                total = int(len(group))
                accepted = int((group[score_col] >= thr).sum())
                per_regime[str(regime)] = float(accepted / total) if total else 0.0

        rows.append(
            {
                "threshold": float(thr),
                "accepted_trade_count": accepted_trade_count,
                "accept_rate_vs_baseline": accept_rate,
                "pnl_net": pnl_net,
                "expected_net_pnl": expected_net,
                "total_fees": total_fees,
                "total_slippage": total_slippage,
                "win_rate": win_rate,
                "expected_shortfall_95": tail.get("expected_shortfall_95", 0.0),
                "per_regime_accept_rate": per_regime,
            }
        )

    sweep_df = pd.DataFrame(rows)
    sweep_df.to_csv(artifacts_dir / "meta_threshold_sweep.csv", index=False)
    (artifacts_dir / "meta_threshold_sweep.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    eligible = sweep_df[sweep_df["accepted_trade_count"] >= min_required]
    if eligible.empty:
        recommendation = {
            "status": "failed",
            "reason": "No threshold meets minimum trade count requirement.",
            "min_required": min_required,
        }
        (artifacts_dir / "meta_threshold_recommendation.json").write_text(
            json.dumps(recommendation, indent=2), encoding="utf-8"
        )
        raise ValueError("No threshold meets minimum trade count requirement.")

    sort_cols = ["pnl_net", "total_fees", "threshold"]
    if "expected_net_pnl" in eligible.columns and eligible["expected_net_pnl"].notna().any():
        sort_cols = ["expected_net_pnl", "total_fees", "threshold"]
    eligible = eligible.sort_values(by=sort_cols, ascending=[False, True, True])
    best = eligible.iloc[0]
    recommendation = {
        "status": "ok",
        "recommended_threshold": float(best["threshold"]),
        "accepted_trade_count": int(best["accepted_trade_count"]),
        "accept_rate_vs_baseline": float(best["accept_rate_vs_baseline"]),
        "pnl_net": float(best["pnl_net"]),
        "expected_net_pnl": float(best["expected_net_pnl"]) if "expected_net_pnl" in best else None,
        "total_fees": float(best["total_fees"]),
        "total_slippage": float(best["total_slippage"]),
        "win_rate": float(best["win_rate"]),
        "expected_shortfall_95": float(best["expected_shortfall_95"]),
        "min_required": min_required,
    }
    (artifacts_dir / "meta_threshold_recommendation.json").write_text(
        json.dumps(recommendation, indent=2), encoding="utf-8"
    )

    return sweep_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep meta-label thresholds using existing meta scores.")
    parser.add_argument("--artifacts", default=None, help="Artifacts directory (default: latest under ./artifacts).")
    parser.add_argument(
        "--thresholds",
        default="0.05,0.10,...,0.95",
        help="Comma-separated thresholds (supports ... for ranges).",
    )
    args = parser.parse_args()
    sweep_meta_thresholds(Path(args.artifacts) if args.artifacts else None, thresholds=args.thresholds)


if __name__ == "__main__":
    main()
