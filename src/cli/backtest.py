from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.backtest.metrics import (
    compute_hazard_exit_counts,
    compute_summary,
    compute_tail_metrics,
    compute_time_in_trade,
)
from src.backtest.simulator import run_backtest, run_backtest_enhanced
from src.backtest.walkforward import generate_walkforward_folds, run_walkforward
from src.modeling.train_hazard import train_hazard_model
from src.regime.regime import classify_regime
from src.strategy.entries_5m import generate_entries_5m


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_bars(processed_dir: Path) -> pd.DataFrame:
    bars_path = processed_dir / "bars_5m.parquet"
    if not bars_path.exists():
        raise FileNotFoundError(f"Missing bars file: {bars_path}")
    return pd.read_parquet(bars_path)


def _load_hazard_model(path: Path) -> dict:
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


def _load_bars_1m(processed_dir: Path) -> pd.DataFrame:
    bars_path = processed_dir / "bars_1m.parquet"
    if not bars_path.exists():
        raise FileNotFoundError(f"Missing 1m bars file: {bars_path}")
    return pd.read_parquet(bars_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline backtest.")
    parser.add_argument("--config", required=True, help="Path to config yaml")
    parser.add_argument(
        "--walkforward",
        action="store_true",
        help="Run walk-forward evaluation instead of single backtest.",
    )
    parser.add_argument(
        "--mode",
        default="both",
        choices=["both", "trend_only", "range_only"],
        help="Entry mode for walk-forward evaluation.",
    )
    parser.add_argument(
        "--fee-mults",
        default="1.0",
        help="Comma-separated fee multipliers for walk-forward (e.g., 0.5,1.0,2.0).",
    )
    parser.add_argument(
        "--slippage-mults",
        default="1.0",
        help="Comma-separated slippage multipliers for walk-forward (e.g., 0.5,1.0,2.0).",
    )
    parser.add_argument(
        "--save-trades",
        action="store_true",
        help="Save per-fold trades during walk-forward evaluation.",
    )
    parser.add_argument(
        "--enhanced",
        action="store_true",
        help="Run enhanced hazard policy backtest with baseline comparison.",
    )
    parser.add_argument(
        "--hazard-policy",
        default="full_policy",
        help="Comma-separated hazard policy variants: full_policy,hazard_exit_only,fail_fast_only.",
    )
    parser.add_argument(
        "--hazard-constant",
        default=None,
        help="Optional constant hazard probability (e.g., 0.0 or 1.0).",
    )
    parser.add_argument(
        "--hazard-model",
        default=None,
        help="Path to hazard model pickle (default: artifacts/models/hazard_model.pkl).",
    )
    parser.add_argument(
        "--hazard-features",
        default=None,
        help="Path to hazard features parquet (default: artifacts/hazard_features.parquet).",
    )
    parser.add_argument(
        "--hazard-dataset",
        default=None,
        help="Path to hazard dataset parquet for per-fold training (default: artifacts/hazard_dataset.parquet).",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    processed_dir = Path(cfg["paths"]["processed_dir"])
    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    bars_5m = _load_bars(processed_dir)

    if args.walkforward and not args.enhanced:
        fee_mults = [float(x) for x in args.fee_mults.split(",") if x]
        slippage_mults = [float(x) for x in args.slippage_mults.split(",") if x]
        save_trades_dir = artifacts_dir / "walkforward_trades" if args.save_trades else None

        summary = run_walkforward(
            bars_5m,
            cfg,
            modes=[args.mode],
            fee_mults=fee_mults,
            slippage_mults=slippage_mults,
            save_trades_dir=save_trades_dir,
        )

        summary_path = artifacts_dir / "walkforward_summary.parquet"
        summary.to_parquet(summary_path, index=False)
        print("Walk-forward summary:")
        if summary.empty:
            print("No folds produced.")
        else:
            print(summary.to_string(index=False))
    elif not args.walkforward and not args.enhanced:
        regime = classify_regime(bars_5m, cfg["regime"])
        events = generate_entries_5m(bars_5m, regime, cfg["strategy"]["entries_5m"])

        trades, equity, summary = run_backtest(bars_5m, events, cfg)

        trades_path = artifacts_dir / "trades.parquet"
        equity_path = artifacts_dir / "equity.parquet"
        trades.to_parquet(trades_path, index=False)
        equity.to_parquet(equity_path)

        print("Backtest summary:")
        for key, value in summary.items():
            print(f"{key}: {value}")
    else:
        bars_1m = _load_bars_1m(processed_dir)
        hazard_features_path = (
            Path(args.hazard_features)
            if args.hazard_features
            else artifacts_dir / "hazard_features.parquet"
        )
        hazard_dataset_path = (
            Path(args.hazard_dataset)
            if args.hazard_dataset
            else artifacts_dir / "hazard_dataset.parquet"
        )

        hazard_constant = float(args.hazard_constant) if args.hazard_constant is not None else None
        policy_variants = [v.strip() for v in args.hazard_policy.split(",") if v.strip()]

        if hazard_constant is None:
            if not hazard_features_path.exists():
                raise FileNotFoundError(f"Missing hazard features: {hazard_features_path}")
            hazard_features = pd.read_parquet(hazard_features_path)
        else:
            hazard_features = None

        if not args.walkforward and hazard_constant is None:
            hazard_model_path = (
                Path(args.hazard_model)
                if args.hazard_model
                else artifacts_dir / "models" / "hazard_model.pkl"
            )
            if not hazard_model_path.exists():
                raise FileNotFoundError(f"Missing hazard model: {hazard_model_path}")
            hazard_model_single = _load_hazard_model(hazard_model_path)
        else:
            hazard_model_single = None

        rows = []
        trades_baseline_all = []
        trades_enhanced_all = []

        if args.walkforward:
            val_cfg = cfg["model"]["training"]["validation"]
            splits = val_cfg["splits"]
            folds = generate_walkforward_folds(
                pd.to_datetime(bars_5m["timestamp"]) if "timestamp" in bars_5m.columns else bars_5m.index,
                train_days=int(splits["train_days"]),
                test_days=int(splits["test_days"]),
                step_days=int(splits["step_days"]),
                embargo_minutes=int(val_cfg.get("embargo_minutes", 0)),
            )
        else:
            folds = [
                {
                    "fold_id": 0,
                    "train_start": bars_5m.index.min(),
                    "train_end": bars_5m.index.min(),
                    "test_start": bars_5m.index.min(),
                    "test_end": bars_5m.index.max(),
                }
            ]

        for fold in folds:
            test_start = fold["test_start"]
            test_end = fold["test_end"]

            bars_5m_slice = bars_5m.loc[bars_5m.index <= test_end]
            regime = classify_regime(bars_5m_slice, cfg["regime"])
            events_all = generate_entries_5m(bars_5m_slice, regime, cfg["strategy"]["entries_5m"])
            if not events_all.empty:
                events = events_all.loc[
                    (events_all["entry_ts"] >= test_start) & (events_all["entry_ts"] <= test_end)
                ].reset_index(drop=True)
            else:
                events = events_all

            base_trades, base_equity, _ = run_backtest(bars_5m_slice, events, cfg)
            base_equity_test = base_equity.loc[(base_equity.index >= test_start) & (base_equity.index <= test_end)]
            base_summary = compute_summary(base_trades, base_equity_test, cfg["backtest"]["initial_capital"])
            base_tail = compute_tail_metrics(base_trades)
            base_time = compute_time_in_trade(base_trades)

            if hazard_constant is None and args.walkforward:
                if not hazard_dataset_path.exists():
                    raise FileNotFoundError(f"Missing hazard dataset: {hazard_dataset_path}")
                hazard_df = pd.read_parquet(hazard_dataset_path)
                hazard_df["t"] = pd.to_datetime(hazard_df["t"])
                train_df = hazard_df[hazard_df["t"] <= fold["train_end"]]

                features_df = hazard_features.copy()
                if "t" in features_df.columns:
                    features_df["t"] = pd.to_datetime(features_df["t"])
                elif "timestamp" in features_df.columns:
                    features_df["t"] = pd.to_datetime(features_df["timestamp"])
                elif "ts" in features_df.columns:
                    features_df["t"] = pd.to_datetime(features_df["ts"])
                else:
                    raise ValueError("hazard_features must include a timestamp column")

                features_train = features_df[features_df["t"] <= fold["train_end"]]
                fold_dir = artifacts_dir / "models" / f"fold_{fold['fold_id']}"
                train_hazard_model(train_df, features_train, cfg, output_dir=fold_dir)
                hazard_model_path = fold_dir / "models" / "hazard_model.pkl"
                hazard_model = _load_hazard_model(hazard_model_path)
            else:
                hazard_model = hazard_model_single

            bars_1m_slice = bars_1m.loc[bars_1m.index <= test_end]

            for variant in policy_variants:
                if hazard_constant is not None:
                    hazard_prob = pd.Series(
                        float(hazard_constant), index=bars_1m_slice.index, dtype="float64"
                    )
                else:
                    hazard_prob = None

                enh_trades, enh_equity, enh_summary, diagnostics = run_backtest_enhanced(
                    bars_5m_slice,
                    events,
                    cfg,
                    bars_1m_slice,
                    hazard_features=hazard_features,
                    hazard_model=hazard_model,
                    hazard_prob=hazard_prob,
                    policy_mode=variant,
                )
                enh_equity_test = enh_equity.loc[(enh_equity.index >= test_start) & (enh_equity.index <= test_end)]
                enh_summary = compute_summary(enh_trades, enh_equity_test, cfg["backtest"]["initial_capital"])
                enh_tail = compute_tail_metrics(enh_trades)
                enh_time = compute_time_in_trade(enh_trades)
                hazard_counts = compute_hazard_exit_counts(enh_trades)

                baseline_sl = base_trades[base_trades["exit_reason"] == "barrier_sl"]
                sl_reduction = 0.0
                if not baseline_sl.empty and "event_id" in baseline_sl.columns:
                    merged = baseline_sl.merge(
                        enh_trades[["event_id", "exit_ts", "exit_reason"]],
                        on="event_id",
                        how="left",
                        suffixes=("_base", "_enh"),
                    )
                    improved = merged[
                        (merged["exit_reason"] != "barrier_sl")
                        & (pd.to_datetime(merged["exit_ts_enh"]) < pd.to_datetime(merged["exit_ts_base"]))
                    ]
                    sl_reduction = float(len(improved) / len(baseline_sl))

                row = {
                    "fold_id": fold["fold_id"],
                    "test_start": test_start,
                    "test_end": test_end,
                    "policy_variant": variant,
                    "baseline_pnl_net": base_summary["pnl_net"],
                    "enhanced_pnl_net": enh_summary["pnl_net"],
                    "delta_pnl_net": enh_summary["pnl_net"] - base_summary["pnl_net"],
                    "baseline_sharpe": base_summary["sharpe"],
                    "enhanced_sharpe": enh_summary["sharpe"],
                    "delta_sharpe": enh_summary["sharpe"] - base_summary["sharpe"],
                    "baseline_max_drawdown": base_summary["max_drawdown"],
                    "enhanced_max_drawdown": enh_summary["max_drawdown"],
                    "delta_max_drawdown": enh_summary["max_drawdown"] - base_summary["max_drawdown"],
                    "baseline_trade_count": base_summary["trade_count"],
                    "enhanced_trade_count": enh_summary["trade_count"],
                    "baseline_win_rate": base_summary["win_rate"],
                    "enhanced_win_rate": enh_summary["win_rate"],
                    "baseline_avg_win": base_summary["avg_win"],
                    "enhanced_avg_win": enh_summary["avg_win"],
                    "baseline_avg_loss": base_summary["avg_loss"],
                    "enhanced_avg_loss": enh_summary["avg_loss"],
                    "baseline_total_fees": base_summary["total_fees"],
                    "enhanced_total_fees": enh_summary["total_fees"],
                    "baseline_total_slippage": base_summary["total_slippage"],
                    "enhanced_total_slippage": enh_summary["total_slippage"],
                    "baseline_turnover": base_summary["turnover"],
                    "enhanced_turnover": enh_summary["turnover"],
                    "baseline_worst_5pct_trade": base_tail["worst_5pct_trade"],
                    "enhanced_worst_5pct_trade": enh_tail["worst_5pct_trade"],
                    "baseline_worst_week_pnl": base_tail["worst_week_pnl"],
                    "enhanced_worst_week_pnl": enh_tail["worst_week_pnl"],
                    "baseline_avg_time_in_trade_min": base_time,
                    "enhanced_avg_time_in_trade_min": enh_time,
                    "delta_avg_time_in_trade_min": enh_time - base_time,
                    "hazard_exits": hazard_counts["hazard_exits"],
                    "fail_fast_exits": hazard_counts["fail_fast_exits"],
                    "sl_hit_reduction_rate": sl_reduction,
                    "hazard_enabled": diagnostics.get("hazard_enabled", 0),
                }
                rows.append(row)

                trades_baseline_all.append(base_trades.assign(fold_id=fold["fold_id"]))
                trades_enhanced_all.append(enh_trades.assign(fold_id=fold["fold_id"], policy_variant=variant))

        compare = pd.DataFrame(rows)
        compare_path = artifacts_dir / "compare_summary.parquet"
        compare.to_parquet(compare_path, index=False)

        trades_baseline = pd.concat(trades_baseline_all, ignore_index=True) if trades_baseline_all else pd.DataFrame()
        trades_enhanced = pd.concat(trades_enhanced_all, ignore_index=True) if trades_enhanced_all else pd.DataFrame()

        trades_baseline.to_parquet(artifacts_dir / "trades_baseline.parquet", index=False)
        trades_enhanced.to_parquet(artifacts_dir / "trades_enhanced.parquet", index=False)

        print("Comparison summary:")
        if compare.empty:
            print("No folds produced.")
        else:
            print(compare.to_string(index=False))


if __name__ == "__main__":
    main()
