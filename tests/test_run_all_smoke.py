from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli import run_all as run_mod


def test_run_all_smoke(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    artifacts_dir = tmp_path / "artifacts"
    processed_dir.mkdir(parents=True, exist_ok=True)

    ts = pd.date_range("2025-01-01", periods=10, freq="1min", tz="UTC")
    bars_1m = pd.DataFrame(
        {
            "timestamp": ts,
            "mid_open": 100.0,
            "mid_high": 101.0,
            "mid_low": 99.0,
            "mid_close": 100.5,
            "volume": 1.0,
            "vwap": 100.5,
        }
    )
    bars_1m.to_parquet(processed_dir / "bars_1m.parquet", index=False)

    bars_5m = pd.DataFrame(
        {
            "timestamp": [ts[4], ts[9]],
            "mid_open": [100.0, 100.0],
            "mid_high": [101.0, 101.0],
            "mid_low": [99.0, 99.0],
            "mid_close": [100.5, 100.5],
            "volume": [5.0, 5.0],
            "vwap": [100.5, 100.5],
        }
    )
    bars_5m.to_parquet(processed_dir / "bars_5m.parquet", index=False)

    cfg = {
        "paths": {
            "raw_dir": str(raw_dir),
            "processed_dir": str(processed_dir),
            "artifacts_dir": str(artifacts_dir),
        }
    }
    cfg["data"] = {"quality": {"min_coverage": 0.0}}
    cfg["regime"] = {}
    cfg["strategy"] = {"entries_5m": {}}
    cfg["policy"] = {
        "exit": {"enabled": True, "hazard_threshold": 0.7},
        "fail_fast": {"enabled": True, "hazard_threshold": 0.9},
        "add_risk": {"enabled": True, "hazard_max_to_add": 0.2},
    }

    def fake_build_dataset(*args, **kwargs):
        return None

    def fake_load_raw_dataset(*args, **kwargs):
        dataset = args[4] if len(args) > 4 else kwargs.get("dataset")
        if dataset == "book_depth":
            return pd.DataFrame(
                {
                    "ts": [1735689600000],
                    "bid_price_1": [100.0],
                    "bid_size_1": [1.0],
                    "ask_price_1": [100.1],
                    "ask_size_1": [1.2],
                }
            )
        return pd.DataFrame(
            {
                "ts": [1735689600000],
                "agg_id": [1],
                "price": [100.0],
                "qty": [0.1],
                "is_buyer_maker": [False],
            }
        )

    def fake_classify_regime(bars, cfg):
        out = bars.copy()
        out["regime"] = "RANGE"
        return out[["regime"]]

    def fake_generate_entries_5m(bars, regime, cfg):
        return pd.DataFrame(
            {
                "event_id": ["evt1"],
                "entry_ts": [bars.index[0]],
                "side": [1],
                "entry_price": [float(bars["mid_close"].iloc[0])],
                "symbol": ["BTCUSDT"],
            }
        )

    def fake_run_backtest(bars, events, cfg):
        trades = pd.DataFrame(
            {
                "event_id": ["evt1"],
                "entry_ts": [bars.index[0]],
                "exit_ts": [bars.index[-1]],
                "side": [1],
                "entry_price": [100.5],
                "exit_price": [100.6],
                "symbol": ["BTCUSDT"],
            }
        )
        equity = pd.DataFrame({"equity": [10000.0]}, index=[bars.index[-1]])
        return trades, equity, {"pnl_net": 0.0}

    def fake_build_hazard_dataset(trades, bars, cfg):
        return pd.DataFrame(
            {
                "event_id": ["evt1"],
                "t": [bars.index[-1]],
                "y": [0],
                "horizon_end_ts": [bars.index[-1]],
            }
        )

    def fake_build_hazard_features(bars, l2, trades, cfg):
        return pd.DataFrame({"t": [bars.index[-1]], "feat1": [0.0]})

    def fake_train_hazard_model(hazard_df, features_df, cfg, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "models").mkdir(parents=True, exist_ok=True)
        import pickle
        import numpy as np

        model_pack = {
            "model": np.zeros(2),
            "features": ["feat1"],
            "mean": np.zeros(1),
            "std": np.ones(1),
            "clip_zscore": None,
            "calibrator": None,
        }
        with open(output_dir / "models" / "hazard_model.pkl", "wb") as handle:
            pickle.dump(model_pack, handle)
        pd.DataFrame({"event_id": [], "t": [], "y": [], "p_hat": [], "fold_id": []}).to_parquet(
            output_dir / "hazard_oof_predictions.parquet", index=False
        )
        (output_dir / "hazard_report.json").write_text("{}", encoding="utf-8")
        return pd.DataFrame(), {}

    def fake_run_enhanced_walkforward(*args, **kwargs):
        compare = pd.DataFrame({"fold_id": [0], "baseline_pnl_net": [0.0], "enhanced_pnl_net": [0.0]})
        trades_base = pd.DataFrame({"event_id": ["evt1"]})
        trades_enh = pd.DataFrame({"event_id": ["evt1"]})
        return compare, trades_base, trades_enh

    def fake_data_gap_report(
        raw_dir,
        exchange,
        market,
        symbol,
        datasets,
        start,
        end,
        cfg_hash,
        artifacts_dir,
        coverage=None,
        dataset_roots=None,
    ):
        report = {"missing": [], "config_hash": cfg_hash}
        (artifacts_dir / "data_gap_report.json").write_text("{}", encoding="utf-8")
        return report

    monkeypatch.setattr(run_mod, "build_dataset", fake_build_dataset)
    monkeypatch.setattr(run_mod, "_load_raw_dataset", fake_load_raw_dataset)
    monkeypatch.setattr(run_mod, "classify_regime", fake_classify_regime)
    monkeypatch.setattr(run_mod, "generate_entries_5m", fake_generate_entries_5m)
    monkeypatch.setattr(run_mod, "run_backtest", fake_run_backtest)
    monkeypatch.setattr(run_mod, "build_hazard_dataset", fake_build_hazard_dataset)
    monkeypatch.setattr(run_mod, "_build_hazard_features", fake_build_hazard_features)
    monkeypatch.setattr(run_mod, "train_hazard_model", fake_train_hazard_model)
    monkeypatch.setattr(run_mod, "_run_enhanced_walkforward", fake_run_enhanced_walkforward)
    monkeypatch.setattr(run_mod, "_data_gap_report", fake_data_gap_report)

    compare = run_mod.run_all(
        cfg,
        exchange="binance",
        symbol="BTCUSDT",
        start="2025-01-01",
        end="2025-01-02",
        source="vision",
        datasets="klines_1m,agg_trades,book_ticker,book_depth",
        vision_dir=str(tmp_path / "vision"),
        vision_auto_download=False,
        policy_variants=["full_policy"],
    )

    assert not compare.empty
    assert (artifacts_dir / "hazard_dataset.parquet").exists()
    assert (artifacts_dir / "hazard_features.parquet").exists()
    assert (artifacts_dir / "hazard_report.json").exists()
    assert (artifacts_dir / "compare_summary.parquet").exists()
    assert (artifacts_dir / "compare_summary.json").exists()
    assert (artifacts_dir / "trades_baseline.parquet").exists()
    assert (artifacts_dir / "trades_enhanced.parquet").exists()
    assert (artifacts_dir / "data_gap_report.json").exists()
    assert (artifacts_dir / "feature_health_report.json").exists()
    assert (artifacts_dir / "backtest_report.json").exists()
    assert (artifacts_dir / "model_card.md").exists()
    assert (artifacts_dir / "signal_1m.parquet").exists()
