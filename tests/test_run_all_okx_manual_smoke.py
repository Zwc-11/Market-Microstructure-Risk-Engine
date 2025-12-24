from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli import run_all as run_mod


def test_run_all_okx_manual_smoke(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    artifacts_dir = tmp_path / "artifacts"
    processed_dir.mkdir(parents=True, exist_ok=True)

    fixtures = Path(__file__).resolve().parent / "fixtures" / "okx_manual_small"
    cfg = {
        "paths": {
            "raw_dir": str(raw_dir),
            "processed_dir": str(processed_dir),
            "artifacts_dir": str(artifacts_dir),
        },
        "data": {
            "quality": {"min_coverage": 0.0},
            "okx": {
                "manual": {
                    "candles_dir": str(fixtures / "candles"),
                    "trades_dir": str(fixtures / "trades"),
                    "book_dir": str(fixtures / "book"),
                    "store_top_levels": 2,
                    "allow_baseline_only": False,
                }
            },
        },
        "regime": {},
        "strategy": {"entries_5m": {}},
        "backtest": {"initial_capital": 10000},
        "features": {"ofi": {"enabled": False}, "kyle_lambda": {"enabled": False}, "replenishment": {"enabled": False}},
        "hazard": {},
        "model": {"training": {"validation": {"splits": {"train_days": 1, "test_days": 1, "step_days": 1}}}},
    }

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
                "symbol": ["BTC-USDT-SWAP"],
            }
        )

    def fake_run_backtest(bars, events, cfg):
        trades = pd.DataFrame(
            {
                "event_id": ["evt1"],
                "entry_ts": [bars.index[0]],
                "exit_ts": [bars.index[-1]],
                "side": [1],
                "entry_price": [100.0],
                "exit_price": [101.0],
                "symbol": ["BTC-USDT-SWAP"],
                "gross_pnl": [1.0],
                "net_pnl": [0.9],
                "fees": [0.05],
                "slippage": [0.05],
                "notional": [100.0],
                "exit_reason": ["barrier_tp"],
            }
        )
        equity = pd.DataFrame({"equity": [10000.9]}, index=[bars.index[-1]])
        return trades, equity, {"pnl_net": 0.9}

    def fake_build_hazard_dataset(trades, bars, cfg):
        return pd.DataFrame({"event_id": ["evt1"], "t": [bars.index[-1]], "y": [0], "horizon_end_ts": [bars.index[-1]]})

    def fake_build_hazard_features(bars, l2, trades, cfg):
        return pd.DataFrame({"t": [bars.index[-1]], "feat1": [0.0]})

    def fake_train_hazard_model(hazard_df, features_df, cfg, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "hazard_report.json").write_text("{}", encoding="utf-8")
        (output_dir / "hazard_oof_predictions.parquet").write_text("", encoding="utf-8")

    def fake_run_enhanced_walkforward(*args, **kwargs):
        compare = pd.DataFrame({"fold_id": [0], "baseline_pnl_net": [0.9], "enhanced_pnl_net": [0.9]})
        trades_base = pd.DataFrame({"event_id": ["evt1"]})
        trades_enh = pd.DataFrame({"event_id": ["evt1"]})
        return compare, trades_base, trades_enh

    monkeypatch.setattr(run_mod, "classify_regime", fake_classify_regime)
    monkeypatch.setattr(run_mod, "generate_entries_5m", fake_generate_entries_5m)
    monkeypatch.setattr(run_mod, "run_backtest", fake_run_backtest)
    monkeypatch.setattr(run_mod, "build_hazard_dataset", fake_build_hazard_dataset)
    monkeypatch.setattr(run_mod, "_build_hazard_features", fake_build_hazard_features)
    monkeypatch.setattr(run_mod, "train_hazard_model", fake_train_hazard_model)
    monkeypatch.setattr(run_mod, "_run_enhanced_walkforward", fake_run_enhanced_walkforward)

    compare = run_mod.run_all(
        cfg,
        exchange="okx",
        symbol="BTC-USDT-SWAP",
        start="2025-12-01",
        end="2025-12-02",
        source="okx_manual",
        okx_manual_candles_dir=str(fixtures / "candles"),
        okx_manual_trades_dir=str(fixtures / "trades"),
        okx_manual_book_dir=str(fixtures / "book"),
        okx_store_top_levels=2,
    )

    assert not compare.empty
    assert (artifacts_dir / "compare_summary.parquet").exists()
    assert (artifacts_dir / "trades_baseline.parquet").exists()
    assert (artifacts_dir / "data_gap_report.json").exists()
    assert (artifacts_dir / "run_manifest.json").exists()
