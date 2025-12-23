# CONTINUITY.md

## Goal (incl. success criteria):
- Build a research-to-production crypto trading pipeline:
  - Layer A: Regime classifier (30m–4h): TREND / RANGE / HAZARD
  - Layer B: 5m entry strategy (trend follow + range boundary mean reversion)
  - Layer C: 1m “hazard meter” (risk detector) to improve exits / stop fail-fast / prevent add-risk
- Success criteria:
  - Baseline (5m-only) vs Enhanced (5m + 1m hazard policy) backtest comparison shows:
    - Lower max drawdown AND/OR better Sharpe after fees/slippage (primary)
    - No-lookahead validated, purged walk-forward splits
  - Repo is reproducible: one-command dataset build, train, backtest
  - CI passes (tests + lint) and produces deterministic results given fixed seed/config

## Constraints/Assumptions:
- Target market: crypto perpetuals (focus: small/mid caps; BTC/ETH optional benchmark)
- Exchanges: Binance + Deepcoin (UNCONFIRMED final set; user preference mentioned)
- Data requirements:
  - Trades required (timestamp, price, qty, side)
  - L2 required for microstructure features (at least top N levels snapshots or depth updates)
  - If true cancel/add events are unavailable, replenishment/cancel features use proxies (UNCONFIRMED)
- Must avoid lookahead/leakage:
  - Features at time t use only data <= t
  - Purged walk-forward due to overlapping events
- Cost model must be included:
  - Fees + slippage + optional latency buffer (all configurable)
- Implementation language: Python for pipeline/backtest (C++/Rust optional later) (UNCONFIRMED timeline)

## Key decisions:
- Architecture decision: COMBINE “repo-style labeling discipline” with “our hazard meter”
  - Use event sampling (CUSUM optional) + Triple-Barrier labeling for entry events
  - Hazard meter is NOT a direction predictor; it is a risk detector for exit timing / no-add-risk / fail-fast
- Labeling:
  - Primary: Triple-Barrier labels for each 5m entry event (TP / SL / timeout)
  - Hazard dataset: per-minute labels for active trades
    - Default y_t = 1 if adverse event (e.g., SL hit or “end event”) occurs within next 5 minutes, else 0
    - Alternative “end event” definition (drawdown/structure break) is configurable (UNCONFIRMED final)
- Feature set (Layer C, 1m):
  - Keep: Kyle lambda (rolling impact regression), OFI / multi-level OFI, replenishment/resilience proxies
  - Optional: VPIN as regime/risk toggle (not core alpha)
  - Absorption vs price discovery defined via combinations of lambda/r²/residuals + OFI + depth/replenishment
- Validation:
  - Purged walk-forward + embargo
  - Evaluate trading outcomes (PnL after costs, Sharpe, maxDD), not just AUC/accuracy
- Interpretability:
  - Policy rules log “why” (hazard threshold, slope, absorption spike, replenishment drop)
- Data lake layout + schemas:
  - Parquet partitioning: data/raw/<exchange>/<market>/<symbol>/<dataset>/date=YYYY-MM-DD/part-000.parquet
  - Manifest per dataset with schema_version, time_range_covered, row_count, file hashes/sizes, created_at
  - Canonical schemas for klines_1m, agg_trades, l2_snapshots in src/data/schemas.py
- Binance Vision ingestion:
  - Use daily Binance Vision files (zip) with optional .CHECKSUM verification
  - Supported datasets: klines_1m, agg_trades, book_ticker, book_depth, premium_kline, mark_kline

## State:
- Done:
  - Final direction chosen: combine triple-barrier labeling framework with 1m hazard meter + 5m entries
  - Drafted AGENTS.md structure and repo blueprint (folders/modules/tests/CI expectations)
  - Decided to keep Kyle lambda / OFI / absorption-price-discovery features; VPIN optional
  - Implemented triple-barrier labeling with OHLC detection, tie-breaks, and no-lookahead tests; ran `pytest -q`
  - Implemented symmetric CUSUM event sampling with rolling volatility threshold + tests; ran `pytest -q`
  - Strengthened no-lookahead tests with forward-merge detection and future-perturbation guards; ran `pytest -q`
  - Implemented 1m time bars with mid OHLC, volume, vwap, and L2-derived spread/microprice; added tests; ran `pytest -q`
  - Added 5m resampling from 1m time bars with OHLC/VWAP/volume aggregation; added tests; ran `pytest -q`
  - Implemented regime classifier with hazard/trend/range logic, hysteresis, and tests; ran `pytest -q`
  - Implemented 5m entry generator with RANGE/TREND logic, cooldowns, deterministic IDs, and tests; ran `pytest -q`
  - Implemented baseline event-driven backtest (triple-barrier exits, costs, metrics, CLI) with tests; ran `pytest -q`
  - Implemented walk-forward evaluation + ablations for baseline with summary output and tests; ran `pytest -q`
  - Implemented hazard dataset builder with per-minute labels, SL recompute, and tests; ran `pytest -q`
  - Implemented microstructure features (OFI, Kyle lambda, replenishment) with tests; ran `pytest -q`
  - Implemented hazard model training pipeline with walk-forward validation, calibration, and artifacts; ran `pytest -q`
  - Implemented hazard policy integration (1m exits) with enhanced backtest, comparison metrics, and tests; ran `pytest -q`
  - Implemented Phase A data ingestion: Binance USD-M REST klines + aggTrades, storage/manifest validation, build_dataset CLI, and tests; ran `pytest -q`
  - Implemented Phase C Deepcoin REST klines + trades mapping to canonical schemas; ran `pytest -q`
  - Implemented Binance Vision ingestion with checksum verification, dataset selection, and tests/fixtures; ran `pytest -q`
- Now:
  - Implement end-to-end evidence run on real data (run_all CLI + artifacts) after Vision ingest is wired
- Next:
  - Implement Phase B Binance local order book (WS sync + snapshots/diffs) and L2 storage pipeline
  - Add Deepcoin L2 snapshots/diffs support and throttled backfill
  - Add run_all CLI for full pipeline (bars -> features -> train -> backtest)
  - Add execution realism + risk kill switches (backtest-only)
  - Set up GitHub Actions CI

## Open questions (UNCONFIRMED if needed):
- Data availability:
  - Do we have reliable L2 snapshots/updates for Binance + Deepcoin for the same symbols?
  - Do we have cancel/add event flags, or only snapshots?
- Exact barriers:
  - Triple-barrier horizons (5m vs 10m) and PT/SL multipliers (sigma/ATR-based) not finalized
- Universe:
  - Which symbols/time period will be used for first “MVP proof” run?
- Execution assumptions:
  - Maker/taker fee schedule, slippage model, latency assumptions for backtest not finalized

## Working set (files/ids/commands):
- Key files:
  - AGENTS.md (agent instructions + coding standards)
  - CONTINUITY.md (this file)
  - configs/default.yaml
  - src/data/{exchange_base.py,binance.py,deepcoin.py,schemas.py,storage.py}
  - src/utils/http.py
  - src/cli/build_dataset.py
  - tests/test_data_binance.py, tests/test_data_deepcoin.py, tests/test_storage_manifest.py, tests/test_dataset_build_smoke.py
  - tests/test_vision_parser_klines.py, tests/test_vision_parser_aggtrades.py, tests/test_vision_parser_bookticker.py
  - tests/test_vision_parser_bookdepth.py, tests/test_dataset_build_vision_smoke.py
  - tests/fixtures/vision/...
- Commands (planned):
  - pytest -q
  - python -m src.cli.build_dataset --config configs/default.yaml --exchange binance --symbol BTCUSDT --start 2025-01-01 --end 2025-01-31 --mode rest
  - python -m src.cli.build_dataset --config configs/default.yaml --exchange binance --symbol BTCUSDT --start 2025-01-01 --end 2025-01-31 --source vision --datasets klines_1m,agg_trades,book_ticker,book_depth,premium_kline,mark_kline
  - python -m src.cli.build_dataset --config configs/default.yaml --exchange deepcoin --symbol BTCUSDT --start 2025-01-01 --end 2025-01-31 --mode rest
  - python -m src.cli.train --config configs/default.yaml
  - python -m src.cli.backtest --config configs/default.yaml
