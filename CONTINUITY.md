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
- Now:
  - Add synthetic deterministic dataset tests
- Next:
  - Implement bar builders (1m/5m) from trades + L2
  - Implement regime classifier + 5m entry generator
  - Build hazard dataset + microstructure features
  - Train model + calibration + policy integration
  - Backtest baseline vs hazard-enhanced, include costs; set up GitHub Actions CI

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
  - src/labeling/triple_barrier.py
  - src/labeling/cusum.py
  - src/labeling/purge.py
  - src/features/{microstructure.py,impact.py,replenishment.py}
  - src/strategy/{entries_5m.py,policy_1m.py}
  - src/backtest/{simulator.py,metrics.py}
  - tests/test_triple_barrier.py, tests/test_cusum.py, tests/test_no_lookahead.py
- Commands (planned):
  - pytest -q
  - python -m src.cli.build_dataset ...
  - python -m src.cli.train --config configs/default.yaml
  - python -m src.cli.backtest --config configs/default.yaml
