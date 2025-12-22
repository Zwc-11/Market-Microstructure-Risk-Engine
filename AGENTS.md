# AGENTS.md

This repository builds a **5m execution strategy** (range mean-reversion / trend follow) and a **1m microstructure hazard meter** that estimates **near-term exit risk** for active trades. The hazard meter is trained and evaluated using **event sampling + Triple-Barrier labeling** with **purged walk-forward** validation.

The project is designed to be **compaction-safe** and easy for coding agents (Codex, Copilot, etc.) to contribute without losing context.

---

## 0) Agent Operating Protocol (MANDATORY)

When you are asked to work on this repo:

1) **Read `CONTINUITY.md` first**. Treat it as the only authoritative session state.
2) Produce a short **Ledger Snapshot** (Goal + Now/Next + Open Questions).
3) Work in **small steps**:
   - Implement ONE module (or one feature group) at a time.
   - Add/extend tests for that module.
   - Run tests locally.
4) **Do not proceed** to the next step unless:
   - `pytest -q` passes, and
   - any lint/check command in the repo (if present) passes.
5) After each completed step:
   - Update `CONTINUITY.md` (Done/Now/Next + decisions).
   - Summarize exactly what changed and which commands were run.

### Stop Conditions (DO NOT GUESS / DO NOT HIDE)
Stop and report immediately if any of these happen:
- Tests fail and you cannot fix within the current step.
- Data assumptions are missing (e.g., L2 snapshots not available but feature requires them).
- Ambiguity that affects correctness (barrier definitions, fees, price source).
Mark unknowns as `UNCONFIRMED` in `CONTINUITY.md`.

---

## 1) Continuity Ledger (compaction-safe)

Maintain a single Continuity Ledger in `CONTINUITY.md`. Do not rely on earlier chat text unless it is reflected in the ledger.

**Update `CONTINUITY.md` whenever**:
- goals/constraints/assumptions change
- barrier definitions change
- feature set changes
- validation methodology changes
- backtest assumptions change
- any important tool outcome changes

`CONTINUITY.md` headings must remain:
- Goal (incl. success criteria)
- Constraints/Assumptions
- Key decisions
- State (Done/Now/Next)
- Open questions (UNCONFIRMED)
- Working set (files/ids/commands)

---

## 2) Project Definitions (do not change without updating CONTINUITY.md)

### Strategy layers
- **Layer A (Regime, 30m–4h):** `TREND / RANGE / HAZARD`
- **Layer B (Entry, 5m):**
  - RANGE: boundary mean-reversion entries
  - TREND: breakout/pullback entries
- **Layer C (Risk, 1m hazard meter):**
  - outputs `P(adverse/end in next 5m | features)`
  - used for exit timing, no-add-risk gating, fail-fast exits

### Labeling standard
- Use **Triple-Barrier labeling** for each entry event (TP / SL / timeout).
- Build **per-minute hazard dataset** for active trades:
  - default: `y_t = 1` if adverse barrier occurs within next 5 minutes, else `0`
- Use **purged walk-forward + embargo** to avoid leakage from overlapping events.

---

## 3) Coding Standards (enforced by tests)
- **No lookahead**: features at time `t` use only data `<= t`.
- Deterministic results: fixed seeds; record config hash.
- Prefer clean modules over notebooks.
- Every new feature/label must include:
  - docstring with exact formula
  - unit tests
  - at least one “no lookahead” assertion/test

### Required “No Lookahead” Guards
- Any time-based join must be backward-only (`asof` style).
- If resampling, ensure right-closed intervals do not leak future data.
- Add a `test_no_lookahead.py` that fails if any feature uses timestamps > row timestamp.

---

## 4) Definition of Done (for each step)
A step is “Done” only if:
- `pytest -q` passes
- module has docstrings with formulas
- one smoke test exists for the module
- `CONTINUITY.md` updated (Done/Now/Next)

---

## 5) Deliverables (minimum)
1) Parquet ingest for trades + L2 snapshots
2) 1m + 5m bars
3) Regime classifier
4) 5m entry generator → event list
5) Triple-barrier labels for events
6) 1m hazard dataset builder
7) Model training + calibration + purged walk-forward validation
8) Backtest baseline vs hazard-enhanced policy (with fees + slippage)
9) CLI
