# Research Stage 3 Report

## Runs
- Baseline: artifacts/stage3/baseline_20251224_040759
- Meta (0.25): artifacts/stage3/meta_025_20251224_040839
- Zero Cost Baseline: artifacts/stage3/zero_cost_20251224_040933
- Trend-only Baseline: artifacts/stage3/trend_only_baseline_20251224_041037
- Trend-only Meta (0.25): artifacts/stage3/trend_only_meta_025_20251224_041118

## Gross vs Costs Breakdown
- Baseline: gross=-40.76, fees=259.20, slippage=129.60, net=-429.56, turnover=32.40
- Meta (0.25): gross=-4.08, fees=148.80, slippage=74.40, net=-227.28, turnover=18.60
- Zero Cost Baseline: gross=-40.76, fees=0.00, slippage=0.00, net=-40.76, turnover=32.40
- Trend-only Baseline: gross=-40.76, fees=259.20, slippage=129.60, net=-429.56, turnover=32.40
- Trend-only Meta (0.25): gross=-28.56, fees=115.20, slippage=57.60, net=-201.36, turnover=14.40

## Loss Drivers
- Baseline gross pnl is negative (-40.76).
- Worst regime by net pnl: TREND (-429.56).
- Worst entry type by net pnl: trend_pullback_long (-275.68).

## Meta vs Baseline (Turnover-Controlled)
- Baseline net pnl -429.56 with turnover 32.40.
- Meta(0.25) net pnl -227.28 with turnover 18.60.
- Meta reduces turnover materially (32.4 -> 18.6) and improves net pnl, but gross pnl remains negative.

## Zero-Cost Signal Check
- Zero-cost baseline net pnl -40.76 (equals gross), still negative => signal loss, not just friction.

## Trend-only Check
- Trend-only baseline net pnl -429.56 (unchanged vs baseline).
- Trend-only meta(0.25) net pnl -201.36 with turnover 14.40.

## Recommendation
- Prioritize fixing signal logic (trend entry conditions/thresholds, barrier settings) over execution.
- Meta can reduce losses via lower turnover, but it does not create positive gross alpha in this window.

