# Stage 4 Report

## Runs
- E0_baseline real: artifacts\stage4\E0_baseline_real
- E0_baseline zero: artifacts\stage4\E0_baseline_zero
- E1_disable_tpl_long real: artifacts\stage4\E1_disable_tpl_long_real
- E1_disable_tpl_long zero: artifacts\stage4\E1_disable_tpl_long_zero
- E2_disable_tpl_short real: artifacts\stage4\E2_disable_tpl_short_real
- E2_disable_tpl_short zero: artifacts\stage4\E2_disable_tpl_short_zero
- E3_horizon_60 real: artifacts\stage4\E3_horizon_60_real
- E3_horizon_60 zero: artifacts\stage4\E3_horizon_60_zero
- E4_tpl_long_filter real: artifacts\stage4\E4_tpl_long_filter_real
- E4_tpl_long_filter zero: artifacts\stage4\E4_tpl_long_filter_zero

## Gross vs Net Summary (baseline trades)
| experiment | cost | gross_pnl | fees | slippage | net_pnl | trade_count | turnover |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E0_baseline | real | -30.4630 | 259.2000 | 129.6000 | -419.2630 | 54 | 32.4000 |
| E0_baseline | zero | -30.4630 | 0.0000 | 0.0000 | -30.4630 | 54 | 32.4000 |
| E1_disable_tpl_long | real | -1.8844 | 105.6000 | 52.8000 | -160.2844 | 22 | 13.2000 |
| E1_disable_tpl_long | zero | -1.8844 | 0.0000 | 0.0000 | -1.8844 | 22 | 13.2000 |
| E2_disable_tpl_short | real | -32.2331 | 158.4000 | 79.2000 | -269.8331 | 33 | 19.8000 |
| E2_disable_tpl_short | zero | -32.2331 | 0.0000 | 0.0000 | -32.2331 | 33 | 19.8000 |
| E3_horizon_60 | real | -32.3728 | 187.2000 | 93.6000 | -313.1728 | 39 | 23.4000 |
| E3_horizon_60 | zero | -32.3728 | 0.0000 | 0.0000 | -32.3728 | 39 | 23.4000 |
| E4_tpl_long_filter | real | 0.9702 | 134.4000 | 67.2000 | -200.6298 | 28 | 16.8000 |
| E4_tpl_long_filter | zero | 0.9702 | 0.0000 | 0.0000 | 0.9702 | 28 | 16.8000 |

## Entry-Type Gross PnL (sorted, baseline trades)

### E0_baseline real
- trend_pullback_long: -32.2331
- trend_pullback_short: 1.7701

### E0_baseline zero
- trend_pullback_long: -32.2331
- trend_pullback_short: 1.7701

### E1_disable_tpl_long real
- trend_pullback_short: -1.8844

### E1_disable_tpl_long zero
- trend_pullback_short: -1.8844

### E2_disable_tpl_short real
- trend_pullback_long: -32.2331

### E2_disable_tpl_short zero
- trend_pullback_long: -32.2331

### E3_horizon_60 real
- trend_pullback_long: -26.7405
- trend_pullback_short: -5.6323

### E3_horizon_60 zero
- trend_pullback_long: -26.7405
- trend_pullback_short: -5.6323

### E4_tpl_long_filter real
- trend_pullback_short: -1.8844
- trend_pullback_long: 2.8545

### E4_tpl_long_filter zero
- trend_pullback_short: -1.8844
- trend_pullback_long: 2.8545

## Conclusion
- Best zero-cost gross PnL: E4_tpl_long_filter (gross=0.9702, trades=28, turnover=16.8000).
- Apply this change and re-evaluate under realistic costs for net performance.
