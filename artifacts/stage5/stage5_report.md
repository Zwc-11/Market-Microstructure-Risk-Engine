# Stage 5 Report

## Gross vs Net Decomposition
| variant | trades | turnover | gross | fees | slippage | net | win_rate | ES95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 28 | 16.8000 | 0.9702 | 134.4000 | 67.2000 | -200.6298 | 0.0357 | -17.1045 |
| meta_veto | 8 | 4.8000 | -15.9679 | 38.4000 | 19.2000 | -73.5679 | 0.1250 | -19.3635 |

## Acceptance Rate vs Threshold
Recommended threshold: 0.35 (expected_net_pnl=-40.531357403701286)
| threshold | accept_rate | expected_net_pnl | pnl_net | trade_count |
| --- | --- | --- | --- | --- |
| 0.05 | 0.4943 | -262.7896 | -262.7896 | 43 |
| 0.10 | 0.4368 | -210.2999 | -210.2999 | 38 |
| 0.15 | 0.4138 | -193.3508 | -193.3508 | 36 |
| 0.20 | 0.3908 | -164.4971 | -164.4971 | 34 |
| 0.25 | 0.3218 | -115.3031 | -115.3031 | 28 |
| 0.30 | 0.2414 | -42.7101 | -42.7101 | 21 |
| 0.35 | 0.2299 | -40.5314 | -40.5314 | 20 |
| 0.40 | 0.2299 | -40.5314 | -40.5314 | 20 |
| 0.45 | 0.2184 | -28.2824 | -28.2824 | 19 |
| 0.50 | 0.2069 | -13.4369 | -13.4369 | 18 |
| 0.55 | 0.1839 | -5.1154 | -5.1154 | 16 |
| 0.60 | 0.1724 | -10.6763 | -10.6763 | 15 |
| 0.65 | 0.1724 | -10.6763 | -10.6763 | 15 |
| 0.70 | 0.1494 | 16.4745 | 16.4745 | 13 |
| 0.75 | 0.1494 | 16.4745 | 16.4745 | 13 |
| 0.80 | 0.1379 | 35.8380 | 35.8380 | 12 |
| 0.85 | 0.1379 | 35.8380 | 35.8380 | 12 |
| 0.90 | 0.1379 | 35.8380 | 35.8380 | 12 |
| 0.95 | 0.1379 | 35.8380 | 35.8380 | 12 |

## Entry-Type Gross PnL (baseline)
- trend_pullback_short: -1.8844
- trend_pullback_long: 2.8545

## Entry-Type Gross PnL (meta_veto)
- trend_pullback_short: -15.4998
- trend_pullback_long: -0.4682

## Baseline vs Meta Veto (from compare_summary)
- baseline: trades=28 turnover=16.8000 gross=0.9702 fees=134.4000 slippage=67.2000 net=-200.6298 win_rate=0.0357 ES95=-17.1045
- meta_veto: trades=8 turnover=4.8000 gross=-15.9679 fees=38.4000 slippage=19.2000 net=-73.5679 win_rate=0.1250 ES95=-19.3635

## 1m Plot Review
Generated 15 trade plots under artifacts\stage5\meta_veto_7d\plots\trades\index.md.
Entry/exit markers align to 1m windows for manual validation of pre-entry structure.
