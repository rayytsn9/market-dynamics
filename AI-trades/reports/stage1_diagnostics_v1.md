# Stage 1 diagnostics

Training standardization and PCA only. No regime model and no trading rule.

## Sample

- Feature snapshot: feature_matrix_v1.csv (6025 rows, 228 columns).
- Dates: 2002-12-04 through 2026-11-17.
- QQQ prices: qqq_ohlc_v1.csv, 1999-03-10 through 2026-11-17.
- Feature dates missing QQQ: 0.
- QQQ dates outside the feature calendar: 942.
- Train: 2002-12-04 through 2018-12-31 (4044 rows).
- Test, held out of every fit: 2019-01-02 through 2026-11-17 (1981 rows).
- Missing feature values: 0. Infinite values: 0.
- Train columns with zero variance: 0.
- Exact duplicate column groups on the training window: 0.

## PCA and Marchenko-Pastur

- Fitted columns: 228. gamma = p/n = 0.056380.
- MP support at sigma^2 = 1: [0.581491, 1.531269].
- Iterated bulk sigma^2: None. Converged: False. bulk mean collapsed toward zero because many eigenvalues are near zero.
- Largest eigenvalue: 62.031416.
- Participation ratio (effective rank): 7.87 of 228 components.
- Sklearn versus 1/n eigenvalue max absolute difference: 9.237e-14.

| rule | components | variance share | definition |
|---|---:|---:|---|
| kaiser_eigenvalue_gt_1 | 25 | 0.9071 | correlation eigenvalue above 1 |
| mp_sigma2_1 | 19 | 0.8742 | above (1+sqrt(gamma))^2 = 1.531269 with sigma^2 = 1 |
| mp_iterated_sigma2 | — | — | not used: bulk mean collapsed toward zero because many eigenvalues are near zero |
| cumulative_variance_80 | 12 | 0.8100 | smallest leading count whose eigenvalue share reaches 80% |
| cumulative_variance_90 | 24 | 0.9023 | smallest leading count whose eigenvalue share reaches 90% |
| cumulative_variance_95 | 40 | 0.9519 | smallest leading count whose eigenvalue share reaches 95% |
| all_components_diagnostic_only | 228 | 1.0000 | full training basis; not a retention choice |

No retention rule is selected.

Test rows are standardized and projected with the frozen training basis. A shift here is a preprocessing finding, not a model-selection metric.

## Frozen projection, first five components

| component | train mean | train std | test mean | test std |
|---:|---:|---:|---:|---:|
| 1 | 0.0000 | 7.8760 | 39.0180 | 14.6003 |
| 2 | 0.0000 | 6.5102 | -7.9286 | 11.5639 |
| 3 | 0.0000 | 4.5322 | 5.5822 | 6.0346 |
| 4 | -0.0000 | 3.4187 | 0.4119 | 3.3191 |
| 5 | -0.0000 | 3.2695 | -3.5806 | 3.9944 |

## Largest frozen z-score shifts on the test window

| column | test mean z | test std z |
|---|---:|---:|
| QQQ|Average True Range | 9.24 | 5.05 |
| QQQ|Bollinger Bands_upperband | 8.00 | 3.92 |
| QQQ|Weighted Close | 7.95 | 3.90 |
| QQQ|Typical Price | 7.95 | 3.90 |
| QQQ|Median Price | 7.95 | 3.90 |
| QQQ|Linear Regression Indicator | 7.95 | 3.91 |
| QQQ|Triple Exponential Moving Average | 7.95 | 3.91 |
| QQQ|Linear Regression | 7.95 | 3.91 |
| QQQ|Bollinger Bands_middleband | 7.95 | 3.90 |
| QQQ|Kaufman Adaptive Moving Average | 7.94 | 3.90 |
| QQQ|Weighted Moving Average | 7.93 | 3.90 |
| QQQ|Exponential Moving Average | 7.93 | 3.89 |
| QQQ|Moving Average | 7.92 | 3.90 |
| QQQ|Parabolic SAR | 7.92 | 3.92 |
| QQQ|Bollinger Bands_lowerband | 7.89 | 3.89 |

Gaps longer than five calendar days in the feature index:

- None.
