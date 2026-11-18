# Stage 2 transformed PCA

Price-relative features, training fit through 2018-12-31. Stage 1 artifacts were not modified.

- Feature source hash: 4ecefa23bb0321deea57d55a05bc45cba11424ba852554ae40b1e3666ed457ad
- Adjusted OHLC hash: 7e9317cbc535e850cdb34c4aae87cb7f8a826b9ea948b415d8ab29e62c309a6c
- Level columns, formula L/C - 1: 84
- Width columns, formula V/C: 48
- Scale-free columns, unchanged: 96
- Minimum reconstruction correlation against main.py prices: 0.999208

| | Stage 1 raw | Stage 2 transformed |
|---|---:|---:|
| Largest eigenvalue | 62.03 | 61.14 |
| Participation ratio | 7.87 | 9.60 |
| PC1 test mean | 39.02 | -0.38 |
| PC1 training std | 7.88 | 7.82 |
| Max absolute column mean z | 9.24 | 0.75 |

| rule | components | variance share |
|---|---:|---:|
| kaiser_eigenvalue_gt_1 | 30 | 0.8968 |
| mp_sigma2_1 | 23 | 0.8585 |
| mp_iterated_sigma2 | — | — |
| cumulative_variance_80 | 17 | 0.8103 |
| cumulative_variance_90 | 31 | 0.9010 |
| cumulative_variance_95 | 49 | 0.9505 |
| all_components_diagnostic_only | 228 | 1.0000 |

No component count is selected.
