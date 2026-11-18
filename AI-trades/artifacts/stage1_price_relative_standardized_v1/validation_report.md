# Stage 1 validation

Packaging of the frozen price-relative transform and the Stage 2 training scaler.
The scaler was not refit. Full-sample standardization was not performed.

- Artifact: `stage1_price_relative_standardized_v1`
- Created: 2026-11-17T21:55:47Z
- Source matrix: `data/asset_indicator_matrix.csv`
- Source SHA-256: `4ecefa23bb0321deea57d55a05bc45cba11424ba852554ae40b1e3666ed457ad`
- Adjusted OHLC SHA-256: `7e9317cbc535e850cdb34c4aae87cb7f8a826b9ea948b415d8ab29e62c309a6c`
- Frozen scaler: `AI-trades/models/stage2_price_relative_train2018_v1/scaler.joblib`
- Frozen scaler SHA-256: `1a4afa857672bfaa2db44c7360d2d749a4a5649e6aa0df25a87c4f044e5decb4`
- Training cutoff, inclusive: 2018-12-31
- Training rows seen by the scaler: 4044
- Test rows transformed with that scaler: 1981

The CSV keeps the MultiIndex columns. The first header row is the asset and the second is the indicator. The date index is the first column. Read it with `pandas.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)`.

| Check | Passed | Detail |
|---|---|---|
| shape | True | `{"columns": 228, "rows": 6025}` |
| finite | True | `{"nonfinite": 0}` |
| training_mean_zero | True | `{"divisor": "population mean of the exported training rows", "max_abs_column_mean": 2.091567934344271e-14, "rows": 4044, "tolerance": 1e-08}` |
| training_unit_variance | True | `{"divisor": "population variance, ddof=0, matching StandardScaler", "max_abs_variance_gap": 4.440892098500626e-16, "rows": 4044, "tolerance": 1e-08}` |
| test_rows_use_frozen_scaler | True | `{"full_sample_standardization": false, "max_abs_difference_from_frozen_transform": 0.0, "max_abs_test_column_mean": 0.7458073199834934, "scaler_n_samples_seen": 4044, "test_rows": 1981}` |
| matches_stage2_pca_input | True | `{"max_abs_difference_from_scaler_affine_map": 0.0, "max_abs_gap_from_pca_training_mean": 0.0, "max_abs_training_mean_gap": 0.0, "max_abs_training_scale_gap": 0.0}` |
| pipeline_reproduces_matrix | True | `{"max_abs_difference": 0.0}` |
| provenance | True | `{"adjusted_ohlc_sha256": "7e9317cbc535e850cdb34c4aae87cb7f8a826b9ea948b415d8ab29e62c309a6c", "column_order_matches_source": true, "dates_match_source": true, "feature_sha256": "4ecefa23bb0321deea57d55a05bc45cba11424ba852554ae40b1e3666ed457ad", "price_convention": "Ticker.history(period=max, interval=1d) with yfinance default auto_adjust=True", "stage2_adjusted_ohlc_sha256": "7e9317cbc535e850cdb34c4aae87cb7f8a826b9ea948b415d8ab29e62c309a6c", "stage2_feature_sha256": "4ecefa23bb0321deea57d55a05bc45cba11424ba852554ae40b1e3666ed457ad", "train_end": "2018-12-31"}` |
| csv_roundtrip | True | `{"max_abs_difference": 3.552713678800501e-15, "tolerance": 1e-08}` |
| reloaded_pipeline_reproduces_matrix | True | `{"fit_refused": true, "max_abs_difference": 0.0, "scaler_n_samples_seen": 4044}` |

Stage 1 of the Capstone LDI roadmap is not complete. The project owner has not approved it.
