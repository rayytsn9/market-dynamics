# Stage 1 v2 validation

Portable repackaging of the v1 preprocessor. The scaler was not refit.
Full-sample standardization was not performed. Stage 1 is not approved.

- Artifact: `stage1_price_relative_standardized_v2`
- Created: 2026-11-17T22:10:39Z
- Class: `ai_trades.preprocessing.FrozenPriceRelativePreprocessor`
- Predecessor: `stage1_price_relative_standardized_v1`
- Scaler samples seen: 4044
- Matrix SHA-256, byte copy of v1: `46d23c335f90377c758efc0118f12e9b01877fc4a5ce078a938418710a9a732b`

A fresh Python process called `joblib.load` on `preprocessing_pipeline.joblib` without registering a class on `__main__`. It resolved `ai_trades.preprocessing.FrozenPriceRelativePreprocessor`.

The same kind of fresh process then transformed 2026-11-04 through 2026-11-17 (10 sessions, all after 2018-12-31). The transform received a 10 × 228 feature frame and a 10 × 6 close frame.

| Check | Passed | Detail |
|---|---|---|
| class_module | True | `"ai_trades.preprocessing"` |
| shape_10_by_228 | True | `[10, 228]` |
| dates_and_columns_match | True | `["2026-11-04", "2026-11-05", "2026-11-06", "2026-11-09", "2026-11-10", "2026-11-11", "2026-11-12", "2026-11-13", "2026-11-16", "2026-11-17"]` |
| all_finite | True | `true` |
| max_abs_difference_below_1e-10 | True | `8.881784197001252e-16` |
| repeated_inference_identical | True | `0.0` |
| scaler_still_4044 | True | `[4044, 4044]` |
| scaler_parameters_unchanged | True | `true` |
| held_out_dates | True | `true` |

v1 SHA-256 values were unchanged by this repackaging.

| v1 file | SHA-256 |
|---|---|
| `artifacts/stage1_price_relative_standardized_v1/preprocessing_pipeline.joblib` | `6b6621833719e272123e85d1bd5aabb698113ae0f697f411f09acf4e1f9ae93b` |
| `artifacts/stage1_price_relative_standardized_v1/feature_metadata.json` | `d57a9a160d564e72e39ece79061b9432083e21f7317160f84c2cb4c7be2b550d` |
| `artifacts/stage1_price_relative_standardized_v1/feature_matrix_standardized.csv` | `46d23c335f90377c758efc0118f12e9b01877fc4a5ce078a938418710a9a732b` |
| `artifacts/stage1_price_relative_standardized_v1/validation_report.md` | `ebee68350332ff3b6f193600a284c81f2d5c7eaf0d109930a645c66f63a69b02` |

No fitting method was invoked. The check is that `n_samples_seen_` stayed 4,044 and the saved mean and scale vectors were unchanged after both transforms. A fit on these 10 rows would have replaced those values.
