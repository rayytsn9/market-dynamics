# Stage 1 incremental-inference smoke test

Verification only. The frozen preprocessor was applied to the last 10 feature dates. The scaler was not refit. No model, metadata, or artifact file was written. Stage 1 is not approved, and Stage 2 was not started.

The runner was `/tmp/stage1_incremental_smoke.py`. It is not part of the project.

## Input

The source matrix and the adjusted-close panel were read only to cut out this batch. `transform` received those 10 rows and no other dates.

- Feature source: `data/asset_indicator_matrix.csv`
- Adjusted closes: `AI-trades/data/asset_ohlc_main_convention_v1.csv`
- Dates: 2026-11-04, 2026-11-05, 2026-11-06, 2026-11-09, 2026-11-10, 2026-11-11, 2026-11-12, 2026-11-13, 2026-11-16, 2026-11-17
- Range: 2026-11-04 through 2026-11-17
- Held-out period: every date is after 2018-12-31
- Feature batch shape: 10 × 228
- Adjusted-close batch shape: 10 × 6
- Output shape: 10 × 228

## Checks

| Check | Result |
|---|---|
| Shape is 10 × 228 | Passed |
| Dates match the exported matrix | Passed |
| Feature-column order matches `feature_metadata.json` | Passed |
| All outputs finite | Passed |
| Maximum absolute difference below `1e-10` | Passed. Maximum absolute difference `8.882e-16` |
| `n_samples_seen_` remains 4,044 | Passed. 4,044 before the calls and 4,044 after |
| No `fit`, `fit_transform`, or `partial_fit` | Passed. Call counts were 0. Those methods were replaced with sentinels that raise, and the sentinels were not entered |
| Repeated transform of the same batch | Passed. Maximum absolute difference 0 |
| `scaler.transform` saw only the 10-row batch | Passed. It was called twice, with 10 rows each time |
| Artifact hashes unchanged | Passed. 13 files, no hash changed |

## Artifact hashes

SHA-256 before the transform calls. The after hashes are the same values.

| File | SHA-256 |
|---|---|
| `AI-trades/artifacts/stage1_price_relative_standardized_v1/preprocessing_pipeline.joblib` | `6b6621833719e272123e85d1bd5aabb698113ae0f697f411f09acf4e1f9ae93b` |
| `AI-trades/artifacts/stage1_price_relative_standardized_v1/feature_metadata.json` | `d57a9a160d564e72e39ece79061b9432083e21f7317160f84c2cb4c7be2b550d` |
| `AI-trades/artifacts/stage1_price_relative_standardized_v1/feature_matrix_standardized.csv` | `46d23c335f90377c758efc0118f12e9b01877fc4a5ce078a938418710a9a732b` |
| `AI-trades/artifacts/stage1_price_relative_standardized_v1/validation_report.md` | `ebee68350332ff3b6f193600a284c81f2d5c7eaf0d109930a645c66f63a69b02` |
| `data/asset_indicator_matrix.csv` | `4ecefa23bb0321deea57d55a05bc45cba11424ba852554ae40b1e3666ed457ad` |
| `AI-trades/data/asset_ohlc_main_convention_v1.csv` | `7e9317cbc535e850cdb34c4aae87cb7f8a826b9ea948b415d8ab29e62c309a6c` |
| `AI-trades/data/asset_ohlc_raw_v1.csv` | `f486dfa34996f0b1e18a7df5307f4c87cab7f0155cd9cbcd8e9fb5820bd35d8f` |
| `AI-trades/models/stage1_train2018_v1/metadata.json` | `c35084510dfcaaf1384514baf48ce924c54b0f33c1df2f91f72e18e8d85b8bdb` |
| `AI-trades/models/stage1_train2018_v1/pca.joblib` | `a92a88260c3dc970f5d9a7b46059b4d3a0daf6aaf50c662a426ea4afba52c664` |
| `AI-trades/models/stage1_train2018_v1/scaler.joblib` | `f4e6cea864f8c1b6f7f3e8fd8a8f1467c4b9854d513227497ff04b8324c03fc2` |
| `AI-trades/models/stage2_price_relative_train2018_v1/metadata.json` | `7dd85f9d52e917b83950ceb8da245805756275019823a244f8d3f0fcf22e01ca` |
| `AI-trades/models/stage2_price_relative_train2018_v1/pca.joblib` | `0320c1946543f5b82943d8b7379af54009342f52ea7f56c9b857bf43fb508a55` |
| `AI-trades/models/stage2_price_relative_train2018_v1/scaler.joblib` | `1a4afa857672bfaa2db44c7360d2d749a4a5649e6aa0df25a87c4f044e5decb4` |

## Limitations

The saved `preprocessing_pipeline.joblib` does not load by import path alone. It was written by `python -m ai_trades.stage1_deliverable`, and the pickle names the class `__main__.FrozenPriceRelativePreprocessor`. The first load in this test raised `AttributeError`. The successful run registered that class on the smoke-test process and did not rewrite the joblib file. A later process that does not register the same name will fail at load.

The 10 rows are stored indicator values. This test does not recompute TA-Lib from a short price window, so it does not show that a live update can build those indicators from ten sessions.

The scaler carries the 228 training means and population scales. The transform uses those stored vectors. It does not read the other 6,015 feature rows. `n_samples_seen_` stayed 4,044.

The maximum absolute difference against the exported CSV is `8.882e-16`, inside the `1e-10` bound. That comparison is against the CSV text, not a bit-identical in-memory copy.

Stage 1 remains unapproved.
