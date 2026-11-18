# AI-trades development log

Append-only journal. New entries go at the bottom. Earlier entries are not rewritten when later code changes.

`AI-trades/` is untracked. Repository HEAD at the time of this log is `288ac2906f879a2629c741c550d934407b18f6e0`, and that commit does not contain this directory. Entries below therefore have no module commit hash.

## 2026-11-17 — Stage 1 baseline

**Objective and scope.** Read the cleaned feature matrix, fit standardization and a full PCA on rows through 2018-12-31, and compare component-retention rules. Do not select a rule, cluster, or trade.

**Files created.** `ai_trades/__init__.py`, `constants.py`, `data_prep.py`, `pca_diagnostics.py`, `stage1.py`, `regime_qqq.ipynb`, `README.md`, `tests/conftest.py`, `tests/test_alignment.py`, `tests/test_leakage.py`, `tests/test_pca_mp.py`, `tests/test_isolation.py`, `tests/test_saved_model.py`, `data/feature_matrix_v1.csv`, `data/qqq_ohlc_v1.csv`, `data/MANIFEST.txt`, `models/stage1_train2018_v1/`, `reports/stage1_diagnostics_v1.md`, `reports/stage1_diagnostics_v1.json`, `reports/stage1_eigenvalues_v1.csv`, `reports/stage1_retention_v1.csv`, `reports/stage1_scree_v1.png`.

**Implementation.** The research CSV was copied once to `feature_matrix_v1.csv`. QQQ OHLC was downloaded separately, with raw OHLC, Yahoo `Adj Close`, and auto-adjusted OHLC. `StandardScaler` and a full sklearn PCA were fit on the 4,044 training rows only. The 1,981 later rows were transformed with that frozen fit. Artifacts were written under `stage1_train2018_v1` and are not overwritten by a later run.

**Statistical decisions.**

- Training cutoff is 2018-12-31 inclusive. It was set as the first frozen split, not by scanning test-period returns.
- Eigenvalues used for Marchenko–Pastur are those of `(1/n) Z'Z` after training standardization, so their average is 1. `gamma = n_features / n_training_rows`.
- Retention rules are reported side by side. None is applied. `retention_rule_selected` in the model metadata is null.
- The iterated bulk-mean estimate of noise variance was computed and rejected as a cutoff. On this spectrum it collapses toward zero because many correlation eigenvalues sit near zero.

**Results.** Source report: `reports/stage1_diagnostics_v1.md`. Sample is 6,025 rows by 228 columns, 2002-12-04 through 2026-11-17, with no missing or infinite values and no training column of zero variance. `gamma = 0.056380`. Marchenko–Pastur support at `sigma^2 = 1` is `[0.581491, 1.531269]`. Largest eigenvalue 62.031416. Participation ratio 7.87. Sklearn and the `1/n` eigenvalues differ by at most `9.237e-14`.

| Rule | Components | Training variance |
|---|---:|---:|
| Cumulative 80% | 12 | 0.8100 |
| Marchenko–Pastur, sigma^2 = 1 | 19 | 0.8742 |
| Cumulative 90% | 24 | 0.9023 |
| Kaiser, eigenvalue > 1 | 25 | 0.9071 |
| Cumulative 95% | 40 | 0.9519 |

Principal component 1 has training mean 0, training standard deviation 7.8760, and test mean 39.0180. The largest training-standardized column shift is QQQ average true range, test mean z-score 9.24. The saved stage 1 markdown still names the snapshot in its first sample bullet. The matrix hash is `4ecefa23bb0321deea57d55a05bc45cba11424ba852554ae40b1e3666ed457ad`.

**Tests.** Unit tests covered the cutoff, calendar alignment, train-only scaler means, Marchenko–Pastur edges, retention counts, and the ban on importing `src`. The notebook was executed with `nbconvert` and completed. Exact early pass counts were 12, then 15 after the archive tests in the next entry. The suite was not left in a failing state at the end of that work.

**Limitations.** No component count was chosen. The iterated noise estimator does not converge. Price-level columns dominate the out-of-sample shift. Clustering, Markov transitions, and trading signals were not implemented.

**Next step at the time.** Decide how to treat price-level columns before any clustering. Not approved as a clustering stage.

**Git.** None.

## 2026-11-17 — Source CSV is read in place

**Objective and scope.** Keep `main.py` as the only writer of `data/asset_indicator_matrix.csv`. Have the notebook read that file directly. Archive a new hash instead of refusing or overwriting an old snapshot. Stop if a saved model's columns or training means no longer match.

**Files modified.** `ai_trades/data_prep.py`, `ai_trades/stage1.py`, `ai_trades/constants.py`, `regime_qqq.ipynb`, `README.md`, `tests/test_saved_model.py`. **Files created.** `tests/test_archive.py`, `data/manifest.jsonl`. `data/MANIFEST.txt` was regenerated from the manifest records. The stage 1 model directory was not rewritten. A rerun reported `snapshot_created: false` for `feature_matrix_v1.csv`.

**Implementation.** `archive_feature_matrix` copies the source only when its SHA-256 is absent from `manifest.jsonl`. The same hash returns the existing `feature_matrix_v1.csv`. A changed hash would become `feature_matrix_vN.csv` and would not replace v1. `assert_schema_matches_model` and `assert_training_matches_scaler` raise before a saved scaler is changed. The existing v1 manifest timestamp `2026-11-17T20:49:56Z` was kept.

**Rejected approach.** The first snapshot helper refused to copy when the research CSV changed. That was replaced because a new dataset version should be archived, not blocked, while the fitted model stays frozen unless its inputs still match.

**Tests.** `tests/test_archive.py` checks that a new hash creates v2 without replacing v1, and that a missing or extra column, or a changed training window, raises. Those tests passed as part of the 15-test run after this change.

**Limitations.** No second feature-matrix version has been produced. The stage 1 diagnostics markdown was not regenerated, so its wording still points at the snapshot file.

**Approval.** This was a data-path change. It did not approve clustering.

**Git.** None.

## 2026-11-17 — Preprocessing audit

**Objective and scope.** Classify the 228 columns and recommend a transformation. Do not change the feature matrix, the stage 1 model, or fit a new one.

**Files.** No model, report, or matrix file was created for this review. The evidence used later is the stage 1 shift table and the formulas stored in the stage 2 metadata.

**Findings that are in the stage 1 report.** Price-level and price-unit columns on QQQ sit many training standard deviations above their 2002–2018 means. Bounded oscillators and percent changes do not. QQQ average true range is the extreme, at a test mean z-score of 9.24. Bollinger bands move with the close, so they were treated as levels rather than as widths.

**Recommendation, not yet applied in this entry.** Transform price levels as `L/C - 1` and price widths as `V/C`, using the same asset's same-day adjusted close. Leave scale-free columns unchanged. Dividing by price changes units. It was not claimed to produce a stationary series. Dropping the price-unit columns was the alternative that needs no new prices and loses the distance from a moving average. Keeping the raw matrix leaves principal component 1 as a price-era factor.

**Rejected during the later unit check, recorded here so it is not reused.** A cross-asset slope of median absolute feature value against median price does not separate widths from oscillators. Average true range had a slope of about -0.12 because its dollar values are similar across assets whose prices are not. That rule was not used to assign the stage 2 formulas.

**Approval.** The direction was approved only subject to a data-consistency check. Clustering remained unapproved.

**Git.** None.

## 2026-11-17 — Stage 2, price-relative PCA

**Objective and scope.** Verify adjusted closes and indicator units. If those checks pass, transform the 132 price-unit columns, refit PCA through 2018-12-31, and save a new model. Do not overwrite stage 1 or the research CSV.

**Files created.** `ai_trades/prices.py`, `consistency.py`, `transform.py`, `stage2.py`, `tests/test_transform.py`, `data/asset_ohlc_main_convention_v1.csv`, `data/asset_ohlc_raw_v1.csv`, `models/stage2_price_relative_train2018_v1/`, `reports/stage2_diagnostics_v1.md`, `reports/stage2_diagnostics_v1.json`, `reports/stage2_eigenvalues_v1.csv`, `reports/stage2_retention_v1.csv`, `reports/stage2_scree_v1.png`. **Files modified.** `regime_qqq.ipynb`, `README.md`, `data/manifest.jsonl`, `data/MANIFEST.txt`.

**Implementation.** Adjusted and raw panels were downloaded for SPY, QQQ, IWM, TLT, `^VIX`, and `DX-Y.NYB`. The adjusted download uses `Ticker.history(period='max', interval='1d')` and does not pass `auto_adjust`, matching `StockFetcher.fetch_ohlcv`. Installed yfinance defaults that flag to `True`. Each indicator was recomputed with TA-Lib defaults on both panels. The stored matrix matched the adjusted recomputation. Families were assigned from QQQ correlations with the close, the dollar range, and dollar versus percent changes. Level columns use `L/C - 1`. Width columns use `V/C`. Ninety-six columns are unchanged. The new scaler and PCA were fit on the transformed training rows only.

**Checks that passed before the fit.** Every feature date has a finite positive adjusted close. Minimum correlation between the stored series and the adjusted recomputation is 0.999208. Median relative error is below `1e-3` for every indicator. Raw equity closes do not match the matrix. Median absolute gaps between adjusted and raw close are $31.32 (SPY), $6.94 (QQQ), $13.89 (IWM), $33.68 (TLT), and 0 for `^VIX` and `DX-Y.NYB`. Bollinger bands correlate about 1.0 with the close and have a median ratio near 1.01, so they are levels. Standard deviation correlates 0.77 with the close and 0.82 with the dollar range, so it is a width. Adjusted OHLC hash `7e9317cbc535e850cdb34c4aae87cb7f8a826b9ea948b415d8ab29e62c309a6c`. Model timestamp `2026-11-17T21:17:22Z`.

**Results.** Source: `reports/stage2_diagnostics_v1.md`. Same 4,044 / 1,981 split and the same feature hash as stage 1.

| | Stage 1 raw | Stage 2 transformed |
|---|---:|---:|
| Largest eigenvalue | 62.03 | 61.14 |
| Participation ratio | 7.87 | 9.60 |
| PC1 test mean | 39.02 | -0.38 |
| PC1 training standard deviation | 7.88 | 7.82 |
| Largest absolute column mean z | 9.24 | 0.75 |

Transformed retention counts, none selected: cumulative 80% keeps 17 components (variance share 0.8103), Marchenko–Pastur at sigma^2 = 1 keeps 23 (0.8585), Kaiser keeps 30 (0.8968), cumulative 90% keeps 31 (0.9010), cumulative 95% keeps 49 (0.9505). The iterated noise estimate still does not converge. The largest remaining column shift is DXY average true range at -0.75 training standard deviations.

**Tests.** `test_level_width_and_scale_free_formulas` first failed with `KeyError: ('QQQ', 'Bollinger Bands_upperband')` because the helper required every indicator column on a three-column toy frame. The helper was changed to transform columns that are present, and `run_stage2` checks the full list before transforming. After that fix, `pytest AI-trades/tests` collected 17 tests and reported `17 passed in 3.74s` on 2026-11-17. The notebook, including the stage 2 section, was executed with `nbconvert` and completed.

**Limitations.** The leading eigenvalue stays near 61 because the transformed moving averages remain highly collinear. `L/C - 1` and `V/C` remove the price unit and do not establish stationarity. No component count, whitening, K-means, Markov matrix, or trading rule was fit. Stage 1 files were left in place.

**Approval.** Stage 2 is a second diagnostic PCA. It is not approval to cluster or trade.

**Git.** None.

## 2026-11-17 — Documentation checkpoint

**Objective and scope.** Add this journal and `METHODOLOGY.md` from the artifacts above. Do not change the statistical pipeline.

**Files created.** `DEV_LOG.md`, `METHODOLOGY.md`.

**Tests.** The 17-test result quoted in the stage 2 entry is the run used to confirm the suite before these documents were added. No further statistical fit was run.

**Approval.** Development remains stopped after the stage 2 PCA. Clustering is not approved.

**Git.** None.

## 2026-11-17 — Methodology moved to LaTeX

**Objective and scope.** Replace the Markdown methods note with a self-contained LaTeX methods document. Do not change statistical code, fitted models, or the research checkpoint.

**Files created.** `METHODOLOGY.tex`, `METHODOLOGY.pdf`. **Files modified.** `METHODOLOGY.md`, reduced to a pointer to `METHODOLOGY.tex`. No code, notebook, or README file cited `METHODOLOGY.md`. The journal entry above still records that the Markdown note was created; that sentence is left as history.

**Implementation.** The LaTeX note keeps the implemented pipeline: wall-clock dates, the adjusted-close convention, the `L/C - 1` and `V/C` families, training-only standardization, and PCA through 2018-12-31. It separates eigenvalues of `(1/n) Z'Z` from scikit-learn's `1/(n-1)` explained variance and states the conversion used in code. Marchenko–Pastur is described as a diagnostic whose i.i.d. hypotheses do not hold for these indicators. Clustering, Markov transitions, whitening, and trading remain in a section of methods that are not implemented. Hashes and creation times were copied from `data/manifest.jsonl` and the two `metadata.json` files.

**Tests.** `pdflatex` was run twice on `METHODOLOGY.tex` with TeX Live 2025. The second run wrote `METHODOLOGY.pdf`, 8 pages, and the log contained no overfull or underfull box warnings. The statistical test suite was not run. No model was refit.

**Approval.** Development remains stopped after the stage 2 PCA. This migration is not approval to cluster or trade.

**Git.** None.

## 2026-11-17 — Capstone LDI roadmap added to the methods note

**Objective and scope.** Record the proposed eight-stage roadmap in `METHODOLOGY.tex`. Do not fit a new model or close Stage 1 or Stage 2 of that roadmap.

**Files modified.** `METHODOLOGY.tex`, `METHODOLOGY.pdf`.

**Implementation.** A section titled "Research Architecture, Milestones, and Deliverables" was inserted immediately after Scope. Roadmap Stages 0 through 2 keep the statuses supplied with the request: Stage 0 complete, Stages 1 and 2 implemented only as far as the existing diagnostics, with export and approval still open. Stages 3 through 7 are marked planned. The saved directories `stage1_train2018_v1` and `stage2_price_relative_train2018_v1` are identified as the earlier internal names for the two diagnostic PCA fits, so they are not read as acceptance of roadmap Stages 1 and 2. The supplied Stage 7 text ended at "Daily signal reports,". The remainder of that deliverable states a dated paper-trading log, no automatic order, a default of no trade, and inference without refitting. Those constraints were already part of the unfitted paper-trading design. No new trading rule was added.

**Tests.** `pdflatex` was run twice on `METHODOLOGY.tex` with TeX Live 2025. The second run wrote `METHODOLOGY.pdf`, 10 pages, and the log contained no overfull or underfull box warnings. The statistical test suite was not run. No model was refit.

**Approval.** Development remains stopped after the diagnostic PCA. Adding the roadmap does not approve clustering, Markov estimation, return modeling, backtesting, or paper trading.

**Git.** None.

## 2026-11-17 — Stage 1 artifact export

**Objective and scope.** Package the existing price-relative transform and the frozen Stage 2 scaler as the roadmap Stage 1 deliverable. Do not refit the scaler, select a PCA dimension, or mark Stage 1 complete.

**Files created.** `ai_trades/stage1_deliverable.py`, `tests/test_stage1_deliverable.py`, `artifacts/stage1_price_relative_standardized_v1/feature_matrix_standardized.csv`, `artifacts/stage1_price_relative_standardized_v1/preprocessing_pipeline.joblib`, `artifacts/stage1_price_relative_standardized_v1/feature_metadata.json`, `artifacts/stage1_price_relative_standardized_v1/validation_report.md`. **Files modified.** `METHODOLOGY.tex`, `METHODOLOGY.pdf`. The research matrix and both saved model directories were not rewritten. The frozen scaler SHA-256 remained `1a4afa857672bfaa2db44c7360d2d749a4a5649e6aa0df25a87c4f044e5decb4`.

**Implementation.** Every row of `data/asset_indicator_matrix.csv` was transformed with the existing `L/C - 1` and `V/C` formulas and then standardized with `models/stage2_price_relative_train2018_v1/scaler.joblib`. The CSV keeps the asset and indicator header rows. `feature_metadata.json` records `stage_complete: false`. The methods note now states the completion policy: a stage is complete only after its artifacts exist, validation passes, documentation is updated, and the project owner explicitly approves it.

**Results.** Source: `artifacts/stage1_price_relative_standardized_v1/validation_report.md`. Created 2026-11-17T21:55:47Z. Shape 6,025 by 228, with no non-finite values. Training rows: 4,044. Maximum absolute training column mean `2.092e-14`. Maximum absolute gap from unit population variance `4.441e-16`. The scaler's `n_samples_seen_` is 4,044. The 1,981 test rows match the frozen transform with maximum absolute difference 0, and their largest absolute column mean is 0.7458. The training block matches the saved scaler affine map and the saved PCA training mean with maximum absolute difference 0. The reloaded pipeline reproduces the matrix with maximum absolute difference 0 and refuses `fit`. The CSV round trip differs by at most `3.553e-15`. Source and adjusted-close hashes match the Stage 2 metadata. Full-sample standardization was not performed.

**Tests.** The export script reported all ten checks passed. After the final code edit, `pytest AI-trades/tests` collected 18 tests and reported `18 passed in 3.39s`. `pdflatex` was run twice on `METHODOLOGY.tex`. The second run wrote `METHODOLOGY.pdf`, 10 pages, and the log contained no overfull or underfull box warnings.

**Limitations.** Stage 1 is not approved. The CSV is not bit-identical to the in-memory matrix. The TA-Lib consistency recomputation from the Stage 2 fit was not repeated; this export checked that the feature and adjusted-panel hashes still match that fit. The roadmap Stage 2 score matrix is still not exported. Clustering, Markov estimation, return forecasts, and trading rules were not implemented.

**Approval.** Pending. This export does not complete Stage 1 and does not approve Stage 2.

**Git.** None.

## 2026-11-17 — Stage 1 incremental-inference smoke test

**Objective and scope.** Transform the last 10 held-out rows with the frozen Stage 1 pipeline. Do not refit, modify artifacts, approve Stage 1, or start Stage 2.

**Files created.** `reports/stage1_incremental_inference_smoke_test_v1.md`. The runner was `/tmp/stage1_incremental_smoke.py` and was not added to the project. No model, metadata, or artifact file was modified.

**Implementation.** The batch is 2026-11-04 through 2026-11-17, ten sessions, all after 2018-12-31. `transform` received a 10 × 228 feature frame and a 10 × 6 adjusted-close frame. `fit`, `fit_transform`, and `partial_fit` were replaced with sentinels that raise. Those sentinels were not entered. SHA-256 hashes of 13 existing files were taken before and after.

**Results.** Output shape 10 × 228. Maximum absolute difference from the exported standardized matrix `8.882e-16`. A second call on the same batch differed by 0. `n_samples_seen_` stayed 4,044. `scaler.transform` saw 10 rows on each of two calls. All 13 hashes were unchanged. The first `joblib.load` failed with `AttributeError` because the pickle names the class `__main__.FrozenPriceRelativePreprocessor`. The successful run registered that class in the smoke-test process and did not rewrite the joblib file.

**Tests.** This smoke test is the verification. The pytest suite was not run.

**Limitations.** Loading the pipeline still depends on that `__main__` alias. The test uses stored indicators and does not recompute TA-Lib from a ten-session price window. The scaler uses its stored training moments and does not read the other feature rows. Stage 1 is not approved.

**Approval.** This test does not complete Stage 1 and does not approve Stage 2.

**Git.** None.

## 2026-11-17 — Stage 1 pipeline serialization made portable

**Objective and scope.** Make the frozen preprocessor load with ordinary `joblib.load` in a fresh process. Do not refit the scaler, change formulas, or mark Stage 1 complete.

**Serialization issue.** `artifacts/stage1_price_relative_standardized_v1/preprocessing_pipeline.joblib` names the class `__main__.FrozenPriceRelativePreprocessor` because the v1 export ran as `python -m ai_trades.stage1_deliverable`. A fresh process raises `AttributeError` unless that name is registered by hand.

**Resolution.** The class now lives in `ai_trades/preprocessing.py`, whose import path is `ai_trades.preprocessing.FrozenPriceRelativePreprocessor`. v1 was read once, using that historical alias, and the same scaler object was wrapped and saved as v2. The scaler was not refit. The standardized matrix was copied byte for byte.

**Files created.** `ai_trades/preprocessing.py`, `artifacts/stage1_price_relative_standardized_v2/preprocessing_pipeline.joblib`, `artifacts/stage1_price_relative_standardized_v2/feature_matrix_standardized.csv`, `artifacts/stage1_price_relative_standardized_v2/feature_metadata.json`, `artifacts/stage1_price_relative_standardized_v2/validation_report.md`. **Files modified.** `ai_trades/stage1_deliverable.py`, `METHODOLOGY.tex`, `METHODOLOGY.pdf`. v1 files were not rewritten.

**Results.** Created 2026-11-17T22:10:39Z. A fresh process loaded v2 without a `__main__` alias and resolved `ai_trades.preprocessing.FrozenPriceRelativePreprocessor`. A second fresh process transformed 2026-11-04 through 2026-11-17. Output shape 10 × 228. Maximum absolute difference from the v1 matrix `8.882e-16`. Repeated inference differed by 0. `n_samples_seen_` stayed 4,044, and the mean and scale vectors were unchanged. v2 pipeline SHA-256 `e371ba64cd72fb831dbb4b34d169aeace07dd3af6922c836b788c3446ac62755`. v2 matrix SHA-256 `46d23c335f90377c758efc0118f12e9b01877fc4a5ce078a938418710a9a732b`, identical to v1. v2 metadata SHA-256 `adb5315474e80df1a4997887d1eff101423b44f445249e5ce0beb4414752a20a`. v2 report SHA-256 `79b6689c582213c5ebdeda769b26c01ae1fc09d6cfc22ec67aaedbc9b3348778`. v1 hashes remained `6b6621833719e272123e85d1bd5aabb698113ae0f697f411f09acf4e1f9ae93b` (pipeline), `d57a9a160d564e72e39ece79061b9432083e21f7317160f84c2cb4c7be2b550d` (metadata), `46d23c335f90377c758efc0118f12e9b01877fc4a5ce078a938418710a9a732b` (matrix), and `ebee68350332ff3b6f193600a284c81f2d5c7eaf0d109930a645c66f63a69b02` (report).

**Tests.** The two fresh-process checks passed, as recorded above. `pytest AI-trades/tests` collected 18 tests and reported `18 passed in 3.38s`. `pdflatex` was run twice on `METHODOLOGY.tex`. The second run wrote `METHODOLOGY.pdf`, 10 pages, and the log contained no overfull or underfull box warnings.

**Approval.** Pending. This repackaging does not complete Stage 1 and does not approve Stage 2.

**Git.** None.

## 2026-11-17 — Stage 1 formally approved

**Objective and scope.** Record the project owner's approval of roadmap Stage 1. Do not change artifacts, models, the feature matrix, or the preprocessing methodology, and do not start Stage 2.

**Files modified.** `DEV_LOG.md`, `METHODOLOGY.tex`, `METHODOLOGY.pdf`.

**Approval.** On November 17, 2026, the project owner approved Stage 1, Data Preparation and Statistical Validation. The approved artifact is `artifacts/stage1_price_relative_standardized_v2/`. The four completion gates are recorded as satisfied: the four artifact files exist, the v2 validation checks passed, the methods note describes them, and this approval was given explicitly. Validation status remains the result in `artifacts/stage1_price_relative_standardized_v2/validation_report.md`, including the fresh-process load and the 10-session maximum absolute difference `8.882e-16`. The v2 metadata field `stage_complete` is still false because the artifact files were preserved and not rewritten.

**Roadmap status.** Stage 1 is Complete / Approved. Stage 2, PCA and Random Matrix Analysis, is Pending and is awaiting the next research session. That session is to start with a conceptual and mathematical review of PCA. No Stage 2 implementation was begun.

**Tests.** No statistical test or refit was run. `pdflatex` was run twice on `METHODOLOGY.tex`. The second run wrote `METHODOLOGY.pdf`, 10 pages, and the log contained no overfull or underfull box warnings.

**Git.** None.

## 2026-11-17 — Stage 1 approval record reconciled

**Objective and scope.** Separate the current milestone status from the pre-approval field in `feature_metadata.json`. Do not change the v2 preprocessing files, refit the scaler, or start Stage 2.

**Files created.** `artifacts/stage1_approval.json`. **Files modified.** `DEV_LOG.md`, `METHODOLOGY.tex`, `METHODOLOGY.pdf`.

**Implementation.** `artifacts/stage1_approval.json` is the authoritative Stage 1 milestone record: status `complete`, approval status `approved`, approval date `2026-11-17`, artifact `stage1_price_relative_standardized_v2`, validation report `artifacts/stage1_price_relative_standardized_v2/validation_report.md`. The v2 `feature_metadata.json` field `stage_complete` remains `false`. That value was written before approval and is not the current milestone status. The four v2 files were not rewritten.

**Tests.** `pytest AI-trades/tests` collected 18 tests and reported `18 passed in 24.93s`. SHA-256 of the four v2 files matched the hashes taken before this reconciliation: matrix `46d23c335f90377c758efc0118f12e9b01877fc4a5ce078a938418710a9a732b`, metadata `adb5315474e80df1a4997887d1eff101423b44f445249e5ce0beb4414752a20a`, pipeline `e371ba64cd72fb831dbb4b34d169aeace07dd3af6922c836b788c3446ac62755`, and validation report `79b6689c582213c5ebdeda769b26c01ae1fc09d6cfc22ec67aaedbc9b3348778`. `pdflatex` was run twice on `METHODOLOGY.tex`. The second run wrote `METHODOLOGY.pdf`, 10 pages, and the log contained no overfull or underfull box warnings. No scaler was refit.

**Git.** None.
