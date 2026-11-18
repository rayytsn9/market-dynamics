# AI-trades

Self-contained experiment for a macro-regime view of QQQ. `main.py` remains the only entry point that downloads, cleans, and writes `data/asset_indicator_matrix.csv`. This package does not import `src/` and does not write that CSV.

`AI-trades/regime_qqq.ipynb` reads `data/asset_indicator_matrix.csv` directly. Each new file hash is copied to `AI-trades/data/feature_matrix_vN.csv` and appended to `manifest.jsonl` with the hash, shape, date range, and timestamp. Older snapshots stay in place. QQQ OHLC is stored separately under `AI-trades/data/`. A saved model is not refit on a later run. If the live columns or the training-window values no longer match that model, the run stops and the model directory is left unchanged.

This directory is at stage 1: data alignment and exploratory PCA. Clustering, Markov transitions, return evaluation, and paper trading are not implemented. No component count, regime count, or trading rule has been selected.

## How to run

From the repository root, with the project virtualenv:

```bash
PYTHONPATH=AI-trades ./venv/bin/jupyter lab AI-trades/regime_qqq.ipynb
```

The notebook reads `data/asset_indicator_matrix.csv` and calls `ai_trades.stage1.run_stage1`. The first run archives that file as `AI-trades/data/feature_matrix_v1.csv`, downloads QQQ OHLC to `AI-trades/data/qqq_ohlc_v1.csv`, fits a training-only scaler and PCA, and writes reports. A later run reads the source again. It copies a new snapshot only when the source hash changes, and it reuses the saved model. A changed column set or a changed training window stops the run.

Tests, also from the repository root:

```bash
./venv/bin/pytest AI-trades/tests
```

## Stage 1 methodology

1. Read `data/asset_indicator_matrix.csv` without writing it. Archive a new hash under `AI-trades/data/` and record the SHA-256, shape, date range, and timestamp in `manifest.jsonl`.
2. Download QQQ separately. The price file keeps raw open, high, low, and close, Yahoo `Adj Close`, and OHLC from `auto_adjust=True` (`open_adjusted`, `high_adjusted`, `low_adjusted`, `close_adjusted`).
3. Align both series on the exchange calendar date. Timezone-aware stamps keep the wall-clock date; they are not converted to UTC.
4. Split at 2018-12-31 inclusive. Every later date is test data.
5. Fit `StandardScaler` on training rows only. Columns with zero training variance would be dropped. This snapshot has none.
6. Fit a full PCA on the standardized training matrix. Eigenvalues used for Marchenko–Pastur are those of `(1/n) Z'Z`, so their average is 1. `gamma = n_features / n_training_rows`.
7. Report retention rules side by side: Kaiser (`λ > 1`), Marchenko–Pastur with `σ² = 1`, an iterated bulk-mean `σ²` when that iteration converges, and cumulative variance at 80%, 90%, and 95%. None is applied. On this feature matrix the iterated estimate collapses because many correlation eigenvalues are near zero; the report says so and does not invent a cutoff.
8. Transform the test window with the frozen scaler and PCA and report the mean and scale of the first five scores. That is a shift check, not a performance score.

Saved model: `models/stage1_train2018_v1/`. Reports: `reports/stage1_diagnostics_v1.md` and the scree plot next to it.

## Stage 2

`run_stage2` keeps that baseline. It checks adjusted closes for all six assets against a recomputation of the 38 indicators, then replaces price-level columns with \(L/C - 1\) and price-width columns with \(V/C\). The refit is stored in `models/stage2_price_relative_train2018_v1/`. No component count is selected.

## Assumptions

- A feature row dated `t` is treated as known at the close of `t`. This module does not recompute the indicators, so it cannot prove that the upstream matrix is causal. The tests lock the fit to dates on or before the cutoff.
- The training cutoff is a research choice for a first frozen model, not a result of scanning test-period returns.
- Adjusted QQQ close is the price later stages should use for returns. Stage 1 does not compute strategy returns.
- The Marchenko–Pastur edge assumes a noisy sample covariance. Indicator columns are dependent and serially correlated, so the edge is a diagnostic, not a proof that components above it are signal.
- `gamma = p/n` uses the same edge formula as a correlation matrix scaled to noise variance 1: `(1 ± sqrt(p/n))²`.

## Limitations

- One cutoff cannot represent regime drift inside 2002–2018 or after 2018. Walk-forward refits belong in a later stage and need new model directories.
- The full training window has more rows than columns (`p/n` is well below 1). A short rolling window would not. This stage does not estimate rolling covariances.
- Overlapping 5-day and 10-day returns are not evaluated yet. When they are, overlapping windows need non-overlapping tests or lag-adjusted standard errors.
- Price-level indicators (moving averages, bands, typical price, ATR) are not stationary. A scaler fit through 2018 assigns very large z-scores to later QQQ levels. Stage 1 records that shift and does not refit on the test window to remove it.

## Daily use at this stage

Re-run the notebook through the diagnostics. Do not place a trade from it. There is no signal section, no decision rule, and no order path. A daily inference section will load a saved model version only after a retention rule and a validation design have been approved.
