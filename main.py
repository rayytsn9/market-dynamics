import config
from data.fetcher import instantiate_exchange, get_top_m_symbols, construct_dataset
from data.processor import build_panel, mean_center
from analysis.covariance import sample_covariance, validate_covariance
from analysis.mp import estimate_sigma2, mp_support, get_outliers
from analysis.pca import get_eigenpair, extract_factors, project
from interpretation.loadings import build_loading_matrix, get_factor_exposures, get_factor_groups
from interpretation.regimes import build_scores, classify_regimes, get_today_regime
import pandas as pd
import numpy as np
import datetime as dt

if __name__ == '__main__':

    now = dt.datetime.utcnow()
    start = now - dt.timedelta(hours = 24*(config.N+1))
    start = start.replace(second=0, microsecond=0, minute=0)
    since = int(start.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)

    # ── 1. Data ──────────────────────────────────────────
    print("Fetching data...")
    exchange = instantiate_exchange()
    symbols  = get_top_m_symbols(exchange, config.M_BUFFER, config.STABLECOINS)
    raw_data = construct_dataset(exchange, symbols, config.N, since=since)
    panel    = build_panel(raw_data, config.N, config.M)
    X        = mean_center(panel)


    print(f"Dataset shape: {panel.shape}")
    print(f"Aspect ratio γ = {config.M / config.N:.3f}")

    # ── 2. Covariance ────────────────────────────────────
    print("\nComputing covariance matrix...")
    C = sample_covariance(X)
    validate_covariance(C)

    # ── 3. MP Analysis ───────────────────────────────────
    print("\nFitting Marchenko-Pastur...")
    eigvals, eigvecs = get_eigenpair(C)
    gamma   = config.M / config.N
    sigma2  = estimate_sigma2(eigvals)
    a, b    = mp_support(gamma, sigma2)
    outliers = get_outliers(eigvals, b)

    print(f"sigma^2 = {sigma2:.6f}")
    print(f"MP support: [{a:.6f}, {b:.6f}]")
    print(f"Outlier eigenvalues: {outliers}")
    print(f"Number of factors: {len(outliers)}")

    # ── 4. PCA ───────────────────────────────────────────
    print("\nExtracting factors...")
    n_factors = len(outliers)
    PC        = extract_factors(eigvecs, n_factors)
    projected = project(X, PC)

    print(f"PC matrix shape: {PC.shape}")
    print(f"Projected shape: {projected.shape}")

    # ── 5. Loadings ──────────────────────────────────────
    print("\nComputing loadings...")
    pc_loadings     = build_loading_matrix(PC, panel.columns.tolist(), outliers)
    exposure_matrix = get_factor_exposures(pc_loadings, config.THRESHOLD)

    print(pc_loadings)
    print(exposure_matrix)

    # ── 6. Regimes ───────────────────────────────────────
    print("\nClassifying regimes...")
    scores  = build_scores(projected, panel.index, n_factors)
    regimes = classify_regimes(scores)
    today   = get_today_regime(scores, regimes)

    print(f"\nToday's regime: {today}")
    print(f"\nRegime distribution:\n{regimes['regime'].value_counts()}")