"""Marchenko-Pastur edges, retention counts, and PCA numerical consistency."""

import numpy as np
import pandas as pd

from ai_trades.pca_diagnostics import (
    correlation_eigenvalues,
    eigenvalue_consistency,
    fit_train_representation,
    iterated_noise_variance,
    mp_support,
    retention_comparison,
)


def test_mp_support_at_gamma_one():
    lower, upper = mp_support(1.0, 1.0)
    assert lower == 0.0
    assert upper == 4.0


def test_retention_rules_are_reported_and_not_collapsed():
    eigenvalues = np.array([4.0, 1.3, 0.5, 0.2])
    gamma = 4 / 100
    rows = {row["rule"]: row for row in retention_comparison(eigenvalues, gamma)}
    # (1 + sqrt(0.04))^2 = 1.44, so only the leading eigenvalue clears it.
    assert rows["kaiser_eigenvalue_gt_1"]["n_components"] == 2
    assert rows["mp_sigma2_1"]["n_components"] == 1
    assert rows["cumulative_variance_80"]["n_components"] == 2
    assert rows["mp_iterated_sigma2"]["n_components"] == 2
    assert rows["all_components_diagnostic_only"]["n_components"] == 4


def test_iterated_noise_variance_reports_collapse():
    eigenvalues = np.concatenate([np.array([10.0]), np.geomspace(0.2, 1e-16, 40)])
    outcome = iterated_noise_variance(eigenvalues, gamma=len(eigenvalues) / 400)
    assert outcome["converged"] is False
    rows = {row["rule"]: row for row in retention_comparison(eigenvalues, len(eigenvalues) / 400)}
    assert rows["mp_iterated_sigma2"]["n_components"] is None
    assert rows["mp_sigma2_1"]["n_components"] >= 1


def test_sklearn_and_correlation_eigenvalues_agree():
    rng = np.random.default_rng(1)
    index = pd.bdate_range("2016-01-04", periods=80)
    values = rng.normal(size=(80, 5))
    frame = pd.DataFrame(
        values,
        index=index,
        columns=pd.MultiIndex.from_product([["SPY"], ["a", "b", "c", "d", "e"]]),
    )
    fitted = fit_train_representation(frame)
    direct = correlation_eigenvalues(fitted["standardized_train"])
    check = eigenvalue_consistency(direct, fitted["pca"], fitted["n_obs"])
    assert check["within_tolerance"]
    assert abs(check["explained_variance_ratio_sum"] - 1.0) < 1e-8
