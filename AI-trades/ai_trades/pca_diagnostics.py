"""Training-only standardization, PCA, and Marchenko-Pastur diagnostics.

Component-retention rules are compared and returned. None is selected here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .constants import CUMULATIVE_VARIANCE_TARGETS, EIGENVALUE_ATOL, ZERO_VARIANCE_TOLERANCE


def mp_support(gamma: float, sigma2: float = 1.0) -> tuple[float, float]:
    """Lower and upper edges of the Marchenko-Pastur bulk. gamma = p / n."""
    if gamma <= 0 or sigma2 <= 0:
        raise ValueError("gamma and sigma2 must be positive")
    width = np.sqrt(gamma)
    return float(sigma2 * (1.0 - width) ** 2), float(sigma2 * (1.0 + width) ** 2)


def correlation_eigenvalues(standardized: np.ndarray) -> np.ndarray:
    """Eigenvalues of (1/n) Z'Z, descending. Standardized columns have variance 1."""
    n_obs = standardized.shape[0]
    if n_obs == 0:
        raise ValueError("standardized matrix has no rows")
    covariance = (standardized.T @ standardized) / n_obs
    eigenvalues = np.linalg.eigvalsh(covariance)
    # Tiny negatives are numerical zeros from a symmetric eigendecomposition.
    return np.clip(np.sort(eigenvalues)[::-1], 0.0, None)


def iterated_noise_variance(eigenvalues: np.ndarray, gamma: float, n_iter: int = 50) -> dict:
    """Iterate the mean of eigenvalues inside the Marchenko-Pastur bulk.

    A correlation spectrum with a large pile of near-zero eigenvalues makes
    this estimate collapse. That outcome is reported instead of being used as
    a cutoff. It is not a component-selection rule.
    """
    sigma2 = 1.0
    for _ in range(n_iter):
        _, upper = mp_support(gamma, sigma2)
        bulk = eigenvalues[eigenvalues <= upper]
        if len(bulk) == 0:
            return {"sigma2": None, "converged": False, "reason": "bulk became empty"}
        updated = float(np.mean(bulk))
        if updated <= 1e-8:
            return {
                "sigma2": None,
                "converged": False,
                "reason": "bulk mean collapsed toward zero because many eigenvalues are near zero",
            }
        if abs(updated - sigma2) <= 1e-8:
            return {
                "sigma2": updated,
                "converged": True,
                "n_bulk": int(len(bulk)),
                "edge": float(upper),
                "reason": "bulk mean stabilized",
            }
        sigma2 = updated
    return {"sigma2": None, "converged": False, "reason": "iteration did not stabilize"}


def sklearn_eigenvalues_on_same_scale(pca: PCA, n_obs: int) -> np.ndarray:
    """Map sklearn's 1/(n-1) explained variance onto the 1/n correlation scale."""
    return pca.explained_variance_ * (n_obs - 1) / n_obs


def eigenvalue_consistency(eigenvalues: np.ndarray, pca: PCA, n_obs: int) -> dict:
    converted = sklearn_eigenvalues_on_same_scale(pca, n_obs)
    width = min(len(eigenvalues), len(converted))
    gap = np.abs(eigenvalues[:width] - converted[:width])
    return {
        "max_abs_difference": float(gap.max()) if len(gap) else 0.0,
        "within_tolerance": bool(len(gap) and gap.max() <= EIGENVALUE_ATOL),
        "tolerance": EIGENVALUE_ATOL,
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
    }


def cumulative_count(eigenvalues: np.ndarray, target: float) -> int:
    share = np.cumsum(eigenvalues) / eigenvalues.sum()
    return int(np.searchsorted(share, target) + 1)


def retention_comparison(eigenvalues: np.ndarray, gamma: float) -> list[dict]:
    """Compare rules. The caller decides whether any rule is adopted."""
    total = float(eigenvalues.sum())
    iterated = iterated_noise_variance(eigenvalues, gamma)
    _, edge_unit = mp_support(gamma, 1.0)
    rules = [
        ("kaiser_eigenvalue_gt_1", eigenvalues > 1.0, "correlation eigenvalue above 1"),
        (
            "mp_sigma2_1",
            eigenvalues > edge_unit,
            f"above (1+sqrt(gamma))^2 = {edge_unit:.6f} with sigma^2 = 1",
        ),
    ]
    rows = []
    for name, mask, definition in rules:
        kept = eigenvalues[mask]
        rows.append(
            {
                "rule": name,
                "n_components": int(mask.sum()),
                "variance_share": float(kept.sum() / total) if len(kept) else 0.0,
                "smallest_kept_eigenvalue": float(kept.min()) if len(kept) else None,
                "definition": definition,
            }
        )
    if iterated["converged"]:
        edge_iter = iterated["edge"]
        mask = eigenvalues > edge_iter
        kept = eigenvalues[mask]
        rows.append(
            {
                "rule": "mp_iterated_sigma2",
                "n_components": int(mask.sum()),
                "variance_share": float(kept.sum() / total) if len(kept) else 0.0,
                "smallest_kept_eigenvalue": float(kept.min()) if len(kept) else None,
                "definition": (
                    f"above {edge_iter:.6f} with iterated bulk sigma^2 = {iterated['sigma2']:.6f}"
                ),
            }
        )
    else:
        rows.append(
            {
                "rule": "mp_iterated_sigma2",
                "n_components": None,
                "variance_share": None,
                "smallest_kept_eigenvalue": None,
                "definition": f"not used: {iterated['reason']}",
            }
        )
    for target in CUMULATIVE_VARIANCE_TARGETS:
        count = cumulative_count(eigenvalues, target)
        rows.append(
            {
                "rule": f"cumulative_variance_{int(target * 100)}",
                "n_components": count,
                "variance_share": float(eigenvalues[:count].sum() / total),
                "smallest_kept_eigenvalue": float(eigenvalues[count - 1]),
                "definition": f"smallest leading count whose eigenvalue share reaches {target:.0%}",
            }
        )
    rows.append(
        {
            "rule": "all_components_diagnostic_only",
            "n_components": int(len(eigenvalues)),
            "variance_share": 1.0,
            "smallest_kept_eigenvalue": float(eigenvalues[-1]),
            "definition": "full training basis; not a retention choice",
        }
    )
    return rows


def fit_train_representation(train_frame: pd.DataFrame) -> dict:
    """Fit scaler and full PCA on training rows only."""
    if train_frame.isna().any().any():
        raise ValueError("training matrix contains missing values; stage 1 does not impute")
    std = train_frame.std(axis=0, ddof=0)
    keep = std > ZERO_VARIANCE_TOLERANCE
    dropped = [f"{asset}|{indicator}" for (asset, indicator) in train_frame.columns[~keep]]
    kept_frame = train_frame.loc[:, keep]
    scaler = StandardScaler()
    standardized = scaler.fit_transform(kept_frame.to_numpy(dtype=float))
    if scaler.n_samples_seen_ != len(kept_frame):
        raise AssertionError("scaler saw a different number of rows than the training frame")
    eigenvalues = correlation_eigenvalues(standardized)
    pca = PCA(svd_solver="full")
    pca.fit(standardized)
    n_obs, n_features = standardized.shape
    gamma = n_features / n_obs
    return {
        "scaler": scaler,
        "pca": pca,
        "kept_columns": list(kept_frame.columns),
        "dropped_columns": dropped,
        "standardized_train": standardized,
        "eigenvalues": eigenvalues,
        "n_obs": int(n_obs),
        "n_features": int(n_features),
        "gamma": float(gamma),
        "consistency": eigenvalue_consistency(eigenvalues, pca, n_obs),
        "retention": retention_comparison(eigenvalues, gamma),
        "mp_edge_sigma2_1": mp_support(gamma, 1.0),
        "iterated_noise": iterated_noise_variance(eigenvalues, gamma),
    }


def frozen_score_shift(pca: PCA, standardized_train: np.ndarray, standardized_test: np.ndarray, n_components: int = 5) -> list[dict]:
    """Compare leading scores after a frozen training projection. Not a trading metric."""
    train_scores = pca.transform(standardized_train)
    test_scores = pca.transform(standardized_test)
    rows = []
    width = min(n_components, train_scores.shape[1])
    for position in range(width):
        rows.append(
            {
                "component": position + 1,
                "train_mean": float(train_scores[:, position].mean()),
                "train_std": float(train_scores[:, position].std(ddof=0)),
                "test_mean": float(test_scores[:, position].mean()),
                "test_std": float(test_scores[:, position].std(ddof=0)),
            }
        )
    return rows


def column_shift(standardized_test: np.ndarray, kept_columns, limit: int = 15) -> list[dict]:
    """Largest average test-set z-scores under the frozen training scaler."""
    means = standardized_test.mean(axis=0)
    scales = standardized_test.std(axis=0, ddof=0)
    order = np.argsort(np.abs(means))[::-1][:limit]
    rows = []
    for position in order:
        asset, indicator = kept_columns[position]
        rows.append(
            {
                "column": f"{asset}|{indicator}",
                "test_mean_z": float(means[position]),
                "test_std_z": float(scales[position]),
            }
        )
    return rows


def transform_with_frozen_scaler(scaler: StandardScaler, frame: pd.DataFrame, kept_columns) -> np.ndarray:
    ordered = frame.loc[:, kept_columns]
    return scaler.transform(ordered.to_numpy(dtype=float))
