"""Stage 1 entry point: snapshot, QQQ prices, and exploratory PCA."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .constants import FEATURE_SOURCE, MODEL_DIR, MODEL_VERSION, REPORT_DIR, TRAIN_END
from .data_prep import (
    archive_feature_matrix,
    assert_schema_matches_model,
    assert_training_matches_scaler,
    audit_alignment,
    audit_features,
    ensure_qqq_prices,
    load_feature_matrix,
    load_qqq_prices,
    split_by_cutoff,
)
from .pca_diagnostics import (
    column_shift,
    fit_train_representation,
    frozen_score_shift,
    mp_support,
    transform_with_frozen_scaler,
)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _column_label(column) -> str:
    asset, indicator = column
    return f"{asset}|{indicator}"


def _model_dir():
    return MODEL_DIR / MODEL_VERSION


def _save_model(fitted: dict, feature_hash: str, price_hash: str) -> None:
    directory = _model_dir()
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite {directory}")
    directory.mkdir(parents=True)
    joblib.dump(fitted["scaler"], directory / "scaler.joblib")
    joblib.dump(fitted["pca"], directory / "pca.joblib")
    metadata = {
        "stage": 1,
        "model_version": MODEL_VERSION,
        "train_end": TRAIN_END.date().isoformat(),
        "feature_sha256": feature_hash,
        "qqq_sha256": price_hash,
        "n_obs": fitted["n_obs"],
        "n_features": fitted["n_features"],
        "gamma_p_over_n": fitted["gamma"],
        "dropped_zero_variance_columns": fitted["dropped_columns"],
        "kept_columns": [_column_label(column) for column in fitted["kept_columns"]],
        "scaler_n_samples_seen": int(fitted["scaler"].n_samples_seen_),
        "retention_rule_selected": None,
        "regime_count_selected": None,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": (
            "Exploratory PCA fit on the training window only. "
            "No component count, regime count, or trading rule is selected."
        ),
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def _load_model():
    directory = _model_dir()
    return {
        "scaler": joblib.load(directory / "scaler.joblib"),
        "pca": joblib.load(directory / "pca.joblib"),
        "metadata": json.loads((directory / "metadata.json").read_text()),
    }


def _scree_path():
    return REPORT_DIR / "stage1_scree_v1.png"


def _write_scree(eigenvalues: np.ndarray, gamma: float) -> None:
    path = _scree_path()
    if path.exists():
        return
    _, upper = mp_support(gamma, 1.0)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(np.arange(1, len(eigenvalues) + 1), eigenvalues, color="black", lw=1.2)
    axis.axhline(1.0, color="tab:blue", ls="--", lw=1, label="Kaiser threshold 1")
    axis.axhline(upper, color="tab:red", ls="--", lw=1, label=f"MP edge {upper:.2f} (sigma^2=1)")
    axis.set_yscale("log")
    axis.set_xlabel("Component")
    axis.set_ylabel("Correlation eigenvalue (1/n scale)")
    axis.set_title("Training-sample PCA spectrum through 2018-12-31")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _report_paths():
    return (
        REPORT_DIR / "stage1_diagnostics_v1.json",
        REPORT_DIR / "stage1_diagnostics_v1.md",
        REPORT_DIR / "stage1_eigenvalues_v1.csv",
        REPORT_DIR / "stage1_retention_v1.csv",
    )


def _write_reports(result: dict) -> None:
    json_path, markdown_path, eigenvalue_path, retention_path = _report_paths()
    if json_path.exists():
        return
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in result.items() if key != "eigenvalues"}
    json_path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n")
    eigenvalues = np.asarray(result["eigenvalues"])
    pd.DataFrame(
        {"component": np.arange(1, len(eigenvalues) + 1), "eigenvalue": eigenvalues}
    ).to_csv(eigenvalue_path, index=False)
    pd.DataFrame(result["retention"]).to_csv(retention_path, index=False)
    markdown_path.write_text(_markdown(result))


def _markdown(result: dict) -> str:
    alignment = result["alignment"]
    features = result["features"]
    mp_lower, mp_upper = result["mp_edge_sigma2_1"]
    lines = [
        "# Stage 1 diagnostics",
        "",
        "Training standardization and PCA only. No regime model and no trading rule.",
        "",
        "## Sample",
        "",
        f"- Feature source, read only: data/asset_indicator_matrix.csv ({features['n_train'] + features['n_test']} rows, {alignment['n_feature_columns']} columns).",
        f"- Archived snapshot: {result['snapshot_file']}. New copy this run: {result['snapshot_created']}.",
        f"- Dates: {alignment['feature_start']} through {alignment['feature_end']}.",
        f"- QQQ prices: {result['qqq_file']}, {alignment['price_start']} through {alignment['price_end']}.",
        f"- Feature dates missing QQQ: {alignment['n_feature_dates_missing_qqq']}.",
        f"- QQQ dates outside the feature calendar: {alignment['n_qqq_dates_outside_features']}.",
        f"- Train: {features['train_start']} through {features['train_end']} ({features['n_train']} rows).",
        f"- Test, held out of every fit: {features['test_start']} through {features['test_end']} ({features['n_test']} rows).",
        f"- Missing feature values: {features['missing_values']}. Infinite values: {features['infinite_values']}.",
        f"- Train columns with zero variance: {len(features['zero_variance_train_columns'])}.",
        f"- Exact duplicate column groups on the training window: {len(features['exact_duplicate_train_column_groups'])}.",
        "",
        "## PCA and Marchenko-Pastur",
        "",
        f"- Fitted columns: {result['n_features']}. gamma = p/n = {result['gamma']:.6f}.",
        f"- MP support at sigma^2 = 1: [{mp_lower:.6f}, {mp_upper:.6f}].",
        f"- Iterated bulk sigma^2: {result['iterated_noise']['sigma2']}. Converged: {result['iterated_noise']['converged']}. {result['iterated_noise']['reason']}.",
        f"- Largest eigenvalue: {result['largest_eigenvalue']:.6f}.",
        f"- Participation ratio (effective rank): {result['participation_ratio']:.2f} of {result['n_features']} components.",
        f"- Sklearn versus 1/n eigenvalue max absolute difference: {result['consistency']['max_abs_difference']:.3e}.",
        "",
        "| rule | components | variance share | definition |",
        "|---|---:|---:|---|",
    ]
    for row in result["retention"]:
        share = "—" if row["variance_share"] is None else f"{row['variance_share']:.4f}"
        count = "—" if row["n_components"] is None else str(row["n_components"])
        lines.append(f"| {row['rule']} | {count} | {share} | {row['definition']} |")
    lines.extend(
        [
            "",
            "No retention rule is selected.",
            "",
            "Test rows are standardized and projected with the frozen training basis. A shift here is a preprocessing finding, not a model-selection metric.",
            "",
            "## Frozen projection, first five components",
            "",
            "| component | train mean | train std | test mean | test std |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["score_shift"]:
        lines.append(
            f"| {row['component']} | {row['train_mean']:.4f} | {row['train_std']:.4f} | {row['test_mean']:.4f} | {row['test_std']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Largest frozen z-score shifts on the test window",
            "",
            "| column | test mean z | test std z |",
            "|---|---:|---:|",
        ]
    )
    for row in result["column_shift"]:
        lines.append(f"| {row['column']} | {row['test_mean_z']:.2f} | {row['test_std_z']:.2f} |")
    lines.extend(
        [
            "",
            "Gaps longer than five calendar days in the feature index:",
            "",
        ]
    )
    gaps = alignment["feature_gaps_longer_than_5_days"]
    if not gaps:
        lines.append("- None.")
    else:
        for gap in gaps:
            lines.append(f"- {gap['date']}: {gap['gap_days']} days")
    lines.append("")
    return "\n".join(lines)


def run_stage1() -> dict:
    """Read the research matrix, archive a new hash, and apply the frozen PCA.

    The source CSV is opened for reading. A hash that is already archived is
    not copied again. A saved model is reused. A schema or training-window
    mismatch raises and leaves the model directory unchanged.
    """
    features = load_feature_matrix(FEATURE_SOURCE)
    snapshot = archive_feature_matrix(FEATURE_SOURCE, features)
    prices = ensure_qqq_prices(features.index)
    qqq = load_qqq_prices(prices["path"])
    alignment = audit_alignment(features, qqq)
    feature_audit = audit_features(features, TRAIN_END)
    if alignment["n_feature_dates_missing_qqq"] != 0:
        raise RuntimeError("feature dates are missing QQQ prices; stopping before PCA")
    if feature_audit["missing_values"] or feature_audit["infinite_values"]:
        raise RuntimeError("feature matrix is not finite; stage 1 does not impute")

    train_index, test_index = split_by_cutoff(features.index, TRAIN_END)
    train_frame = features.loc[train_index]
    test_frame = features.loc[test_index]
    directory = _model_dir()
    if not directory.exists():
        fitted = fit_train_representation(train_frame)
        _save_model(fitted, snapshot["sha256"], prices["sha256"])
        standardized_train = fitted["standardized_train"]
        scaler = fitted["scaler"]
        pca = fitted["pca"]
        kept_columns = fitted["kept_columns"]
        eigenvalues = fitted["eigenvalues"]
        consistency = fitted["consistency"]
        retention = fitted["retention"]
        gamma = fitted["gamma"]
        iterated_noise = fitted["iterated_noise"]
        edge = fitted["mp_edge_sigma2_1"]
        dropped = fitted["dropped_columns"]
    else:
        saved = _load_model()
        scaler = saved["scaler"]
        pca = saved["pca"]
        kept_labels = saved["metadata"]["kept_columns"]
        known = kept_labels + list(saved["metadata"].get("dropped_zero_variance_columns", []))
        assert_schema_matches_model(features, kept_labels, known)
        label_map = {_column_label(column): column for column in features.columns}
        kept_columns = [label_map[label] for label in kept_labels]
        assert_training_matches_scaler(train_frame, scaler, kept_columns)
        standardized_train = transform_with_frozen_scaler(scaler, train_frame, kept_columns)
        from .pca_diagnostics import correlation_eigenvalues, eigenvalue_consistency, iterated_noise_variance, retention_comparison

        eigenvalues = correlation_eigenvalues(standardized_train)
        consistency = eigenvalue_consistency(eigenvalues, pca, len(train_index))
        gamma = len(kept_columns) / len(train_index)
        retention = retention_comparison(eigenvalues, gamma)
        iterated_noise = iterated_noise_variance(eigenvalues, gamma)
        edge = mp_support(gamma, 1.0)
        dropped = saved["metadata"]["dropped_zero_variance_columns"]

    standardized_test = transform_with_frozen_scaler(scaler, test_frame, kept_columns)
    result = {
        "feature_source": "data/asset_indicator_matrix.csv",
        "feature_source_sha256": snapshot["sha256"],
        "snapshot_file": snapshot["file"],
        "snapshot_sha256": snapshot["sha256"],
        "snapshot_created": snapshot["created"],
        "qqq_file": prices["path"].name,
        "qqq_sha256": prices["sha256"],
        "model_version": MODEL_VERSION,
        "alignment": alignment,
        "features": feature_audit,
        "dropped_zero_variance_columns": dropped,
        "n_features": int(len(kept_columns)),
        "gamma": float(gamma),
        "mp_edge_sigma2_1": edge,
        "iterated_noise": iterated_noise,
        "largest_eigenvalue": float(eigenvalues[0]),
        "participation_ratio": float((eigenvalues.sum() ** 2) / np.sum(eigenvalues ** 2)),
        "consistency": consistency,
        "retention": retention,
        "retention_rule_selected": None,
        "score_shift": frozen_score_shift(pca, standardized_train, standardized_test),
        "column_shift": column_shift(standardized_test, kept_columns),
        "eigenvalues": eigenvalues,
        "report_dir": str(REPORT_DIR),
        "model_dir": str(directory),
    }
    _write_scree(eigenvalues, gamma)
    _write_reports(result)
    return result
