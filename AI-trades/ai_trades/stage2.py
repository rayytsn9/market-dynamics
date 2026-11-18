"""Stage 2: price-relative features and a new training-only PCA.

This module reads the research matrix and writes only under its own model and
report names. It does not open the Stage 1 model directory for writing.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .constants import FEATURE_SOURCE, MODEL_DIR, REPORT_DIR, TRAIN_END
from .consistency import ConsistencyError, verify_prices_and_units
from .data_prep import daily_index, load_feature_matrix, sha256_file, split_by_cutoff
from .pca_diagnostics import (
    column_shift,
    fit_train_representation,
    frozen_score_shift,
    mp_support,
    transform_with_frozen_scaler,
)
from .prices import ASSETS, close_frame, ensure_asset_panels
from .transform import apply_price_relative, missing_transform_columns, transformation_table

STAGE2_VERSION = "stage2_price_relative_train2018_v1"
STAGE1_VERSION = "stage1_train2018_v1"


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


def _spectrum(fitted: dict, standardized_test: np.ndarray) -> dict:
    eigenvalues = fitted["eigenvalues"]
    shift = frozen_score_shift(fitted["pca"], fitted["standardized_train"], standardized_test)
    return {
        "n_features": fitted["n_features"],
        "gamma": fitted["gamma"],
        "largest_eigenvalue": float(eigenvalues[0]),
        "participation_ratio": float((eigenvalues.sum() ** 2) / np.sum(eigenvalues ** 2)),
        "mp_edge_sigma2_1": fitted["mp_edge_sigma2_1"],
        "iterated_noise": fitted["iterated_noise"],
        "retention": fitted["retention"],
        "consistency": fitted["consistency"],
        "score_shift": shift,
        "max_abs_pc_test_mean": float(max(abs(row["test_mean"]) for row in shift)),
        "pc1_test_mean": float(shift[0]["test_mean"]),
        "pc1_train_std": float(shift[0]["train_std"]),
    }


def _max_abs_mean_z(train: pd.DataFrame, test: pd.DataFrame) -> float:
    scale = train.std(axis=0, ddof=0).replace(0, np.nan)
    zscores = (test.mean(axis=0) - train.mean(axis=0)) / scale
    return float(zscores.abs().max())


def _aligned_closes(adjusted: dict[str, pd.DataFrame], index) -> pd.DataFrame:
    closes = close_frame(adjusted).apply(lambda column: column.dropna())
    closes.index = daily_index(closes.index)
    closes = closes.groupby(level=0).last()
    return closes.reindex(daily_index(index))


def _save_model(fitted: dict, metadata: dict) -> None:
    directory = MODEL_DIR / STAGE2_VERSION
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite {directory}")
    if directory.name == STAGE1_VERSION:
        raise RuntimeError("stage 2 tried to write the stage 1 model directory")
    directory.mkdir(parents=True)
    joblib.dump(fitted["scaler"], directory / "scaler.joblib")
    joblib.dump(fitted["pca"], directory / "pca.joblib")
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def _write_scree(eigenvalues: np.ndarray, gamma: float) -> None:
    path = REPORT_DIR / "stage2_scree_v1.png"
    if path.exists():
        return
    _, upper = mp_support(gamma, 1.0)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(np.arange(1, len(eigenvalues) + 1), eigenvalues, color="black", lw=1.2)
    axis.axhline(1.0, color="tab:blue", ls="--", lw=1, label="Kaiser threshold 1")
    axis.axhline(upper, color="tab:red", ls="--", lw=1, label=f"MP edge {upper:.2f}")
    axis.set_yscale("log")
    axis.set_xlabel("Component")
    axis.set_ylabel("Correlation eigenvalue (1/n scale)")
    axis.set_title("Transformed training PCA through 2018-12-31")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _write_reports(result: dict) -> None:
    json_path = REPORT_DIR / "stage2_diagnostics_v1.json"
    if json_path.exists():
        return
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in result.items() if key != "eigenvalues"}
    json_path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n")
    eigenvalues = np.asarray(result["eigenvalues"])
    pd.DataFrame({"component": np.arange(1, len(eigenvalues) + 1), "eigenvalue": eigenvalues}).to_csv(
        REPORT_DIR / "stage2_eigenvalues_v1.csv", index=False
    )
    pd.DataFrame(result["transformed"]["retention"]).to_csv(REPORT_DIR / "stage2_retention_v1.csv", index=False)
    (REPORT_DIR / "stage2_diagnostics_v1.md").write_text(_markdown(result))


def _markdown(result: dict) -> str:
    baseline = result["baseline"]
    transformed = result["transformed"]
    lines = [
        "# Stage 2 transformed PCA",
        "",
        "Price-relative features, training fit through 2018-12-31. Stage 1 artifacts were not modified.",
        "",
        f"- Feature source hash: {result['feature_sha256']}",
        f"- Adjusted OHLC hash: {result['adjusted_sha256']}",
        f"- Level columns, formula L/C - 1: {result['n_level_columns']}",
        f"- Width columns, formula V/C: {result['n_width_columns']}",
        f"- Scale-free columns, unchanged: {result['n_scale_free_columns']}",
        f"- Minimum reconstruction correlation against main.py prices: {result['min_reconstruction_correlation']:.6f}",
        "",
        "| | Stage 1 raw | Stage 2 transformed |",
        "|---|---:|---:|",
        f"| Largest eigenvalue | {baseline['largest_eigenvalue']:.2f} | {transformed['largest_eigenvalue']:.2f} |",
        f"| Participation ratio | {baseline['participation_ratio']:.2f} | {transformed['participation_ratio']:.2f} |",
        f"| PC1 test mean | {baseline['pc1_test_mean']:.2f} | {transformed['pc1_test_mean']:.2f} |",
        f"| PC1 training std | {baseline['pc1_train_std']:.2f} | {transformed['pc1_train_std']:.2f} |",
        f"| Max absolute column mean z | {result['baseline_max_abs_mean_z']:.2f} | {result['transformed_max_abs_mean_z']:.2f} |",
        "",
        "| rule | components | variance share |",
        "|---|---:|---:|",
    ]
    for row in transformed["retention"]:
        share = "—" if row["variance_share"] is None else f"{row['variance_share']:.4f}"
        count = "—" if row["n_components"] is None else str(row["n_components"])
        lines.append(f"| {row['rule']} | {count} | {share} |")
    lines.extend(["", "No component count is selected.", ""])
    return "\n".join(lines)


def run_stage2() -> dict:
    """Verify prices, transform price-unit columns, and fit a separate PCA."""
    features = load_feature_matrix(FEATURE_SOURCE)
    panels = ensure_asset_panels()
    audit = verify_prices_and_units(features, panels["adjusted"], panels["raw"])
    if not audit["passed"]:
        preview = "; ".join(audit["issues"][:8])
        raise ConsistencyError(
            "Stopping before the transformed PCA. Price or unit checks failed: " + preview
        )
    closes = _aligned_closes(panels["adjusted"], features.index)
    if closes[ASSETS].isna().any().any():
        raise ConsistencyError("Stopping. Adjusted close does not cover every feature date.")
    missing = missing_transform_columns(features)
    if missing:
        preview = ", ".join(missing[:8])
        raise ConsistencyError(f"Stopping. The feature matrix is missing transform inputs: {preview}")
    transformed = apply_price_relative(features, closes[ASSETS])
    train_index, test_index = split_by_cutoff(features.index, TRAIN_END)
    raw_fit = fit_train_representation(features.loc[train_index])
    raw_test = transform_with_frozen_scaler(raw_fit["scaler"], features.loc[test_index], raw_fit["kept_columns"])
    fitted = fit_train_representation(transformed.loc[train_index])
    standardized_test = transform_with_frozen_scaler(
        fitted["scaler"], transformed.loc[test_index], fitted["kept_columns"]
    )
    correlations = [
        row["min_adjusted_correlation"]
        for row in audit["indicators"]
        if row["min_adjusted_correlation"] is not None
    ]
    metadata = {
        "stage": 2,
        "model_version": STAGE2_VERSION,
        "train_end": TRAIN_END.date().isoformat(),
        "feature_source": "data/asset_indicator_matrix.csv",
        "feature_sha256": sha256_file(FEATURE_SOURCE),
        "adjusted_ohlc_sha256": panels["adjusted_sha256"],
        "raw_ohlc_sha256": panels["raw_sha256"],
        "n_obs": fitted["n_obs"],
        "n_features": fitted["n_features"],
        "scaler_n_samples_seen": int(fitted["scaler"].n_samples_seen_),
        "formulas": transformation_table(),
        "price_convention": "Ticker.history(period=max, interval=1d) with yfinance default auto_adjust=True",
        "retention_rule_selected": None,
        "regime_count_selected": None,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": "Transformed PCA only. Stage 1 model was not modified. No clustering or trading rule.",
    }
    directory = MODEL_DIR / STAGE2_VERSION
    if not directory.exists():
        _save_model(fitted, metadata)
    result = {
        "model_version": STAGE2_VERSION,
        "feature_sha256": metadata["feature_sha256"],
        "adjusted_sha256": panels["adjusted_sha256"],
        "n_level_columns": audit["n_level_columns"],
        "n_width_columns": audit["n_width_columns"],
        "n_scale_free_columns": audit["n_scale_free_columns"],
        "min_reconstruction_correlation": float(min(correlations)),
        "baseline": _spectrum(raw_fit, raw_test),
        "transformed": _spectrum(fitted, standardized_test),
        "baseline_max_abs_mean_z": _max_abs_mean_z(features.loc[train_index], features.loc[test_index]),
        "transformed_max_abs_mean_z": _max_abs_mean_z(transformed.loc[train_index], transformed.loc[test_index]),
        "column_shift": column_shift(standardized_test, fitted["kept_columns"]),
        "eigenvalues": fitted["eigenvalues"],
        "retention_rule_selected": None,
        "model_dir": str(directory),
        "report_dir": str(REPORT_DIR),
    }
    _write_scree(fitted["eigenvalues"], fitted["gamma"])
    _write_reports(result)
    return result
