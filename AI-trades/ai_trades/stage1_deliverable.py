"""Package the frozen price-relative scaler as the Stage 1 deliverable.

This module does not fit a scaler or a PCA. It reads the research matrix, applies
the existing price-relative formulas, and standardizes every row with the scaler
saved in models/stage2_price_relative_train2018_v1/.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from .constants import FEATURE_SOURCE, MODEL_DIR, PACKAGE_ROOT, TRAIN_END
from .consistency import INDICATOR_SPECS
from .data_prep import daily_index, load_feature_matrix, sha256_file, split_by_cutoff
from .preprocessing import FrozenPriceRelativePreprocessor
from .prices import ASSETS, ensure_asset_panels
from .stage2 import STAGE2_VERSION, _aligned_closes
from .transform import FORMULA_BY_FAMILY, apply_price_relative, missing_transform_columns

ARTIFACT_VERSION = "stage1_price_relative_standardized_v1"
ARTIFACT_DIR = PACKAGE_ROOT / "artifacts" / ARTIFACT_VERSION
ARTIFACT_VERSION_V2 = "stage1_price_relative_standardized_v2"
ARTIFACT_DIR_V2 = PACKAGE_ROOT / "artifacts" / ARTIFACT_VERSION_V2
STAGE2_DIR = MODEL_DIR / STAGE2_VERSION

MEAN_ATOL = 1e-8
VARIANCE_ATOL = 1e-8
MATCH_ATOL = 1e-8
CSV_ATOL = 1e-8


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _family_by_indicator() -> dict[str, tuple[str, str]]:
    return {
        name: (family, FORMULA_BY_FAMILY[family])
        for name, _function, _output, family in INDICATOR_SPECS
    }


def _column_records(columns) -> list[dict]:
    families = _family_by_indicator()
    records = []
    for position, (asset, indicator) in enumerate(columns):
        family, formula = families[str(indicator)]
        records.append(
            {
                "position": position,
                "asset": str(asset),
                "indicator": str(indicator),
                "family": family,
                "formula": formula,
                "requires_same_day_adjusted_close": family != "scale_free",
            }
        )
    return records


def _check(name: str, passed: bool, detail: dict) -> dict:
    return {"name": name, "passed": bool(passed), **detail}


def build_standardized_matrix(features: pd.DataFrame, closes: pd.DataFrame, scaler, columns) -> pd.DataFrame:
    preprocessor = FrozenPriceRelativePreprocessor(scaler, columns)
    return preprocessor.transform(features, closes)


def validate_export(
    exported: pd.DataFrame,
    features: pd.DataFrame,
    closes: pd.DataFrame,
    scaler,
    pca,
    stage2_metadata: dict,
    feature_sha: str,
    adjusted_sha: str,
) -> list[dict]:
    """Checks against the frozen Stage 2 scaler. Nothing here is refit."""
    train_index, test_index = split_by_cutoff(features.index, TRAIN_END)
    relative = apply_price_relative(features, closes)
    columns = list(exported.columns)
    train_relative = relative.loc[train_index, columns].to_numpy(dtype=float)
    exported_train = exported.loc[train_index].to_numpy(dtype=float)
    exported_test = exported.loc[test_index].to_numpy(dtype=float)
    manual_train = (train_relative - scaler.mean_) / scaler.scale_
    frozen_test = scaler.transform(relative.loc[test_index, columns].to_numpy(dtype=float))
    train_mean = exported_train.mean(axis=0)
    train_variance = exported_train.var(axis=0, ddof=0)
    pipeline = FrozenPriceRelativePreprocessor(scaler, columns)
    replay = pipeline.transform(features, closes).to_numpy(dtype=float)
    checks = [
        _check(
            "shape",
            exported.shape == (6025, 228),
            {"rows": int(exported.shape[0]), "columns": int(exported.shape[1])},
        ),
        _check(
            "finite",
            bool(np.isfinite(exported.to_numpy(dtype=float)).all()),
            {"nonfinite": int((~np.isfinite(exported.to_numpy(dtype=float))).sum())},
        ),
        _check(
            "training_mean_zero",
            float(np.max(np.abs(train_mean))) <= MEAN_ATOL,
            {
                "rows": int(len(train_index)),
                "max_abs_column_mean": float(np.max(np.abs(train_mean))),
                "tolerance": MEAN_ATOL,
                "divisor": "population mean of the exported training rows",
            },
        ),
        _check(
            "training_unit_variance",
            float(np.max(np.abs(train_variance - 1.0))) <= VARIANCE_ATOL,
            {
                "rows": int(len(train_index)),
                "max_abs_variance_gap": float(np.max(np.abs(train_variance - 1.0))),
                "tolerance": VARIANCE_ATOL,
                "divisor": "population variance, ddof=0, matching StandardScaler",
            },
        ),
        _check(
            "test_rows_use_frozen_scaler",
            int(scaler.n_samples_seen_) == 4044 and np.allclose(exported_test, frozen_test, atol=MATCH_ATOL, rtol=0),
            {
                "test_rows": int(len(test_index)),
                "scaler_n_samples_seen": int(scaler.n_samples_seen_),
                "max_abs_difference_from_frozen_transform": float(np.max(np.abs(exported_test - frozen_test))),
                "max_abs_test_column_mean": float(np.max(np.abs(exported_test.mean(axis=0)))),
                "full_sample_standardization": False,
            },
        ),
        _check(
            "matches_stage2_pca_input",
            np.allclose(exported_train, manual_train, atol=MATCH_ATOL, rtol=0)
            and np.allclose(train_relative.mean(axis=0), scaler.mean_, atol=MATCH_ATOL, rtol=0)
            and np.allclose(train_relative.std(axis=0, ddof=0), scaler.scale_, atol=MATCH_ATOL, rtol=0)
            and int(getattr(pca, "n_features_in_", exported.shape[1])) == exported.shape[1]
            and np.allclose(exported_train.mean(axis=0), np.asarray(pca.mean_, dtype=float), atol=1e-6, rtol=0),
            {
                "max_abs_difference_from_scaler_affine_map": float(np.max(np.abs(exported_train - manual_train))),
                "max_abs_training_mean_gap": float(np.max(np.abs(train_relative.mean(axis=0) - scaler.mean_))),
                "max_abs_training_scale_gap": float(
                    np.max(np.abs(train_relative.std(axis=0, ddof=0) - scaler.scale_))
                ),
                "max_abs_gap_from_pca_training_mean": float(
                    np.max(np.abs(exported_train.mean(axis=0) - np.asarray(pca.mean_, dtype=float)))
                ),
            },
        ),
        _check(
            "pipeline_reproduces_matrix",
            np.allclose(replay, exported.to_numpy(dtype=float), atol=0, rtol=0),
            {"max_abs_difference": float(np.max(np.abs(replay - exported.to_numpy(dtype=float))))},
        ),
        _check(
            "provenance",
            feature_sha == stage2_metadata["feature_sha256"]
            and adjusted_sha == stage2_metadata["adjusted_ohlc_sha256"]
            and list(exported.columns) == list(features.columns)
            and exported.index.equals(features.index)
            and stage2_metadata["train_end"] == TRAIN_END.date().isoformat()
            and stage2_metadata["price_convention"].startswith("Ticker.history(period=max, interval=1d)"),
            {
                "feature_sha256": feature_sha,
                "stage2_feature_sha256": stage2_metadata["feature_sha256"],
                "adjusted_ohlc_sha256": adjusted_sha,
                "stage2_adjusted_ohlc_sha256": stage2_metadata["adjusted_ohlc_sha256"],
                "column_order_matches_source": list(exported.columns) == list(features.columns),
                "dates_match_source": bool(exported.index.equals(features.index)),
                "train_end": stage2_metadata["train_end"],
                "price_convention": stage2_metadata["price_convention"],
            },
        ),
    ]
    return checks


def _markdown(checks: list[dict], metadata: dict) -> str:
    lines = [
        "# Stage 1 validation",
        "",
        "Packaging of the frozen price-relative transform and the Stage 2 training scaler.",
        "The scaler was not refit. Full-sample standardization was not performed.",
        "",
        f"- Artifact: `{metadata['artifact_version']}`",
        f"- Created: {metadata['created_at_utc']}",
        f"- Source matrix: `{metadata['feature_source']}`",
        f"- Source SHA-256: `{metadata['feature_sha256']}`",
        f"- Adjusted OHLC SHA-256: `{metadata['adjusted_ohlc_sha256']}`",
        f"- Frozen scaler: `{metadata['frozen_scaler_path']}`",
        f"- Frozen scaler SHA-256: `{metadata['frozen_scaler_sha256']}`",
        f"- Training cutoff, inclusive: {metadata['train_end']}",
        f"- Training rows seen by the scaler: {metadata['scaler_n_samples_seen']}",
        f"- Test rows transformed with that scaler: {metadata['n_test_rows']}",
        "",
        "The CSV keeps the MultiIndex columns. The first header row is the asset and the second is the indicator. The date index is the first column. Read it with `pandas.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)`.",
        "",
        "| Check | Passed | Detail |",
        "|---|---|---|",
    ]
    for check in checks:
        detail = {key: value for key, value in check.items() if key not in {"name", "passed"}}
        rendered = json.dumps(detail, sort_keys=True)
        lines.append(f"| {check['name']} | {check['passed']} | `{rendered}` |")
    lines.extend(
        [
            "",
            "Stage 1 of the Capstone LDI roadmap is not complete. The project owner has not approved it.",
            "",
        ]
    )
    return "\n".join(lines)


def export_stage1_deliverable(destination=ARTIFACT_DIR) -> dict:
    """Write the Stage 1 artifact directory. Refuse to replace an existing one."""
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    if not STAGE2_DIR.is_dir():
        raise FileNotFoundError(f"missing frozen model {STAGE2_DIR}")
    scaler_path = STAGE2_DIR / "scaler.joblib"
    pca_path = STAGE2_DIR / "pca.joblib"
    metadata_path = STAGE2_DIR / "metadata.json"
    scaler_hash_before = sha256_file(scaler_path)
    pca_hash_before = sha256_file(pca_path)
    stage2_metadata = json.loads(metadata_path.read_text())
    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)
    seen_before = int(scaler.n_samples_seen_)
    mean_before = np.array(scaler.mean_, dtype=float, copy=True)

    features = load_feature_matrix(FEATURE_SOURCE)
    feature_sha = sha256_file(FEATURE_SOURCE)
    panels = ensure_asset_panels()
    if panels["created"]:
        raise RuntimeError("Refusing to export against a newly downloaded price panel.")
    closes = _aligned_closes(panels["adjusted"], features.index)
    if closes[ASSETS].isna().any().any():
        raise RuntimeError("Adjusted close does not cover every feature date.")
    missing = missing_transform_columns(features)
    if missing:
        raise RuntimeError(f"Feature matrix is missing transform inputs: {', '.join(missing[:8])}")

    exported = build_standardized_matrix(features, closes[ASSETS], scaler, list(features.columns))
    if int(scaler.n_samples_seen_) != seen_before or not np.array_equal(scaler.mean_, mean_before):
        raise RuntimeError("The frozen scaler changed during export.")
    checks = validate_export(
        exported,
        features,
        closes[ASSETS],
        scaler,
        pca,
        stage2_metadata,
        feature_sha,
        panels["adjusted_sha256"],
    )
    failed = [check["name"] for check in checks if not check["passed"]]
    if failed:
        raise RuntimeError("Stage 1 export checks failed: " + ", ".join(failed))

    destination.mkdir(parents=True)
    try:
        matrix_path = destination / "feature_matrix_standardized.csv"
        exported.to_csv(matrix_path)
        reread = pd.read_csv(matrix_path, header=[0, 1], index_col=0, parse_dates=True)
        reread.index = daily_index(reread.index)
        reread.index.name = "date"
        csv_gap = float(np.max(np.abs(reread.to_numpy(dtype=float) - exported.to_numpy(dtype=float))))
        csv_check = _check(
            "csv_roundtrip",
            csv_gap <= CSV_ATOL and list(reread.columns) == list(exported.columns) and reread.index.equals(exported.index),
            {"max_abs_difference": csv_gap, "tolerance": CSV_ATOL},
        )
        checks.append(csv_check)
        if not csv_check["passed"]:
            raise RuntimeError(f"CSV roundtrip exceeded tolerance: {csv_gap}")

        pipeline = FrozenPriceRelativePreprocessor(scaler, list(features.columns))
        pipeline_path = destination / "preprocessing_pipeline.joblib"
        joblib.dump(pipeline, pipeline_path)
        reloaded = joblib.load(pipeline_path)
        reloaded_values = reloaded.transform(features, closes[ASSETS]).to_numpy(dtype=float)
        reload_gap = float(np.max(np.abs(reloaded_values - exported.to_numpy(dtype=float))))
        refused = False
        try:
            reloaded.fit()
        except RuntimeError:
            refused = True
        reload_check = _check(
            "reloaded_pipeline_reproduces_matrix",
            reload_gap <= MATCH_ATOL and refused and int(reloaded.scaler.n_samples_seen_) == 4044,
            {
                "max_abs_difference": reload_gap,
                "fit_refused": refused,
                "scaler_n_samples_seen": int(reloaded.scaler.n_samples_seen_),
            },
        )
        checks.append(reload_check)
        if not reload_check["passed"]:
            raise RuntimeError("Reloaded pipeline did not reproduce the exported matrix.")
    except Exception:
        if destination.exists():
            shutil.rmtree(destination)
        raise

    if sha256_file(scaler_path) != scaler_hash_before or sha256_file(pca_path) != pca_hash_before:
        raise RuntimeError("Frozen Stage 2 model files changed during export.")

    metadata = {
        "roadmap_stage": 1,
        "artifact_version": ARTIFACT_VERSION,
        "status": "exported_pending_project_owner_approval",
        "stage_complete": False,
        "created_at_utc": _utc_now(),
        "feature_source": "data/asset_indicator_matrix.csv",
        "feature_sha256": feature_sha,
        "adjusted_ohlc_path": "AI-trades/data/asset_ohlc_main_convention_v1.csv",
        "adjusted_ohlc_sha256": panels["adjusted_sha256"],
        "raw_ohlc_sha256": panels["raw_sha256"],
        "frozen_model_version": STAGE2_VERSION,
        "frozen_scaler_path": "AI-trades/models/stage2_price_relative_train2018_v1/scaler.joblib",
        "frozen_scaler_sha256": scaler_hash_before,
        "frozen_pca_was_not_used_to_transform": True,
        "train_end": TRAIN_END.date().isoformat(),
        "n_rows": int(exported.shape[0]),
        "n_columns": int(exported.shape[1]),
        "n_train_rows": 4044,
        "n_test_rows": 1981,
        "scaler_n_samples_seen": seen_before,
        "standardization": (
            "Each column was centered and scaled with the mean and population standard deviation "
            "of rows on or before 2018-12-31. Later rows were transformed with those frozen values. "
            "Full-sample standardization was not performed."
        ),
        "price_convention": stage2_metadata["price_convention"],
        "adjusted_close_requirement": (
            "Level columns use L/C - 1 and width columns use V/C, where C is the same asset's "
            "adjusted close on the same date. Scale-free columns ignore the close."
        ),
        "serialization": {
            "path": "feature_matrix_standardized.csv",
            "header_rows": ["asset", "indicator"],
            "index_column": "date",
            "read": "pandas.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)",
        },
        "columns": _column_records(exported.columns),
        "matrix_sha256": sha256_file(matrix_path),
        "checks": checks,
    }
    (destination / "feature_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (destination / "validation_report.md").write_text(_markdown(checks, metadata))
    return {"directory": str(destination), "checks": checks, "metadata": metadata}


def _v1_hashes() -> dict[str, str]:
    paths = [
        ARTIFACT_DIR / "preprocessing_pipeline.joblib",
        ARTIFACT_DIR / "feature_metadata.json",
        ARTIFACT_DIR / "feature_matrix_standardized.csv",
        ARTIFACT_DIR / "validation_report.md",
    ]
    return {str(path.relative_to(PACKAGE_ROOT)): sha256_file(path) for path in paths}


def _load_historical_v1_pipeline():
    """Read the v1 pickle. Its class is stored as ``__main__``.

    This alias is used only to open that historical file. The v2 artifact is
    saved from ``ai_trades.preprocessing`` and is loaded without it.
    """
    import __main__

    __main__.FrozenPriceRelativePreprocessor = FrozenPriceRelativePreprocessor
    return joblib.load(ARTIFACT_DIR / "preprocessing_pipeline.joblib")


def repackage_portable_v2(destination=ARTIFACT_DIR_V2) -> dict:
    """Wrap the v1 scaler in the importable class. Do not refit it."""
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    before = _v1_hashes()
    historical = _load_historical_v1_pipeline()
    seen = int(historical.scaler.n_samples_seen_)
    if seen != 4044:
        raise RuntimeError(f"frozen scaler records {seen} rows, not 4044")
    mean = np.array(historical.scaler.mean_, dtype=float, copy=True)
    scale = np.array(historical.scaler.scale_, dtype=float, copy=True)
    portable = FrozenPriceRelativePreprocessor(historical.scaler, historical.columns)
    if type(portable).__module__ != "ai_trades.preprocessing":
        raise RuntimeError(f"class module is {type(portable).__module__}")
    destination.mkdir(parents=True)
    try:
        joblib.dump(portable, destination / "preprocessing_pipeline.joblib")
        shutil.copyfile(
            ARTIFACT_DIR / "feature_matrix_standardized.csv",
            destination / "feature_matrix_standardized.csv",
        )
        copied_hash = sha256_file(destination / "feature_matrix_standardized.csv")
        if copied_hash != before["artifacts/stage1_price_relative_standardized_v1/feature_matrix_standardized.csv"]:
            raise RuntimeError("v2 matrix is not a byte copy of v1")
        if int(historical.scaler.n_samples_seen_) != 4044 or not np.array_equal(historical.scaler.mean_, mean):
            raise RuntimeError("repackaging changed the frozen scaler")
        if not np.array_equal(historical.scaler.scale_, scale):
            raise RuntimeError("repackaging changed the frozen scale")
        prior = json.loads((ARTIFACT_DIR / "feature_metadata.json").read_text())
        metadata = {
            "roadmap_stage": 1,
            "artifact_version": ARTIFACT_VERSION_V2,
            "predecessor": ARTIFACT_VERSION,
            "status": "exported_pending_project_owner_approval",
            "stage_complete": False,
            "created_at_utc": _utc_now(),
            "class_module": type(portable).__module__,
            "class_name": type(portable).__name__,
            "serialization_note": (
                "v1 stored the class as __main__.FrozenPriceRelativePreprocessor. "
                "v2 stores ai_trades.preprocessing.FrozenPriceRelativePreprocessor. "
                "The scaler was copied, not refit. Full-sample standardization was not performed."
            ),
            "feature_source": prior["feature_source"],
            "feature_sha256": prior["feature_sha256"],
            "adjusted_ohlc_sha256": prior["adjusted_ohlc_sha256"],
            "raw_ohlc_sha256": prior["raw_ohlc_sha256"],
            "frozen_model_version": prior["frozen_model_version"],
            "frozen_scaler_sha256": prior["frozen_scaler_sha256"],
            "train_end": prior["train_end"],
            "n_rows": prior["n_rows"],
            "n_columns": prior["n_columns"],
            "n_train_rows": prior["n_train_rows"],
            "n_test_rows": prior["n_test_rows"],
            "scaler_n_samples_seen": seen,
            "standardization": prior["standardization"],
            "price_convention": prior["price_convention"],
            "adjusted_close_requirement": prior["adjusted_close_requirement"],
            "serialization": prior["serialization"],
            "columns": prior["columns"],
            "matrix_sha256": copied_hash,
            "matrix_copied_byte_for_byte_from_v1": True,
            "full_sample_standardization": False,
        }
        (destination / "feature_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    except Exception:
        if destination.exists():
            shutil.rmtree(destination)
        raise
    if _v1_hashes() != before:
        raise RuntimeError("v1 artifacts changed during repackaging")
    return {"v1_hashes_before": before, "metadata": metadata}


def verify_v2_fresh_inference() -> dict:
    """Load v2 with ordinary joblib and transform the last 10 dates only."""
    import __main__

    if hasattr(__main__, "FrozenPriceRelativePreprocessor"):
        raise RuntimeError("fresh inference found a __main__ class alias")
    pipeline = joblib.load(ARTIFACT_DIR_V2 / "preprocessing_pipeline.joblib")
    seen_before = int(pipeline.scaler.n_samples_seen_)
    mean_before = np.array(pipeline.scaler.mean_, dtype=float, copy=True)
    scale_before = np.array(pipeline.scaler.scale_, dtype=float, copy=True)
    features = load_feature_matrix(FEATURE_SOURCE)
    panels = ensure_asset_panels()
    if panels["created"]:
        raise RuntimeError("price panel was created during verification")
    closes = _aligned_closes(panels["adjusted"], features.index)
    dates = list(features.index[-10:])
    batch_features = features.loc[dates].copy()
    batch_closes = closes.loc[dates, ASSETS].copy()
    reference = pd.read_csv(
        ARTIFACT_DIR / "feature_matrix_standardized.csv",
        header=[0, 1],
        index_col=0,
        parse_dates=True,
    )
    reference.index = daily_index(reference.index)
    first = pipeline.transform(batch_features, batch_closes)
    second = pipeline.transform(batch_features, batch_closes)
    expected = reference.loc[dates]
    difference = np.abs(first.to_numpy(dtype=float) - expected.to_numpy(dtype=float))
    repeat = np.abs(first.to_numpy(dtype=float) - second.to_numpy(dtype=float))
    max_abs = float(difference.max())
    checks = {
        "class_module": type(pipeline).__module__ == "ai_trades.preprocessing",
        "shape_10_by_228": first.shape == (10, 228),
        "dates_and_columns_match": bool(first.index.equals(expected.index) and list(first.columns) == list(expected.columns)),
        "all_finite": bool(np.isfinite(first.to_numpy(dtype=float)).all()),
        "max_abs_difference_below_1e-10": max_abs < 1e-10,
        "repeated_inference_identical": float(repeat.max()) == 0.0,
        "scaler_still_4044": seen_before == 4044 and int(pipeline.scaler.n_samples_seen_) == 4044,
        "scaler_parameters_unchanged": bool(
            np.array_equal(pipeline.scaler.mean_, mean_before) and np.array_equal(pipeline.scaler.scale_, scale_before)
        ),
        "held_out_dates": bool(batch_features.index.min() > TRAIN_END),
    }
    return {
        "class_module": type(pipeline).__module__,
        "class_name": type(pipeline).__name__,
        "input_start": batch_features.index.min().date().isoformat(),
        "input_end": batch_features.index.max().date().isoformat(),
        "input_dates": [stamp.date().isoformat() for stamp in batch_features.index],
        "input_feature_shape": list(batch_features.shape),
        "input_close_shape": list(batch_closes.shape),
        "output_shape": list(first.shape),
        "max_abs_difference": max_abs,
        "max_abs_repeat_difference": float(repeat.max()),
        "n_samples_seen_before": seen_before,
        "n_samples_seen_after": int(pipeline.scaler.n_samples_seen_),
        "checks": checks,
        "all_passed": all(checks.values()),
    }


def _fresh_python(code: str) -> str:
    import os
    import subprocess
    import sys

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PACKAGE_ROOT)
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PACKAGE_ROOT.parent,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    return completed.stdout.strip()


def publish_portable_v2() -> dict:
    """Create v2, then load and test it in fresh processes."""
    before = _v1_hashes()
    packaged = repackage_portable_v2()
    load_stdout = _fresh_python(
        "import joblib, __main__\n"
        "assert not hasattr(__main__, 'FrozenPriceRelativePreprocessor')\n"
        f"obj = joblib.load(r'{(ARTIFACT_DIR_V2 / 'preprocessing_pipeline.joblib').as_posix()}')\n"
        "assert type(obj).__module__ == 'ai_trades.preprocessing'\n"
        "assert type(obj).__name__ == 'FrozenPriceRelativePreprocessor'\n"
        "assert int(obj.scaler.n_samples_seen_) == 4044\n"
        "print(type(obj).__module__ + '.' + type(obj).__name__)\n"
    )
    smoke_stdout = _fresh_python(
        "import json\n"
        "from ai_trades.stage1_deliverable import verify_v2_fresh_inference\n"
        "result = verify_v2_fresh_inference()\n"
        "assert result['all_passed']\n"
        "print(json.dumps(result))\n"
    )
    smoke = json.loads(smoke_stdout)
    if _v1_hashes() != before:
        raise RuntimeError("v1 artifacts changed during verification")
    report = _v2_report(packaged["metadata"], smoke, before, load_stdout)
    (ARTIFACT_DIR_V2 / "validation_report.md").write_text(report)
    if _v1_hashes() != before:
        raise RuntimeError("v1 artifacts changed while writing the v2 report")
    return {
        "load_stdout": load_stdout,
        "smoke": smoke,
        "v1_hashes": before,
        "v2_pipeline_sha256": sha256_file(ARTIFACT_DIR_V2 / "preprocessing_pipeline.joblib"),
        "v2_matrix_sha256": sha256_file(ARTIFACT_DIR_V2 / "feature_matrix_standardized.csv"),
        "v2_metadata_sha256": sha256_file(ARTIFACT_DIR_V2 / "feature_metadata.json"),
    }


def _v2_report(metadata: dict, smoke: dict, v1_hashes: dict, load_stdout: str) -> str:
    lines = [
        "# Stage 1 v2 validation",
        "",
        "Portable repackaging of the v1 preprocessor. The scaler was not refit.",
        "Full-sample standardization was not performed. Stage 1 is not approved.",
        "",
        f"- Artifact: `{metadata['artifact_version']}`",
        f"- Created: {metadata['created_at_utc']}",
        f"- Class: `{load_stdout}`",
        f"- Predecessor: `{metadata['predecessor']}`",
        f"- Scaler samples seen: {metadata['scaler_n_samples_seen']}",
        f"- Matrix SHA-256, byte copy of v1: `{metadata['matrix_sha256']}`",
        "",
        "A fresh Python process called `joblib.load` on `preprocessing_pipeline.joblib` without registering a class on `__main__`. It resolved `ai_trades.preprocessing.FrozenPriceRelativePreprocessor`.",
        "",
        f"The same kind of fresh process then transformed {smoke['input_start']} through {smoke['input_end']} ({len(smoke['input_dates'])} sessions, all after 2018-12-31). The transform received a {smoke['input_feature_shape'][0]} × {smoke['input_feature_shape'][1]} feature frame and a {smoke['input_close_shape'][0]} × {smoke['input_close_shape'][1]} close frame.",
        "",
        "| Check | Passed | Detail |",
        "|---|---|---|",
    ]
    details = {
        "class_module": smoke["class_module"],
        "shape_10_by_228": smoke["output_shape"],
        "dates_and_columns_match": smoke["input_dates"],
        "all_finite": smoke["checks"]["all_finite"],
        "max_abs_difference_below_1e-10": smoke["max_abs_difference"],
        "repeated_inference_identical": smoke["max_abs_repeat_difference"],
        "scaler_still_4044": [smoke["n_samples_seen_before"], smoke["n_samples_seen_after"]],
        "scaler_parameters_unchanged": smoke["checks"]["scaler_parameters_unchanged"],
        "held_out_dates": smoke["checks"]["held_out_dates"],
    }
    for name, passed in smoke["checks"].items():
        lines.append(f"| {name} | {passed} | `{json.dumps(details[name])}` |")
    lines.extend(
        [
            "",
            "v1 SHA-256 values were unchanged by this repackaging.",
            "",
            "| v1 file | SHA-256 |",
            "|---|---|",
        ]
    )
    for path, digest in v1_hashes.items():
        lines.append(f"| `{path}` | `{digest}` |")
    lines.extend(
        [
            "",
            "No fitting method was invoked. The check is that `n_samples_seen_` stayed 4,044 and the saved mean and scale vectors were unchanged after both transforms. A fit on these 10 rows would have replaced those values.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    result = export_stage1_deliverable()
    for check in result["checks"]:
        print(f"{check['passed']}\t{check['name']}")
