"""Archived snapshots and the frozen-model guards."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ai_trades.data_prep import (
    FeatureSchemaError,
    FrozenTrainingMismatch,
    archive_feature_matrix,
    assert_schema_matches_model,
    assert_training_matches_scaler,
    load_feature_matrix,
    sha256_file,
)


def _frame(rows, columns):
    index = pd.bdate_range("2018-12-20", periods=rows)
    values = np.arange(rows * len(columns), dtype=float).reshape(rows, len(columns))
    frame = pd.DataFrame(values, index=index, columns=pd.MultiIndex.from_tuples(columns))
    frame.columns.names = ["asset", "indicator"]
    return frame


def test_a_new_hash_is_archived_without_replacing_the_previous(tmp_path):
    source = tmp_path / "asset_indicator_matrix.csv"
    archive = tmp_path / "archive"
    frame = _frame(4, [("QQQ", "a"), ("QQQ", "b")])
    frame.to_csv(source)
    loaded = load_feature_matrix(source)
    before = source.read_bytes()
    first = archive_feature_matrix(source, loaded, data_dir=archive)
    second = archive_feature_matrix(source, loaded, data_dir=archive)
    frame.iloc[0, 0] = 999.0
    frame.to_csv(source)
    revised = load_feature_matrix(source)
    third = archive_feature_matrix(source, revised, data_dir=archive)
    assert first["created"] is True
    assert first["file"] == "feature_matrix_v1.csv"
    assert second["created"] is False
    assert third["created"] is True
    assert third["file"] == "feature_matrix_v2.csv"
    assert sha256_file(archive / "feature_matrix_v1.csv") != sha256_file(archive / "feature_matrix_v2.csv")
    assert source.read_bytes() != before
    manifest = (archive / "manifest.jsonl").read_text().strip().splitlines()
    assert len(manifest) == 2
    text = (archive / "MANIFEST.txt").read_text()
    assert "feature_matrix_v1.csv" in text
    assert "feature_matrix_v2.csv" in text
    assert third["sha256"] in text


def test_schema_mismatch_names_the_missing_or_new_columns():
    columns = [("QQQ", "a"), ("QQQ", "b")]
    frame = _frame(3, columns)
    try:
        assert_schema_matches_model(frame, ["QQQ|a", "QQQ|c"])
    except FeatureSchemaError as exc:
        assert "QQQ|c" in str(exc)
        assert "Stopping without changing the saved model" in str(exc)
    else:
        raise AssertionError("missing column should stop the run")
    try:
        assert_schema_matches_model(frame, ["QQQ|a"], ["QQQ|a"])
    except FeatureSchemaError as exc:
        assert "QQQ|b" in str(exc)
    else:
        raise AssertionError("new column should stop the run")


def test_training_guard_rejects_a_changed_window_without_a_new_fit():
    columns = [("QQQ", "a")]
    frame = _frame(4, columns)
    scaler = StandardScaler().fit(frame.to_numpy(dtype=float))
    assert_training_matches_scaler(frame, scaler, frame.columns)
    longer = _frame(5, columns)
    try:
        assert_training_matches_scaler(longer, scaler, longer.columns)
    except FrozenTrainingMismatch as exc:
        assert "without refitting" in str(exc)
    else:
        raise AssertionError("a longer training window should stop the run")
    shifted = frame.copy()
    shifted.iloc[:, 0] = shifted.iloc[:, 0] + 10
    try:
        assert_training_matches_scaler(shifted, scaler, shifted.columns)
    except FrozenTrainingMismatch as exc:
        assert "column means" in str(exc)
    else:
        raise AssertionError("changed training values should stop the run")
