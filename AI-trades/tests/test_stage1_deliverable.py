"""Checks for the packaged Stage 1 matrix. Does not refit the scaler."""

import json

import numpy as np
import pandas as pd

from ai_trades.constants import TRAIN_END
from ai_trades.data_prep import daily_index, split_by_cutoff
from ai_trades.stage1_deliverable import ARTIFACT_DIR


def test_stage1_artifact_is_frozen_training_standardization():
    matrix_path = ARTIFACT_DIR / "feature_matrix_standardized.csv"
    metadata_path = ARTIFACT_DIR / "feature_metadata.json"
    assert matrix_path.is_file()
    assert metadata_path.is_file()
    metadata = json.loads(metadata_path.read_text())
    assert metadata["stage_complete"] is False
    assert metadata["scaler_n_samples_seen"] == 4044
    assert "Full-sample standardization was not performed." in metadata["standardization"]

    frame = pd.read_csv(matrix_path, header=[0, 1], index_col=0, parse_dates=True)
    frame.index = daily_index(frame.index)
    assert frame.shape == (6025, 228)
    assert np.isfinite(frame.to_numpy(dtype=float)).all()
    names = [(row["asset"], row["indicator"]) for row in metadata["columns"]]
    assert list(frame.columns) == names

    train_index, test_index = split_by_cutoff(frame.index, TRAIN_END)
    assert len(train_index) == 4044
    assert len(test_index) == 1981
    train = frame.loc[train_index].to_numpy(dtype=float)
    assert np.max(np.abs(train.mean(axis=0))) < 1e-8
    assert np.max(np.abs(train.var(axis=0, ddof=0) - 1.0)) < 1e-8
    assert metadata["n_test_rows"] == 1981
