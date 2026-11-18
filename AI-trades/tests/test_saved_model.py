import json

import joblib
import pandas as pd
import pytest

from ai_trades.constants import FEATURE_SOURCE, MODEL_DIR, MODEL_VERSION
from ai_trades.data_prep import load_feature_matrix, split_by_cutoff


def test_saved_model_uses_training_rows_and_selects_nothing():
    directory = MODEL_DIR / MODEL_VERSION
    metadata_path = directory / "metadata.json"
    if not metadata_path.exists():
        pytest.skip("stage 1 model has not been fit")
    metadata = json.loads(metadata_path.read_text())
    features = load_feature_matrix(FEATURE_SOURCE)
    train, test = split_by_cutoff(features.index)
    scaler = joblib.load(directory / "scaler.joblib")
    assert metadata["retention_rule_selected"] is None
    assert metadata["regime_count_selected"] is None
    assert metadata["train_end"] == "2018-12-31"
    assert scaler.n_samples_seen_ == len(train)
    assert metadata["scaler_n_samples_seen"] == len(train)
    assert train.max() <= pd.Timestamp("2018-12-31")
    assert test.min() > train.max()
