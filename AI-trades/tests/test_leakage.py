"""Training fits must not move when later observations change."""

import numpy as np
import pandas as pd

from ai_trades.data_prep import split_by_cutoff
from ai_trades.pca_diagnostics import fit_train_representation


def _panel(index, location):
    rng = np.random.default_rng(0)
    rows = []
    for stamp in index:
        draw = rng.normal(loc=location, size=3)
        rows.append(draw)
    frame = pd.DataFrame(rows, index=index, columns=pd.MultiIndex.from_product([["QQQ"], ["a", "b", "c"]]))
    frame.columns.names = ["asset", "indicator"]
    return frame


def test_scaler_matches_training_rows_only():
    train_index = pd.bdate_range("2018-01-02", periods=40)
    test_index = pd.bdate_range("2019-01-02", periods=15)
    train = _panel(train_index, location=0.0)
    test = _panel(test_index, location=25.0)
    fitted = fit_train_representation(train)
    pooled = fit_train_representation(pd.concat([train, test]))
    assert fitted["scaler"].n_samples_seen_ == len(train)
    assert np.allclose(fitted["scaler"].mean_, train.to_numpy().mean(axis=0))
    assert not np.allclose(fitted["scaler"].mean_, pooled["scaler"].mean_)


def test_cutoff_helper_excludes_post_2018_dates_from_train():
    index = pd.bdate_range("2018-12-20", periods=12)
    train, test = split_by_cutoff(index)
    assert train.max() <= pd.Timestamp("2018-12-31")
    assert (test > pd.Timestamp("2018-12-31")).all()
    changed_test = test + pd.Timedelta(days=365)
    assert set(train).isdisjoint(changed_test)
