"""Calendar alignment and the 2018 training cutoff."""

import pandas as pd

from ai_trades.constants import TRAIN_END
from ai_trades.data_prep import calendar_gaps, daily_index, split_by_cutoff


def test_daily_index_keeps_exchange_calendar_date():
    index = pd.date_range("2020-01-02", periods=2, tz="America/Chicago")
    aligned = daily_index(index)
    assert aligned.tz is None
    assert list(aligned) == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")]


def test_split_is_chronological_and_disjoint():
    index = pd.bdate_range("2018-12-27", periods=6)
    train, test = split_by_cutoff(index, TRAIN_END)
    assert train.max() <= TRAIN_END
    assert test.min() > TRAIN_END
    assert set(train).isdisjoint(set(test))
    assert list(train) == sorted(train)
    assert list(test) == sorted(test)


def test_split_rejects_a_cutoff_that_removes_a_side():
    index = pd.bdate_range("2019-01-02", periods=5)
    try:
        split_by_cutoff(index, TRAIN_END)
    except ValueError as exc:
        assert "both" in str(exc)
    else:
        raise AssertionError("cutoff past the sample should fail")


def test_long_gaps_ignore_a_four_day_weekend():
    index = pd.to_datetime(["2020-01-02", "2020-01-06", "2020-01-20"])
    gaps = calendar_gaps(index, min_days=5)
    assert gaps == [{"date": "2020-01-20", "gap_days": 14}]
