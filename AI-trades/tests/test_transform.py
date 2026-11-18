"""Price-relative formulas leave scale-free columns unchanged."""

import numpy as np
import pandas as pd

from ai_trades.stage2 import STAGE1_VERSION, STAGE2_VERSION
from ai_trades.transform import apply_price_relative


def test_stage2_model_name_is_separate_from_stage1():
    assert STAGE2_VERSION != STAGE1_VERSION
    assert "stage1" not in STAGE2_VERSION


def test_level_width_and_scale_free_formulas():
    index = pd.bdate_range("2020-01-02", periods=3)
    columns = pd.MultiIndex.from_tuples(
        [
            ("QQQ", "Moving Average"),
            ("QQQ", "Average True Range"),
            ("QQQ", "Relative Strength Index (RSI)"),
        ]
    )
    features = pd.DataFrame([[110.0, 2.0, 55.0], [90.0, 1.0, 40.0], [105.0, 3.0, 60.0]], index=index, columns=columns)
    closes = pd.DataFrame({"QQQ": [100.0, 100.0, 100.0]}, index=index)
    transformed = apply_price_relative(features, closes)
    assert np.allclose(transformed[("QQQ", "Moving Average")], [0.10, -0.10, 0.05])
    assert np.allclose(transformed[("QQQ", "Average True Range")], [0.02, 0.01, 0.03])
    assert np.allclose(transformed[("QQQ", "Relative Strength Index (RSI)")], [55.0, 40.0, 60.0])
