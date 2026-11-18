"""Frozen price-relative standardization.

This module is the stable import path for the saved preprocessor. Joblib records
``ai_trades.preprocessing.FrozenPriceRelativePreprocessor``. Do not execute this
file as ``__main__`` when saving an artifact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .transform import apply_price_relative


class FrozenPriceRelativePreprocessor:
    """Price-relative map plus the training-only scaler. transform() does not fit."""

    def __init__(self, scaler, columns: list[tuple[str, str]]):
        self.scaler = scaler
        self.columns = [(str(asset), str(indicator)) for asset, indicator in columns]

    def fit(self, *args, **kwargs):
        raise RuntimeError("The Stage 1 preprocessor is frozen and cannot be fit.")

    def transform(self, features: pd.DataFrame, closes: pd.DataFrame) -> pd.DataFrame:
        relative = apply_price_relative(features, closes)
        ordered = relative.loc[:, self.columns]
        values = self.scaler.transform(np.asarray(ordered, dtype=float))
        columns = pd.MultiIndex.from_tuples(self.columns, names=["asset", "indicator"])
        frame = pd.DataFrame(values, index=ordered.index, columns=columns)
        frame.index.name = "date"
        return frame
