"""Price-relative transforms. Scale-free columns are copied through."""

from __future__ import annotations

import pandas as pd

from .consistency import INDICATOR_SPECS, LEVEL_FORMULA, SCALE_FREE_FORMULA, WIDTH_FORMULA

FORMULA_BY_FAMILY = {
    "level": LEVEL_FORMULA,
    "width": WIDTH_FORMULA,
    "scale_free": SCALE_FREE_FORMULA,
}


def transformation_table() -> list[dict]:
    return [
        {"indicator": name, "family": family, "formula": FORMULA_BY_FAMILY[family]}
        for name, _function, _output, family in INDICATOR_SPECS
    ]


def apply_price_relative(features: pd.DataFrame, closes: pd.DataFrame) -> pd.DataFrame:
    """Divide price-unit columns by the same asset's same-day adjusted close.

    Level columns become L/C - 1. Width columns become V/C. Other columns stay.
    Columns that are not in `features` are skipped. The stage-2 runner checks
    that the full indicator list is present before calling this.
    """
    transformed = features.copy()
    for name, _function, _output, family in INDICATOR_SPECS:
        if family == "scale_free":
            continue
        for asset in closes.columns:
            column = (asset, name)
            if column not in features.columns:
                continue
            if family == "level":
                transformed[column] = features[column] / closes[asset] - 1.0
            elif family == "width":
                transformed[column] = features[column] / closes[asset]
            else:
                raise ValueError(f"unknown family {family} for {name}")
    values = transformed.to_numpy(dtype=float)
    if not pd.notna(values).all():
        raise ValueError("price-relative transform produced missing values")
    return transformed


def missing_transform_columns(features: pd.DataFrame) -> list[str]:
    assets = list(dict.fromkeys(features.columns.get_level_values(0)))
    missing = []
    for name, _function, _output, _family in INDICATOR_SPECS:
        for asset in assets:
            if (asset, name) not in features.columns:
                missing.append(f"{asset}|{name}")
    return missing
