"""Recompute the 38 indicators and check units before any transformation.

A column is a price level when it tracks the close. A column is a price width
when it tracks a range or a difference of prices. Those two cases use different
ratios. Scale-free columns are left unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data_prep import daily_index
from .prices import ASSETS

# function, optional output column, and the formula family proposed for it.
INDICATOR_SPECS = [
    ("ADX", "ADX", None, "scale_free"),
    ("Aroon Oscillator", "AROONOSC", None, "scale_free"),
    ("Average True Range", "ATR", None, "width"),
    ("Bollinger Bands_upperband", "BBANDS", "upperband", "level"),
    ("Bollinger Bands_middleband", "BBANDS", "middleband", "level"),
    ("Bollinger Bands_lowerband", "BBANDS", "lowerband", "level"),
    ("Chande Momentum Oscillator", "CMO", None, "scale_free"),
    ("Commodity Channel Index", "CCI", None, "scale_free"),
    ("Directional Movement", "DX", None, "scale_free"),
    ("Exponential Moving Average", "EMA", None, "level"),
    ("Kaufman Adaptive Moving Average", "KAMA", None, "level"),
    ("Linear Regression", "LINEARREG", None, "level"),
    ("Linear Regression Indicator", "TSF", None, "level"),
    ("MACD Histogram", "MACD", "macdhist", "width"),
    ("MACD Indicator", "MACD", "macd", "width"),
    ("MACD Percentage", "PPO", None, "scale_free"),
    ("Median Price", "MEDPRICE", None, "level"),
    ("Momentum Indicator", "MOM", None, "width"),
    ("Moving Average", "MA", None, "level"),
    ("Parabolic SAR", "SAR", None, "level"),
    ("Price Differential", "SUB", None, "width"),
    ("Price Ratio", "DIV", None, "scale_free"),
    ("Rate of Change (Price)", "ROC", None, "scale_free"),
    ("Relative Strength Index (RSI)", "RSI", None, "scale_free"),
    ("Standard Deviation Channels", "STDDEV", None, "width"),
    ("Stochastic Oscillator_slowk", "STOCH", "slowk", "scale_free"),
    ("Stochastic Oscillator_slowd", "STOCH", "slowd", "scale_free"),
    ("Stochastic RSI_fastk", "STOCHRSI", "fastk", "scale_free"),
    ("Stochastic RSI_fastd", "STOCHRSI", "fastd", "scale_free"),
    ("The Moving Average Oscillator", "APO", None, "width"),
    ("Triple Exponential Moving Average", "TEMA", None, "level"),
    ("TRIX Indicator", "TRIX", None, "scale_free"),
    ("True Range", "TRANGE", None, "width"),
    ("Typical Price", "TYPPRICE", None, "level"),
    ("Ultimate Oscillator", "ULTOSC", None, "scale_free"),
    ("Weighted Close", "WCLPRICE", None, "level"),
    ("Weighted Moving Average", "WMA", None, "level"),
    ("Williams %R", "WILLR", None, "scale_free"),
]

LEVEL_FORMULA = "L / C - 1"
WIDTH_FORMULA = "V / C"
SCALE_FREE_FORMULA = "unchanged"

# Recomputed series must track the stored matrix. These cutoffs separate a
# convention match from a revision large enough to stop the experiment.
MIN_CORRELATION = 0.999
MAX_MEDIAN_RELATIVE_ERROR = 1e-3


class ConsistencyError(RuntimeError):
    """Adjusted prices or indicator units are not safe to transform."""


def compute_spec(frame: pd.DataFrame, function: str, output: str | None) -> pd.Series:
    import talib.abstract as ta

    function_handle = ta.Function(function)
    working = frame.copy()
    working.index = pd.RangeIndex(len(working))
    function_handle.set_input_arrays(working)
    result = function_handle()
    if output is None:
        values = result if isinstance(result, pd.Series) else result.iloc[:, 0]
    else:
        values = result[output]
    values = pd.Series(np.asarray(values, dtype=float), index=frame.index)
    values.index = daily_index(values.index)
    return values


def _usable_ohlc(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in ("open", "high", "low", "close", "volume") if column in frame.columns]
    return frame.loc[:, columns].dropna(subset=["open", "high", "low", "close"])


def empirical_family(features: pd.DataFrame, adjusted: dict[str, pd.DataFrame], name: str) -> str:
    """Separate price levels, price widths, and scale-free series on QQQ.

    A level tracks the close. A width tracks the dollar range or the dollar
    change more closely than the percent range or the percent change.
    """
    ohlc = adjusted["QQQ"]
    close = ohlc["close"].astype(float)
    high = ohlc["high"].astype(float)
    low = ohlc["low"].astype(float)
    for series in (close, high, low):
        series.index = daily_index(series.index)
    feature = features[("QQQ", name)].astype(float)
    compared = pd.concat(
        {
            "feature": feature,
            "close": close,
            "dollar_change": close.diff(10),
            "percent_change": close.pct_change(10, fill_method=None),
            "dollar_range": high - low,
            "percent_range": (high - low) / close,
        },
        axis=1,
    ).dropna()
    values = compared["feature"]
    close_correlation = float(values.corr(compared["close"]))
    level_ratio = float((values / compared["close"]).median())
    if close_correlation >= 0.98 and level_ratio > 0.5:
        return "level"
    dollar_change = float(values.corr(compared["dollar_change"]))
    percent_change = float(values.corr(compared["percent_change"]))
    dollar_range = float(values.corr(compared["dollar_range"]))
    percent_range = float(values.corr(compared["percent_range"]))
    fraction = float((values.abs() / compared["close"]).median())
    if dollar_range >= 0.80 and dollar_range > percent_range:
        return "width"
    if dollar_change >= 0.80 and dollar_change > percent_change:
        return "width"
    if dollar_change > percent_change + 0.15 and dollar_change > 0.4 and fraction < 0.05:
        return "width"
    return "scale_free"


def _relative_error(stored: np.ndarray, recomputed: np.ndarray) -> np.ndarray:
    scale = np.maximum(np.abs(stored), 1e-6)
    return np.abs(stored - recomputed) / scale


def verify_prices_and_units(features: pd.DataFrame, adjusted: dict[str, pd.DataFrame], raw: dict[str, pd.DataFrame]) -> dict:
    feature_index = daily_index(features.index)
    issues = []
    rows = []
    closes = {}
    for asset in ASSETS:
        if asset not in adjusted or asset not in raw:
            issues.append(f"{asset} is missing from a downloaded price panel")
            continue
        close = adjusted[asset]["close"].astype(float)
        close.index = daily_index(close.index)
        close = close.dropna()
        closes[asset] = close
        missing_dates = sorted(set(feature_index) - set(close.index))
        invalid = close.reindex(feature_index)
        n_invalid = int((~np.isfinite(invalid.to_numpy()) | (invalid.to_numpy() <= 0)).sum())
        if missing_dates:
            preview = ", ".join(stamp.date().isoformat() for stamp in missing_dates[:5])
            issues.append(f"{asset} adjusted close is missing {len(missing_dates)} feature dates, first {preview}")
        if n_invalid:
            issues.append(f"{asset} has {n_invalid} feature dates with a non-positive or non-finite adjusted close")
        raw_close = raw[asset]["close"].astype(float)
        raw_close.index = daily_index(raw_close.index)
        both = pd.concat({"adjusted": close, "raw": raw_close}, axis=1).dropna()
        median_gap = float((both["adjusted"] - both["raw"]).abs().median()) if len(both) else None
        rows.append(
            {
                "asset": asset,
                "adjusted_start": close.index.min().date().isoformat(),
                "adjusted_end": close.index.max().date().isoformat(),
                "median_abs_adjusted_minus_raw_close": median_gap,
            }
        )

    comparisons = []
    for name, function, output, family in INDICATOR_SPECS:
        adjusted_correlations = []
        raw_correlations = []
        adjusted_errors = []
        for asset in ASSETS:
            if (asset, name) not in features.columns or asset not in adjusted:
                issues.append(f"stored matrix is missing {asset}|{name}")
                continue
            stored = features[(asset, name)].astype(float)
            stored.index = feature_index
            recomputed_adjusted = compute_spec(_usable_ohlc(adjusted[asset]), function, output).reindex(feature_index)
            recomputed_raw = compute_spec(_usable_ohlc(raw[asset]), function, output).reindex(feature_index)
            pair = pd.concat(
                {"stored": stored, "adjusted": recomputed_adjusted, "raw": recomputed_raw},
                axis=1,
            ).dropna()
            if len(pair) < 100:
                issues.append(f"{asset}|{name} has only {len(pair)} overlapping recomputed rows")
                continue
            adjusted_correlation = float(pair["stored"].corr(pair["adjusted"]))
            raw_correlation = float(pair["stored"].corr(pair["raw"]))
            median_relative = float(np.median(_relative_error(pair["stored"].to_numpy(), pair["adjusted"].to_numpy())))
            adjusted_correlations.append(adjusted_correlation)
            raw_correlations.append(raw_correlation)
            adjusted_errors.append(median_relative)
            if adjusted_correlation < MIN_CORRELATION or median_relative > MAX_MEDIAN_RELATIVE_ERROR:
                issues.append(
                    f"{asset}|{name} does not match the main.py price convention: "
                    f"correlation {adjusted_correlation:.6f}, median relative error {median_relative:.3e}"
                )
        unit = empirical_family(features, adjusted, name) if ("QQQ", name) in features.columns else "missing"
        if unit != family:
            issues.append(f"{name} was treated as {family} but the QQQ series behaves as {unit}")
        formula = {"level": LEVEL_FORMULA, "width": WIDTH_FORMULA, "scale_free": SCALE_FREE_FORMULA}[family]
        comparisons.append(
            {
                "indicator": name,
                "family": family,
                "empirical_unit_qqq": unit,
                "formula": formula,
                "min_adjusted_correlation": float(min(adjusted_correlations)) if adjusted_correlations else None,
                "min_raw_correlation": float(min(raw_correlations)) if raw_correlations else None,
                "max_median_relative_error": float(max(adjusted_errors)) if adjusted_errors else None,
            }
        )

    families = [row["family"] for row in comparisons]
    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "assets": rows,
        "indicators": comparisons,
        "n_level_columns": 6 * families.count("level"),
        "n_width_columns": 6 * families.count("width"),
        "n_scale_free_columns": 6 * families.count("scale_free"),
        "closes": closes,
    }
