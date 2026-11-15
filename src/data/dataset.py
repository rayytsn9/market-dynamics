import pandas as pd

import config
from src.data.fetcher import StockFetcher
from src.data.processor import TaLibProcessor, create_matrix, indicators_common_to


class NoCommonIndicators(Exception):
    """indicators_common_to returned no indicator usable on every asset."""


def drop_duplicate_columns(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that exactly match an earlier column of the same asset.

    Dependencies:
    - matrix: DataFrame with a two-level column MultiIndex (asset, indicator),
      as returned by create_matrix.
    - pandas Series.equals, which treats the full column, including index
      alignment, as the comparison.

    Columns are compared only within one asset. The first column in each
    identical group is kept. Moving Average precedes Simple Moving Average,
    so the simple moving average column is removed for that asset when the
    two series are identical.
    """
    if not isinstance(matrix.columns, pd.MultiIndex) or matrix.columns.nlevels != 2:
        raise ValueError('matrix columns must be a two-level MultiIndex of (asset, indicator)')

    keep = []
    for asset in matrix.columns.get_level_values(0).unique():
        kept_for_asset = []
        for col in matrix.columns:
            if col[0] != asset:
                continue
            series = matrix.loc[:, col]
            if any(series.equals(matrix.loc[:, kept]) for kept in kept_for_asset):
                continue
            kept_for_asset.append(col)
        keep.extend(kept_for_asset)
    return matrix.loc[:, keep]


def build_asset_matrix(processor: TaLibProcessor, n: int | None = None) -> pd.DataFrame:
    """
    Panel of indicators common to every ticker in StockConfig.ASSETS.

    Downloads daily history with period='max', keeps the indicators that
    indicators_common_to accepts for every asset, builds the aligned
    matrix from create_matrix, and drops columns that duplicate an earlier
    column of the same asset.

    Raises NoCommonIndicators when that list is empty, before create_matrix
    would raise ValueError('indicators is empty').
    """
    frames = {
        ticker: StockFetcher.fetch_ohlcv(ticker, period='max')
        for ticker in config.StockConfig.ASSETS
    }
    indicators = indicators_common_to(frames, processor)
    if not indicators:
        raise NoCommonIndicators(
            'no indicator is common to every asset: ' + ', '.join(frames)
        )
    matrix = create_matrix(frames, indicators, processor, n=n)
    return drop_duplicate_columns(matrix)
