import numpy as np
import pandas as pd

def get_log_return_df(series: pd.Series) -> pd.Series:
    return np.log(series).diff()

def build_panel(dataset: dict, N, M) -> pd.DataFrame:
    """
    Builds a panel of log returns from a dictionary of asset data
    dataset: dict of asset data (keys: symbols, values: log return series)
    N: number of trading days to include
    M: number of assets to include
    Returns: DataFrame of log returns with symbols as columns and trading days as index
    """
    panel = pd.concat(dataset, axis=1)
    panel = panel.sort_index()
    panel = panel.dropna(axis=1) 

    if panel.shape[1] == 0:
        raise ValueError("No assets with complete data")

    panel = panel.tail(N)

    if panel.shape[1] < M:
        raise ValueError(f"Only {panel.shape[1]} assets with complete data, need {M}")

    panel = panel.iloc[:, :M]

    return panel


def mean_center(df: pd.DataFrame) -> np.ndarray:
    X = df.values
    return X - X.mean(axis=0)

def clean_dataset(dataset: dict, N: int) -> dict:
    return {k: v for k, v in dataset.items() if len(v) >= N}



