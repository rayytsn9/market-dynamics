import numpy as np
import pandas as pd

def build_loading_matrix(PC: np.ndarray, symbols: list, eigvals: np.ndarray) -> pd.DataFrame:
    """
    Builds a DataFrame of PC loadings per asset
    PC: (m, n_factors)
    Returns DataFrame shape (m, n_factors) with symbols as index
    """
    columns = [f'PC{i+1} ({eigvals[i]:.4f})' for i in range(PC.shape[1])]
    return pd.DataFrame(PC, index=symbols, columns=columns)

def get_factor_exposures(pc_loadings: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    """
    Categorizes each asset's exposure to each factor
    Returns DataFrame of 'positive', 'negative', 'neutral' labels
    shape (m, n_factors)
    """
    exposure_matrix = pd.DataFrame(index=pc_loadings.index)
    
    for col in pc_loadings.columns:
        exposure_matrix[col] = 'neutral'
        exposure_matrix.loc[pc_loadings[col] >  threshold, col] = 'positive'
        exposure_matrix.loc[pc_loadings[col] < -threshold, col] = 'negative'
    
    return exposure_matrix

def get_factor_groups(exposure_matrix: pd.DataFrame, factor: str) -> dict:
    """
    Returns dict of positive/negative/neutral coin lists for a given factor
    """
    return {
        'positive': exposure_matrix[exposure_matrix[factor] == 'positive'].index.tolist(),
        'negative': exposure_matrix[exposure_matrix[factor] == 'negative'].index.tolist(),
        'neutral':  exposure_matrix[exposure_matrix[factor] == 'neutral'].index.tolist()
    }