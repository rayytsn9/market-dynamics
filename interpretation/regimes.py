import numpy as np
import pandas as pd
from scipy import stats

def build_scores(projected: np.ndarray, index, n_factors: int) -> pd.DataFrame:
    """
    Builds a DataFrame of PC scores over time
    projected: (N, n_factors)
    Returns DataFrame with datetime index
    """
    columns = [f'PC{i+1}' for i in range(n_factors)]
    return pd.DataFrame(projected, index=index, columns=columns)


def classify_regimes(scores: pd.DataFrame, n_sigma: float = 1.0) -> pd.DataFrame:
    """
    Classifies each day into a regime based on PC1 score
    Returns DataFrame with regime labels and confidence
    """
    mu    = scores['PC1'].mean()
    sigma = scores['PC1'].std()
    
    regimes = pd.DataFrame(index=scores.index)
    
    # regime label
    regimes['regime'] = 'neutral'
    regimes.loc[scores['PC1'] >  n_sigma * sigma, 'regime'] = 'bull'
    regimes.loc[scores['PC1'] < -n_sigma * sigma, 'regime'] = 'bear'
    
    # confidence
    regimes['confidence'] = scores['PC1'].apply(
        lambda x: stats.norm.cdf((x - mu) / sigma)
        if x > mu
        else 1 - stats.norm.cdf((x - mu) / sigma)
    )
    
    return regimes


def get_today_regime(scores: pd.DataFrame, regimes: pd.DataFrame) -> dict:
    """
    Returns today's regime summary as a dict for dashboard display
    """
    today_scores  = scores.iloc[-1]
    today_regime  = regimes.iloc[-1]
    
    result = {
        'regime':     today_regime['regime'],
        'confidence': today_regime['confidence'],
    }
    
    # dynamically add PC scores
    for col in scores.columns:
        result[col] = float(today_scores[col])
    
    return result