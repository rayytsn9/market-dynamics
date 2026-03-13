"""
Unit tests for crypto factor model. Uncomment and implement each test.
Run: pytest test.py -v   (from project root: crypto_factor_model/)
"""
import numpy as np
import pandas as pd

# Module under test
import data.processor as processor
import data.fetcher as fetcher
import analysis.covariance as covariance
import analysis.pca as pca
import analysis.mp as mp
from utils import get_since
import config

import pytest
# --- data.processor ---

# def test_get_log_return_df():
#     """processor.get_log_return_df(series) -> pd.Series"""
#     pass

# def test_build_panel():
#     """processor.build_panel(dataset, N, M) -> pd.DataFrame"""
#     pass

# def test_mean_center():
#     """processor.mean_center(df) -> np.ndarray"""
#     pass


# --- analysis.covariance ---

# def test_sample_covariance():
#     """covariance.sample_covariance(X) -> np.ndarray"""
#     pass

# def test_validate_covariance():
#     """covariance.validate_covariance(C) -> None"""
#     pass


# --- analysis.pca ---

# def test_pca_statistics():
#     """pca.pca_statistics(X, n_components) -> dict"""
#     pass

# def test_svd_statistics():
#     """pca.svd_statistics(X, n_components) -> dict"""
#     pass

# def test_get_eigenpair():
#     """pca.get_eigenpair(C) -> tuple"""
#     pass

# def test_extract_factors():
#     """pca.extract_factors(eigvecs, n_factors) -> np.ndarray"""
#     pass

# def test_project():
#     """pca.project(X, PC) -> np.ndarray"""
#     pass


# --- analysis.mp (Marchenko–Pastur) ---

# def test_mp_support():
#     """mp.mp_support(gamma, sigma2=1.0) -> (a, b)"""
#     pass

# def test_mp_density():
#     """mp.mp_density(X, gamma, sigma2=1.0, a=None, b=None) -> np.ndarray"""
#     pass

# def test_estimate_sigma2():
#     """mp.estimate_sigma2(eigvals) -> float"""
#     pass

# def test_get_outliers():
#     """mp.get_outliers(eigvals, b) -> np.ndarray"""
#     pass


# --- data.fetcher (may require mocking / fixtures for exchange) ---
test_data_fetcher = True


@pytest.mark.skipif(test_data_fetcher == False, reason=f'test data fetcher: {test_data_fetcher}')
def test_instantiate_exchange():
    """fetcher.instantiate_exchange(exchange_id, enable_rate_limit)"""
    
    exchange = fetcher.instantiate_exchange('binanceus', True)
    assert exchange != None
    
@pytest.mark.skipif(test_data_fetcher == False, reason=f'test data fetcher: {test_data_fetcher}')
def test_get_top_m_symbols():
    """fetcher.get_top_m_symbols(exchange, m, exclude_coins, window) -> list"""
    
    exchange = fetcher.instantiate_exchange()
    symbols = fetcher.get_top_m_symbols(exchange, config.M, window=config.N)
    assert len(symbols) > 0

@pytest.mark.skipif(test_data_fetcher == False, reason=f'test data fetcher: {test_data_fetcher}')
def test_fetch_ohlcv():
    """fetcher.fetch_ohlcv(exchange, symbols, timeframe, since, limit) -> dict"""

    since = get_since(config.N+200)

    try:
        exchange = fetcher.instantiate_exchange()
    except Exception as e:
        pytest.fail(f"Test failed with exception: {e}")
        
    symbols = fetcher.get_top_m_symbols(exchange, config.M, window=config.N)

    since = get_since(config.N+200)

    ohlcv = fetcher.fetch_ohlcv(exchange, symbols[0], since=since, limit=config.N+200)

    print(ohlcv)
    print(ohlcv.shape)

    assert type(ohlcv) == pd.DataFrame
    assert ohlcv.shape[0] == config.N+199
    assert ohlcv.shape[1] == 6

@pytest.mark.skipif(test_data_fetcher == False, reason=f'test data fetcher: {test_data_fetcher}')
def test_construct_log_return_dataset():
    """fetcher.construct_dataset(exchange, symbols, N, since) -> dict"""
    
    since = get_since(config.N)

    try:
        exchange = fetcher.instantiate_exchange()
    except Exception as e:
        pytest.fail(f"Test failed with exception: {e}")

    symbols = fetcher.get_top_m_symbols(exchange, 5, window=config.N)

    since = get_since(config.N + 200)

    dataset = fetcher.construct_log_return_dataset(exchange, symbols, since, config.N+200)

    assert len(dataset) <= len(symbols)
    assert len(dataset) != 0
    

# --- utils (may require mocking / fixtures for exchange) ---
test_utils = False

@pytest.mark.skipif(test_utils == False, reason=f'test data fetcher : {test_utils}')
def test_get_since():

    import datetime as dt

    # Test default usage, 1d timeframe
    now = dt.datetime.utcnow().replace(second=0, microsecond=0, minute=0)
    N = 5
    ms_since = get_since(N, timeframe='1d')
    expected = int((now - dt.timedelta(days=N + 1)).replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
    # Allow difference up to 120s for test timing
    assert abs(ms_since - expected) < 2 * 60 * 1000

    # Test from_time parameter, naive datetime
    custom_ref = dt.datetime(2023, 1, 10, 12, 0, 0)
    ms_since2 = get_since(2, timeframe='1d', from_time=custom_ref)
    expected2 = int((custom_ref.replace(minute=0, second=0, microsecond=0) - dt.timedelta(days=3)).replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
    assert ms_since2 == expected2

    # Test hourly timeframe
    ms_since3 = get_since(10, timeframe='2h', from_time=custom_ref)
    expected3 = int((custom_ref.replace(minute=0, second=0, microsecond=0) - dt.timedelta(hours=2 * 11)).replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
    assert ms_since3 == expected3

    # Test unsupported timeframe
    with pytest.raises(ValueError):
        get_since(5, timeframe='7x')

