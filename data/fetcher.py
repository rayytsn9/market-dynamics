import ccxt
import pandas as pd
import numpy as numpy
import time
import datetime as dt
import config
from .processor import get_log_return_df

def instantiate_exchange(exchange_id = config.EXCHANGE, enable_rate_limit = True):
    try:
        exchange = getattr(ccxt, exchange_id)({'enableRateLimit': enable_rate_limit})
        exchange.load_markets()
        return exchange
    except:
        print('exchange failed to instantiate')
        return None


def get_top_m_symbols(exchange, m: int, exclude_coins=config.STABLECOINS, window=30) -> list:
    
    if exchange.has['fetchTickers']:
        ticker = exchange.fetch_tickers()
    else:
        print("exchange doesnt have 'fetchTickers' method")

    # sorting by quoteVolume and quote = USDT
    usdt_pairs = {k: v for k, v in ticker.items() if k.endswith('/USDT')}

    avg_vol = {}

    for symbol in usdt_pairs:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1d', limit=window)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open','high', 'low', 'close', 'volume'])
            df['quoteVolume'] = df['volume'] * df['close']
            avg_vol[symbol] = df['quoteVolume'].mean()
            time.sleep(exchange.rateLimit / 1000)
        except:
            continue

    sorted_pairs = sorted(avg_vol.items(), key=lambda x: x[1] or 0, reverse=True)
    top_m_symbols = [s[0] for s in sorted_pairs if s[0] not in exclude_coins][:m]

    return top_m_symbols


def fetch_ohlcv(exchange, symbols: list, timeframe='1d', since=None, limit=None) -> dict:
    
    ohlcv_dict = {}

    if exchange.has['fetchOHLCV']:
        for symbol in symbols:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
            ohlcv_dict[symbol] = ohlcv
            time.sleep(exchange.rateLimit / 1000)
    else:
        print('exchange does not have "fetchOHLCV"')

    return ohlcv_dict
    
def construct_dataset(exchange, symbols: list, N: int, since=None):
    
    dataset = {}
    
    for s in symbols:
        try:
            ohlcv = exchange.fetch_ohlcv(s, timeframe='1d', since=since, limit=N+200)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open','high', 'low', 'close', 'volume'])
            df['log_return'] = get_log_return_df(df['close'])
            df = df[['timestamp', 'log_return']].dropna()
            df.set_index('timestamp', inplace=True)
            dataset[s] = df['log_return']
            time.sleep(exchange.rateLimit / 1000)
        except:
            continue

    return dataset