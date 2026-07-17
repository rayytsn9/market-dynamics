import ccxt
import pandas as pd
import numpy as numpy
import time
import datetime as dt
import config
import yfinance as yf
from .processor import CryptoProcessor, StockProcessor

# ===================================================================================
# CRYPTO FETCHER
# ===================================================================================

class CryptoFetcher:


    @staticmethod
    def instantiate_exchange(exchange_id = config.CryptoConfig.EXCHANGE, enable_rate_limit = True):
        try:
            exchange = getattr(ccxt, exchange_id)({'enableRateLimit': enable_rate_limit})
            exchange.load_markets()
            return exchange
        except:
            print('exchange failed to instantiate')
            return None

    @staticmethod
    def get_top_m_symbols(exchange, m: int, exclude_coins=config.CryptoConfig.STABLECOINS, window=30) -> list:
        
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
    
    @staticmethod
    def fetch_ohlcv(exchange, symbol: str, timeframe='1d', since=None, limit=None, period=None) -> pd.DataFrame:
        
        if not config.CRYPTO and not config.ASSETS:
            raise ValueError(f'CRYPTO: {config.CRYPTO}, ASSETS: {config.ASSETS}')
        
        df = None

        # fetching via ccxt
        if config.CRYPTO:

            if exchange.has['fetchOHLCV']:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
                df = pd.DataFrame(ohlcv, columns= ['timestamp' , 'open', 'high', 'low', 'close', 'volume'])
                time.sleep(exchange.rateLimit / 1000)
            else:
                raise ValueError('exchange does not have "fetchOHLCV"')
                
        df['log_return'] = CryptoProcessor.get_log_return_df(df['close'])
        df = df.dropna(subset=['log_return', 'timestamp'])
        df.set_index('timestamp', inplace=True)

        return df

    @staticmethod
    def construct_log_return_dataset(exchange, symbols: list, since=None, limit=None) -> dict:

        dataset = {}

        # if config.CRYPTO:
        for s in symbols:
            try:
                df = CryptoFetcher.fetch_ohlcv(exchange, s, since=since, limit=limit)
                if df is not None:
                    dataset[s] = df['log_return']
            except:
                continue

        return dataset

# ===================================================================================
# STOCK FETCHER
# ===================================================================================

class StockFetcher:

    @staticmethod
    def fetch_assetDataFrame(tickers: list, period=None, interval='1d', start=None, end=None) -> pd.DataFrame:

        try:
            df = yf.download(tickers, period=period, interval=interval, start=start, end=end)
        except:
            raise ValueError('failed to fetch tickers')

        return df

    @staticmethod
    def has_ohlcv(ticker_symbol: str) -> bool:
        
        try:
            df = yf.Ticker(ticker_symbol).history()
            return not df.empty
        except:
            return False

    @staticmethod
    def fetch_ohlcv(symbol: str, timeframe='1d', since=None, limit=None, period=None) -> pd.DataFrame:
        
        if not config.CRYPTO and not config.STOCKS:
            raise ValueError(f'CRYPTO: {config.CRYPTO}, ASSETS: {config.ASSETS}')
        
        df = None

        if config.STOCKS:

            if StockFetcher.has_ohlcv(symbol):
                df = yf.Ticker(symbol).history(period=period, interval=timeframe)
                df = df.reset_index()
                df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits', 'capital_gains']
            else:
                raise ValueError(f'failed to fetch {symbol}')
        else:
            raise ValueError(f'incorrect mode .. config.ASSETS = {config.ASSETS}')

        df['log_return'] = StockProcessor.compute_log_return(df['close'])
        df = df.dropna(subset=['log_return', 'timestamp'])
        df.set_index('timestamp', inplace=True)

        return df
    
    @staticmethod
    def fetch_TA(indicators: list) -> pd.DataFrame:
        pass