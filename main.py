import config
import src.data.fetcher as fetcher
import src.data.processor as processor
import src.analysis.covariance as covariance
import src.analysis.pca as pca
import pandas as pd
from src.data.processor import TaLibProcessor
import src.data.models as models


def main():

    if config.CRYPTO:
        exchange = fetcher.instantiate_exchange()

        symbols = fetcher.get_top_m_symbols(exchange, config.CryptoConfig.M)
        ohlcv = fetcher.fetch_ohlcv(exchange, symbols, timeframe='1d', limit=10)
        # panel = processor.build_panel(ohlcv, config.N, config.M)
        # covariance = covariance.sample_covariance(panel)
        # pca = pca.pca_statistics(panel, config.N)
        # print(pca)

        print(ohlcv.keys())

    elif config.STOCKS:
        
        def test_fetch_assetDataFrame():
            df = fetcher.StockFetcher.fetch_assetDataFrame(config.StockConfig.ASSETS[:2], period='6mo')
            return df
        
        def test_fetch_ohlcv():
            df = fetcher.StockFetcher.fetch_ohlcv('SPY', period='6mo')
            return df

        def test_config_indicators():
            df = pd.DataFrame(config.StockConfig.INDICATORS)
            return df

        def feature_pipeline():
            pass






if __name__ == "__main__":
    main()