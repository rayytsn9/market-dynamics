import config
import src.data.fetcher as fetcher
import src.data.processor as processor
import src.analysis.covariance as covariance
import src.analysis.pca as pca
import pandas as pd
from src.data.processor import TaLibProcessor


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

        def test_working_TI():

            df = fetcher.StockFetcher.fetch_ohlcv(config.StockConfig.ASSETS[0], period='6mo')
            talib_map = {key: val for key, val in config.StockConfig.TALIB_MAP.items() if val is not None}

            
            TaLibProcessor.instantiate_factory(talib_map)

            processor = TaLibProcessor()
            
            indicator_dict = {key: processor.compute(key, df) for key, val in talib_map.items()}

            seen_TI = set()
            unseen_TI = set()

            for key in indicator_dict.keys():
                seen_TI.add(key)

            for key in config.StockConfig.TALIB_MAP.keys():
                if key not in seen_TI:
                    unseen_TI.add(key)

        




if __name__ == "__main__":
    main()