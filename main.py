import config
import data.fetcher as fetcher
import data.processor as processor
import analysis.covariance as covariance
import analysis.pca as pca

def main():

    if config.CRYPTO:
        exchange = fetcher.instantiate_exchange()

        symbols = fetcher.get_top_m_symbols(exchange, config.M)
        ohlcv = fetcher.fetch_ohlcv(exchange, symbols, timeframe='1d', limit=10)
        # panel = processor.build_panel(ohlcv, config.N, config.M)
        # covariance = covariance.sample_covariance(panel)
        # pca = pca.pca_statistics(panel, config.N)
        # print(pca)

        print(ohlcv.keys())

    elif config.ASSETS:
        
        def test_fetch_assetDataFrame():
            df = fetcher.fetch_assetDataFrame(config.ASSETS[0], period='6mo')
            return df
        
        def test_fetch_ohlcv():
            df = fetcher.fetch_ohlcv('yfinance', config.ASSETS[0], period='6mo')
            return df
        
        df = test_fetch_assetDataFrame()
        print(df)        

    
if __name__ == "__main__":
    main()