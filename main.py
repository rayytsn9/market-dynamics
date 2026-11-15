import config
import src.data.fetcher as fetcher
import src.data.processor as processor
import src.analysis.covariance as covariance
import src.analysis.pca as pca
from src.data.dataset import NoCommonIndicators, build_asset_matrix
from src.data.processor import TaLibProcessor, indicators_common_to
from src.data.fetcher import StockFetcher
import pandas as pd
import sys
from pathlib import Path

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
        TaLibProcessor.instantiate_factory(config.StockConfig.TALIB_MAP)
        try:
            matrix = build_asset_matrix(TaLibProcessor())
        except NoCommonIndicators as exc:
            print(exc)
            sys.exit(1)
        else:
            out = Path(__file__).resolve().parent / 'data' / 'asset_indicator_matrix.csv'
            out.parent.mkdir(parents=True, exist_ok=True)
            matrix.to_csv(out)
            print(matrix.shape)
            print(out)


        

if __name__ == "__main__":
    main()