import pytest
from src.data.processor import TaLibProcessor
import pandas as pd

from tests.conftest import ta_processor



class TestStockFetcher:
    
    @pytest.mark.fast
    def test_fetch_assetDataFrame(self):
        pass

    @pytest.mark.fast
    def test_fetch_ohlcv(self):
        pass

    @pytest.mark.fast
    def test_processor(self):
        pass

    @pytest.mark.pipeline
    def test_workflow(self):
        pass
    
class TestTAlibProcessor:

    @pytest.mark.pipeline
    def test_workflow(self, ta_processor, ta_map, sample_market_data):

        processor = ta_processor
        df = sample_market_data

        indicator_dict = {key: processor.compute(key, df) for key, val in ta_map.items() if val is not None}
        
        for key, val in indicator_dict.items():
            print(f'====================================================')
            print(f'                {key}      ')
            print(f'        type:{type(val)}        ')
            print(f'====================================================')
            print(val)
            print(f'====================================================')

        print(len(indicator_dict))
