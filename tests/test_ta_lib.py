import pytest
from src.data.processor import TaLibProcessor
import config

class TestTa:

    @pytest.mark.ta
    def test_indicators(self, indicator_data, sample_market_data, ta_processor):
        
        indicator, func_name = indicator_data

        if func_name is None:
            TestTa._update_metrics(indicator,'XFAIL (Placeholder)')
            pytest.xfail()

        try:
            assert ta_processor.compute(indicator, sample_market_data) is not None, f'{indicator} function returned None!'
            TestTa._update_metrics(indicator,'PASSED')
        
        except Exception as e:
            TestTa._update_metrics(indicator, f'FAILED: {str(e)}')
            raise


    @classmethod
    def _update_metrics(cls, indicator: str, status: str):

        config.StockConfig.INDICATOR_STATUS['status_dict'][indicator] = status
        
        if status == 'PASSED':
            config.StockConfig.INDICATOR_STATUS['passed_count'] += 1
        elif 'FAIL' in status:
            config.StockConfig.INDICATOR_STATUS['failed_count'] += 1

