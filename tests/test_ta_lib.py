import pytest
from src.data.processor import TaLibProcessor
import config

class TestTa:

    @pytest.mark.ta_pipeline
    def test_indicator(self, indicator_data, sample_market_data, ta_processor):
        
        indicator, func_name = indicator_data

        try:
            assert ta_processor.compute(indicator, sample_market_data) is not None, f'{indicator} function returned None!'
            TestTa._update_metrics(indicator,'PASSED')
        
        except Exception as e:
            TestTa._update_metrics(indicator, 'FAILED assert: compute is None')
            raise

    @pytest.mark.ta_fast
    def test_initialize_ta(self, invalid_ta_map, working_ta_map, is_valid=True):

        TaLibProcessor.instantiate_factory(working_ta_map) if is_valid else TaLibProcessor.instantiate_factory(invalid_ta_map)
        processor = TaLibProcessor()
        
        for key, val in processor._static_blueprint.items():
            assert val is None or isinstance(val, str) or isinstance(val, tuple)
            
        if is_valid:
            assert len(processor._static_blueprint) == len(working_ta_map)
        else:
            assert len(processor._static_blueprint) == len(invalid_ta_map)

            

    @classmethod
    def _update_metrics(cls, indicator: str, status: str):

        config.StockConfig.INDICATOR_STATUS['status_dict'][indicator] = status
        
        if status == 'PASSED':
            config.StockConfig.INDICATOR_STATUS['passed_count'] += 1
        elif 'FAIL' in status:
            config.StockConfig.INDICATOR_STATUS['failed_count'] += 1

        

