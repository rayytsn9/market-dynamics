import pytest
from src.data.dataset import NoCommonIndicators, build_asset_matrix, drop_duplicate_columns
from src.data.processor import TaLibProcessor, indicators_common_to, create_matrix
import config
import numpy as np
import pandas as pd

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

    @pytest.mark.pipeline
    def test_workflow(self, ta_processor, ta_map, sample_market_data):

        indicator_dict = {key: ta_processor.compute(key, sample_market_data) for key, val in ta_map.items() if val is not None}

        for key, val in indicator_dict.items():
            print(f'====================================================')
            print(f'                {key}      ')
            print(f'        type:{type(val)}        ')
            print(f'====================================================')
            print(val)
            print(f'====================================================')

        print(len(indicator_dict))

    @pytest.mark.ta_fast
    def test_indicators_common_to(self, ta_processor, sample_market_data):
        """An implemented indicator is kept only when both frames still vary on shared dates.

        Assumptions:
        - ta_processor is built from the full TALIB_MAP, including None entries.
        - sample_market_data is 100 rows of random open, high, low, close, and volume
          with an integer index. The test copies it onto a business-day index so the
          two frames can be aligned by date.
        - priced keeps that random volume. flat_volume is the same prices with volume
          set to 0.
        - The keys 'SPY' and '^VIX' are labels only. The function does not look them up.

        Constraints:
        - Simple Moving Average must remain. It uses close, and both frames share the
          same varying closes, so the aligned series has many distinct finite values.
        - On Balance Volume and Money Flow Index must drop. Both use volume. On the
          zero-volume frame, On Balance Volume stays flat, and Money Flow Index does
          not produce two distinct finite values.
        - This test does not check the other implemented names, a missing shared date,
          an empty frames dictionary, or a live Yahoo download of ^VIX.
        """

        index = pd.bdate_range('2020-01-01', periods=len(sample_market_data))
        priced = sample_market_data.copy()
        priced.index = index
        flat_volume = priced.copy()
        flat_volume['volume'] = 0.0

        common = indicators_common_to(
            {'SPY': priced, '^VIX': flat_volume},
            ta_processor,
        )

        assert 'Simple Moving Average' in common
        assert 'On Balance Volume' not in common
        assert 'Money Flow Index' not in common

    @pytest.mark.ta_fast
    def test_indicators_common_to_aligns_calendar_dates(self, ta_processor):
        """The same trading date matches across exchange timezones.

        Assumptions:
        - Both frames use the same business dates and the same varying prices.
        - SPY is stamped America/New_York and ^VIX is stamped America/Chicago,
          both at local midnight.

        Constraints:
        - Simple Moving Average is common. The one-hour offset is not treated
          as a different observation.
        """
        rows = 80
        dates = pd.bdate_range('2024-01-02', periods=rows)
        rng = np.random.default_rng(1)
        close = 100 + rng.normal(0, 1, rows).cumsum()
        base = pd.DataFrame(
            {
                'open': close,
                'high': close + 1,
                'low': close - 1,
                'close': close,
                'volume': rng.uniform(1000, 5000, rows),
            }
        )
        ny = base.copy()
        ny.index = dates.tz_localize('America/New_York')
        chicago = base.copy()
        chicago.index = dates.tz_localize('America/Chicago')

        common = indicators_common_to({'SPY': ny, '^VIX': chicago}, ta_processor)

        assert 'Simple Moving Average' in common

    @pytest.mark.ta_fast
    def test_create_matrix(self, ta_processor):
        """Two assets over 200 business days produce one aligned panel.

        Assumptions:
        - Both frames share the same business-day index, so the shared calendar is 200 dates.
        - Prices vary and volume is positive, so Simple Moving Average and Bollinger Bands
          both return finite values after their warmup.
        - Indicator order is Simple Moving Average, then Bollinger Bands. Asset order is SPY, then QQQ.

        Constraints:
        - n=100 keeps the last 100 shared rows. The warmup is shorter than 200 - 100,
          so the slice succeeds.
        - The panel has one column per asset for Simple Moving Average, and one column
          per Bollinger Bands output per asset. Every value is finite.
        - The last row is the last input date.
        """
        rows = 200
        index = pd.bdate_range('2020-01-01', periods=rows)
        rng = np.random.default_rng(42)

        def frame(seed_shift: float) -> pd.DataFrame:
            close = 100 + seed_shift + rng.normal(0, 1, rows).cumsum()
            return pd.DataFrame(
                {
                    'open': close + rng.normal(0, 0.2, rows),
                    'high': close + rng.uniform(0.1, 1.0, rows),
                    'low': close - rng.uniform(0.1, 1.0, rows),
                    'close': close,
                    'volume': rng.uniform(1000, 5000, rows),
                },
                index=index,
            )

        frames = {'SPY': frame(0), 'QQQ': frame(5)}
        indicators = ['Simple Moving Average', 'Bollinger Bands']

        matrix = create_matrix(frames, indicators, ta_processor, n=100)

        assert matrix.shape[0] == 100
        assert matrix.notna().all().all()
        assert matrix.index[-1] == index[-1]
        assert matrix.columns[0] == ('SPY', 'Simple Moving Average')
        assert ('QQQ', 'Simple Moving Average') in matrix.columns
        spy_bands = [name for asset, name in matrix.columns if asset == 'SPY' and name.startswith('Bollinger Bands_')]
        qqq_bands = [name for asset, name in matrix.columns if asset == 'QQQ' and name.startswith('Bollinger Bands_')]
        assert len(spy_bands) == 3
        assert spy_bands == qqq_bands

    @pytest.mark.ta_fast
    def test_build_asset_matrix(self, ta_processor, monkeypatch):
        """build_asset_matrix fetches StockConfig.ASSETS and returns their common panel.

        Assumptions:
        - ASSETS is patched to SPY and QQQ. fetch_ohlcv is patched to return the
          two local frames, so the test does not call Yahoo.
        - Both frames share 200 business days, with varying prices and positive volume.

        Constraints:
        - The panel has 50 rows, the n passed to build_asset_matrix.
        - Both tickers appear as the asset level of the column index.
        - Moving Average is present for both. Simple Moving Average is absent,
          because it is an exact duplicate of Moving Average for each asset.
        """
        rows = 200
        index = pd.bdate_range('2020-01-01', periods=rows)
        rng = np.random.default_rng(7)

        def frame(seed_shift: float) -> pd.DataFrame:
            close = 100 + seed_shift + rng.normal(0, 1, rows).cumsum()
            return pd.DataFrame(
                {
                    'open': close + rng.normal(0, 0.2, rows),
                    'high': close + rng.uniform(0.1, 1.0, rows),
                    'low': close - rng.uniform(0.1, 1.0, rows),
                    'close': close,
                    'volume': rng.uniform(1000, 5000, rows),
                },
                index=index,
            )

        frames = {'SPY': frame(0), 'QQQ': frame(5)}
        monkeypatch.setattr(config.StockConfig, 'ASSETS', ['SPY', 'QQQ'])
        monkeypatch.setattr(
            'src.data.dataset.StockFetcher.fetch_ohlcv',
            lambda symbol, timeframe='1d', since=None, limit=None, period=None: frames[symbol],
        )

        matrix = build_asset_matrix(ta_processor, n=50)

        assert len(matrix) == 50
        assert set(matrix.columns.get_level_values('asset')) == {'SPY', 'QQQ'}
        assert ('SPY', 'Moving Average') in matrix.columns
        assert ('QQQ', 'Moving Average') in matrix.columns
        assert ('SPY', 'Simple Moving Average') not in matrix.columns
        assert ('QQQ', 'Simple Moving Average') not in matrix.columns

    @pytest.mark.ta_fast
    def test_build_asset_matrix_no_common_indicators(self, ta_processor, monkeypatch):
        """An empty common-indicator list is reported instead of calling create_matrix."""
        monkeypatch.setattr(config.StockConfig, 'ASSETS', ['SPY'])
        monkeypatch.setattr(
            'src.data.dataset.StockFetcher.fetch_ohlcv',
            lambda symbol, timeframe='1d', since=None, limit=None, period=None: pd.DataFrame(),
        )
        monkeypatch.setattr('src.data.dataset.indicators_common_to', lambda frames, processor: [])

        with pytest.raises(NoCommonIndicators, match='SPY'):
            build_asset_matrix(ta_processor)

    @pytest.mark.ta_fast
    def test_drop_duplicate_columns_per_asset(self):
        """Identical series are dropped within an asset, and the first column is kept.

        Assumptions:
        - Columns are ordered Moving Average, then Simple Moving Average, per asset.
        - For each asset those two series are exactly equal.
        - SPY's Moving Average equals QQQ's Moving Average. That cross-asset
          match is not a duplicate.

        Constraints:
        - Simple Moving Average is removed for SPY and for QQQ.
        - Moving Average remains for both assets.
        """
        index = pd.bdate_range('2024-01-02', periods=4)
        values = [1.0, 2.0, 3.0, 4.0]
        other = [4.0, 3.0, 2.0, 1.0]
        matrix = pd.DataFrame(
            {
                ('SPY', 'Moving Average'): values,
                ('SPY', 'Simple Moving Average'): values,
                ('SPY', 'Momentum Indicator'): other,
                ('QQQ', 'Moving Average'): values,
                ('QQQ', 'Simple Moving Average'): values,
                ('QQQ', 'Momentum Indicator'): other,
            },
            index=index,
        )
        matrix.columns = pd.MultiIndex.from_tuples(matrix.columns, names=['asset', 'indicator'])

        cleaned = drop_duplicate_columns(matrix)

        assert list(cleaned.columns) == [
            ('SPY', 'Moving Average'),
            ('SPY', 'Momentum Indicator'),
            ('QQQ', 'Moving Average'),
            ('QQQ', 'Momentum Indicator'),
        ]

    @classmethod
    def _update_metrics(cls, indicator: str, status: str):

        config.StockConfig.INDICATOR_STATUS['status_dict'][indicator] = status
        
        if status == 'PASSED':
            config.StockConfig.INDICATOR_STATUS['passed_count'] += 1
        elif 'FAIL' in status:
            config.StockConfig.INDICATOR_STATUS['failed_count'] += 1

        

