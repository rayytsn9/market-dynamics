import pytest
import config
from src.data.processor import TaLibProcessor, ta
import numpy as np
import pandas as pd

@pytest.fixture
def invalid_ta_map():

    invalid_map = {
        "Accumulation Distribution": "AD",
        "ADX": "ADX",
        "Aroon Oscillator": "AROONOSC",
        "ATR Bands": None,
        "ATR Trailing Stops": "hehe",
        "Average True Range": "ATR",
    }

    yield invalid_map

@pytest.fixture
def working_ta_map():

    talib_map = {key: val for key, val in config.StockConfig.TALIB_MAP.items() if val is not None}
    yield talib_map
    

@pytest.fixture(scope='function')
def ta_map():
    
    talib_map = config.StockConfig.TALIB_MAP
    yield talib_map


@pytest.fixture(scope='function')
def ta_processor(ta_map):

    print('Setting up ta processor')
    TaLibProcessor.instantiate_factory(ta_map)
    processor = TaLibProcessor()

    yield processor

    TaLibProcessor._STATIC_BLUEPRINT = None
    print('Tearing down ta processor')


@pytest.fixture(scope='session')
def sample_market_data():
    """Generates mock OHLCV data to test baseline TA-Lib matrix compatibility."""
    
    print("Generating sample market data")
    
    np.random.seed(42)
    rows = 100

    df = pd.DataFrame({
        'open': np.random.uniform(100, 110, rows),
        'high': np.random.uniform(110, 115, rows),
        'low': np.random.uniform(95, 100, rows),
        'close': np.random.uniform(100, 110, rows),
        'volume': np.random.uniform(1000, 5000, rows)
    })

    yield df

    print("Tearing down sample market data")

def pytest_generate_tests(metafunc):

    TARGET_ARGUMENTS = ['indicator_data']
    
    for target_arg in TARGET_ARGUMENTS:

        if target_arg in metafunc.fixturenames and target_arg == 'indicator_data':

            parametrized_cases = []
            for indicator, func_name in config.StockConfig.TALIB_MAP.items():

                if func_name is None:
                    test_case = pytest.param((indicator, func_name), marks=pytest.mark.xfail(reason=f'{indicator} not yet implemented'), id=indicator)

                elif isinstance(func_name, str) and not hasattr(ta, func_name.upper()):
                    test_case = pytest.param((indicator, func_name), marks=pytest.mark.xfail(reason=f'{func_name} not supported by TA-LIB.'), id=indicator)

                elif isinstance(func_name, tuple) and not hasattr(ta, func_name[0].upper()):
                    test_case = pytest.param((indicator, func_name), marks=pytest.mark.xfail(reason=f'{func_name} not supported by TA-LIB.'), id=indicator)

                else:
                    test_case = pytest.param((indicator, func_name), id=indicator)

                parametrized_cases.append(test_case)

            metafunc.parametrize(argnames=target_arg,argvalues=parametrized_cases)


def pytest_terminal_summary(terminalreporter):
    """
    Runs automatically at the absolute end of the test suite run.
    """
    # Fetch your centralized metrics dictionary from config.py
    metrics = config.StockConfig.INDICATOR_STATUS
    status_map = metrics['status_dict']

    tr = terminalreporter
    tr.ensure_newline()
    tr.section("CUSTOM METRICS DASHBOARD", sep="=", blue=True)
    
    tr.write_line(f"Passed Operations Count : {metrics['passed_count']}")
    tr.write_line(f"Failed Operations Count : {metrics['failed_count']}")
    
    tr.ensure_newline()
    
    if status_map:
        # Create DataFrame from dictionary keys and values
        df = pd.DataFrame(list(status_map.items()), columns=["Technical Indicator", "Status"])
        
        # Set the Indicator as the index to remove the raw 0, 1, 2 row numbers
        df.set_index("Technical Indicator", inplace=True)
        
        # 3. Convert the DataFrame to a clean string format
        # This keeps columns neatly aligned regardless of indicator name lengths
        df_string = df.to_string(justify="left")
        
        # Print it directly to the Pytest terminal window
        tr.write_line(df_string)
    else:
        tr.write_line("Status Tracker: No metrics recorded.")
        
    tr.ensure_newline()

