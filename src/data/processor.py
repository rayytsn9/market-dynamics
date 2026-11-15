import numpy as np
import pandas as pd
import talib.abstract as ta



class CryptoProcessor:
    """ CCXT specific data preprocessing class """

    @staticmethod
    def get_log_return_df(series: pd.Series) -> pd.Series:
        return np.log(series).diff()

    @staticmethod
    def build_panel(dataset: dict, N, M) -> pd.DataFrame:
        """
        Builds a panel of log returns from a dictionary of asset data
        dataset: dict of asset data (keys: symbols, values: log return series)
        N: number of trading days to include
        M: number of assets to include
        Returns: DataFrame of log returns with symbols as columns and trading days as index
        """
        panel = pd.concat(dataset, axis=1)
        panel = panel.sort_index()
        panel = panel.dropna(axis=1) 

        if panel.shape[1] == 0:
            raise ValueError("No assets with complete data")

        panel = panel.tail(N)

        if panel.shape[1] < M:
            raise ValueError(f"Only {panel.shape[1]} assets with complete data, need {M}")

        panel = panel.iloc[:, :M]

        return panel

    @staticmethod
    def mean_center(df: pd.DataFrame) -> np.ndarray:
        X = df.values
        return X - X.mean(axis=0)

    @staticmethod
    def clean_dataset(dataset: dict, N: int) -> dict:
        return {k: v for k, v in dataset.items() if len(v) >= N}

class StockProcessor:
    """ YFinance specific data preprocessing class """

    @staticmethod
    def compute_log_return(series: pd.Series) -> pd.Series:
        return np.log(series).diff()

    @staticmethod
    def construct_log_return_df(df: pd.DataFrame) -> pd.DataFrame:

        close_prices = df['Close']

        print(close_prices)

        return np.log(close_prices).diff()

    @staticmethod
    def add_log_returns_inplace(df: pd.DataFrame) -> pd.DataFrame:
        tickers = df.columns.get_level_values('Ticker').unique()
        
        for ticker in tickers:
            # Calculate for one ticker
            returns = StockProcessor.compute_log_return(df[('Close', ticker)])
            
            # Directly inject it using the explicit multi-index tuple key
            df[('Log_Return', ticker)] = returns
            
        return df.sort_index(axis=1)

class TaLibProcessor:

    """ TA-lib specific processor """

    _STATIC_BLUEPRINT = None

    @classmethod
    def instantiate_factory(cls, talib_map: dict):

        if cls._STATIC_BLUEPRINT is not None:
            return

        temp_dict = {}
        for key, val in talib_map.items():

            if val is None:
                temp_dict[key] = None
                continue

            func_name = val if isinstance(val, str) else val[0]

            if not hasattr(ta, func_name.upper()):
                raise KeyError(
                    f"Factory instantiation failed: Indicator : unsupported TA-Lib function '{func_name}' in map."
                )

            temp_dict[key] = val.upper() if isinstance(val, str) else val

        cls._STATIC_BLUEPRINT = temp_dict



    def __init__(self) -> None:
        
        if self._STATIC_BLUEPRINT is None:
            raise RuntimeError('Factory error: Call TaLibProcessor.instantiate_factory(TALIB_MAP) first.')

        self.local_functions = {}

        for key, val in self._STATIC_BLUEPRINT.items():

            if val is not None:

                if isinstance(val, str):
                    self.local_functions[key] = ta.Function(val)
                elif isinstance(val, tuple):
                    self.local_functions[key] = ta.Function(val[0])

            elif val is None:

                self.local_functions[key] = None
        

    def compute(self, key: str, df: pd.DataFrame, **override_kwargs):
    
        if key not in self.local_functions:
            raise KeyError(f'Indicator {key} not found')

        abstract_func = self.local_functions[key]
        if abstract_func is None:
            return None
        
        abstract_func.set_input_arrays(df)

        if override_kwargs:
            default_args = abstract_func.info['parameters']
            abstract_func.set_function_args(override_kwargs)

        try:
            output =  abstract_func()
        finally:
            if override_kwargs:
                abstract_func.set_function_args(default_args)
                
        is_split_indicator = isinstance(self._STATIC_BLUEPRINT[key], tuple)

        if is_split_indicator:
            return output.loc[:, self._STATIC_BLUEPRINT[key][1]]
        else:
            return output
        
    @property
    def _local_functions(self):
        return self.local_functions.copy()

    @property
    def _static_blueprint(self):
        return self._STATIC_BLUEPRINT.copy()

def _with_daily_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy a frame onto a timezone-naive calendar-day index.

    Daily bars from different exchanges can share a trading date while
    their midnight timestamps sit in different timezones. Intersection
    matches exact instants, so the timezone is removed and the clock is
    floored to that local date.
    """
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.DatetimeIndex(index)
    if index.tz is not None:
        index = index.tz_localize(None)
    frame = frame.copy()
    frame.index = index.normalize()
    return frame

def indicators_common_to(frames: dict[str, pd.DataFrame], processor: TaLibProcessor) -> list[str]:
    """
    Return the implemented indicator names that are usable on every asset.

    Dependencies:
    - frames: ticker -> OHLCV DataFrame. Each frame must have lowercase
      open, high, low, close, and volume columns and a DatetimeIndex.
      Indexes are aligned on the calendar date with the timezone removed
      before they are intersected.
    - processor: a TaLibProcessor whose factory was already built with
      TaLibProcessor.instantiate_factory. This function reads
      processor._local_functions and calls processor.compute. It does not
      read StockConfig.TALIB_MAP or StockConfig.ASSETS itself.
    - pandas: Index.intersection, Series.reindex, Series.dropna, and
      Series.nunique on each compute result. A DataFrame result uses only
      its first column.
    """
    if not frames:
        raise ValueError('frames is empty')

    frames = {ticker: _with_daily_index(frame) for ticker, frame in frames.items()}
    implemented = [key for key, func in processor._local_functions.items() if func is not None]
    shared_index = frames[next(iter(frames))].index
    for frame in frames.values():
        shared_index = shared_index.intersection(frame.index)

    common = []
    for key in implemented:
        usable = True
        for frame in frames.values():
            try:
                output = processor.compute(key, frame)
            except Exception:
                usable = False
                break
            if output is None:
                usable = False
                break
            values = output if isinstance(output, pd.Series) else output.iloc[:, 0]
            aligned = values.reindex(shared_index).dropna()
            if aligned.empty or aligned.nunique(dropna=True) < 2:
                usable = False
                break
        if usable:
            common.append(key)
    return common

def create_matrix(
    frames: dict[str, pd.DataFrame],
    indicators: list[str],
    processor: TaLibProcessor,
    n: int | None = None,
) -> pd.DataFrame:
    """
    Aligned panel of indicator values.

    Rows are the shared dates, tailed to n when n is set.
    Columns are a MultiIndex of (asset, indicator), in frames order
    then indicators order. A multi-column TA-Lib result (Bollinger Bands)
    becomes one column per output, named '{indicator}_{output}'.
    """
    if not frames:
        raise ValueError('frames is empty')
    if not indicators:
        raise ValueError('indicators is empty')
    if n is not None and n < 1:
        raise ValueError('n must be at least 1')

    frames = {ticker: _with_daily_index(frame) for ticker, frame in frames.items()}
    shared_index = None
    for frame in frames.values():
        shared_index = frame.index if shared_index is None else shared_index.intersection(frame.index)
    if len(shared_index) == 0:
        raise ValueError('frames have no dates in common')

    columns = {}
    for asset, frame in frames.items():
        for indicator in indicators:
            output = processor.compute(indicator, frame)
            if output is None:
                raise ValueError(f'{indicator} is not implemented')
            if isinstance(output, pd.Series):
                columns[(asset, indicator)] = output.reindex(shared_index)
            else:
                for col in output.columns:
                    columns[(asset, f'{indicator}_{col}')] = output[col].reindex(shared_index)

    matrix = pd.DataFrame(columns)
    matrix.columns = pd.MultiIndex.from_tuples(matrix.columns, names=['asset', 'indicator'])
    matrix = matrix.dropna(how='any')
    if matrix.empty:
        raise ValueError('no rows remain after alignment')
    if n is not None:
        if len(matrix) < n:
            raise ValueError(f'aligned matrix has {len(matrix)} rows, n={n}')
        matrix = matrix.tail(n)
    return matrix