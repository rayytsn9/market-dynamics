from ccxt.static_dependencies.toolz.curried import second
import numpy as np
import pandas as pd
import talib.abstract as ta
import inspect
import config


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

        # tickers = df.columns.get_level_values('Ticker').unique().to_list()
        
        # log_df = pd.DataFrame({i: StockProcessor.compute_log_return(df[('Close', i)]) for i in tickers})
        
        # return log_df

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
    

