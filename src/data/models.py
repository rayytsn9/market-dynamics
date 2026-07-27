from dataclasses import dataclass
from datarepository import DataRepository
import pandas as pd


# =========================================
# Stock Design: Value Object -> a change 
# 
# =========================================
class Stock_Deprecated:

    def __init__(self) -> None:
        self._ticker: str | None = None
        self._data: pd.DataFrame | None = None
        self._metadata: dict[str, any] = {} 
        

    def set_data(self, ticker, data, **kwargs):
        
        self._data = data
        self._ticker = ticker

        if kwargs:
            for key, val in kwargs.items():
                self._metadata[key] = val

    @property
    def _info(self):

        info_dict = {
            'ticker' : self._ticker, 
            'data': self._data, 
            'features': self._features if self._features is not None else None,
            'metadata': self._metadata,
        }

        return info_dict

@dataclass(frozen=True)
class Stock(DataRepository):
    ticker: str
    data: pd.DataFrame
    metadata: dict[str, any] = None

    def info(self):
        return {
            'ticker': self.ticker,
            'data': self.data,
            'metadata': self.metadata if self.metadata is not None else {},
        }

    def set_data(self, ticker, data, **kwargs):
        
        self.data = data
        self.ticker = ticker

        if kwargs:
            for key, val in kwargs.items():
                self.metadata[key] = val

    def add(self, **kwargs):
        if kwargs:
            for key, val in kwargs.items():
                if key not in self.metadata:
                    self.metadata[key] = val
                else:
                    raise KeyError(f'key : {key} already exists')


    def get(self, **kwargs):
        if kwargs:
            if 'ticker' in kwargs.keys():
                return (self.data, self.metadata.copy())


# =========================================
# Feature Design:
# 
# cache with metadata = asset_type='stock' or 'crypto' etc
# =========================================
class Feature(DataRepository):

    ticker: str
    data: pd.DataFrame
    metadata: dict[str, any] = None

    def set_data(self, ticker, data, **kwargs):
        self.ticker = ticker
        self.data = data

    def __init__(self) -> None:
        self.asset : "Stock" | None = None
        self.feature_data : pd.Series | pd.DataFrame | None = None
        self.feature_metadata : dict[str, any] = {

            'from_data' : None,
            'to_data' : None,
            'parameters' : {},

        }

    @classmethod
    def from_data(cls, ticker: str, data: pd.DataFrame, **kwargs):
        
        instance = cls()
        stock_instance = Stock()
        stock_instance.set_data(ticker=ticker, data=data, **kwargs)
        instance.asset = stock_instance

        if not data.empty and isinstance(data.index, pd.DatetimeIndex):
            instance.feature_data['from_data'] = data.index.min()
            instance.feature_data['to_data'] = data.index.max()

        return instance

    def standardize(self):
        pass

    def add(self, **kwargs):
        pass

    def get(self, **kwargs):
        pass

