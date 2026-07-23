import pandas as pd


# =========================================
# Stock Design:
# 
# members :  {   
#       'ticker' : ticker_name
#       'data' : pd.DataFrame(ohlcv)
#       'features' : dict[str, Feature]
#       'metadata' : parameters e.g. since, period, limit, ...
# 
#   }
# =========================================

class Stock:

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

# =========================================
# Feature Design:
# 
# cache with metadata = asset_type='stock' or 'crypto' etc
# =========================================
class Feature:

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