from pathlib import path
import pandas as pd
from dbrepository import RepositoryDB


class OhlcvDB(RepositoryDB):

    _PATH_NAME = None

    def __init__(self) -> None:
        pass

    @classmethod
    def instantiate_session(cls):
        cls._PATH_NAME = 'ohlcvdb.json'

    def add(self, data : pd.DataFrame):
        pass

    def get(self, **kwargs):
        pass



    

    