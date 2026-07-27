from pathlib import Path
import pandas as pd
from dbrepository import RepositoryDB

class FeatureDB(RepositoryDB):

    _PATH_NAME = None

    def __init__(self) -> None:
        pass

    @classmethod
    def instantiate_session(cls):
        cls._PATH_NAME = 'featuredb.json'

    def add(self, data : pd.DataFrame):
        pass

    def get(self, **kwargs):
        pass
