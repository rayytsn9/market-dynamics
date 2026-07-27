import abc as abc
import pandas as pd

class RepositoryDB(abc.ABC):

    @abc.abstractmethod
    def add(self, data : pd.DataFrame):
        pass

    @abc.abstractmethod
    def get(self, **kwargs):
        pass    
