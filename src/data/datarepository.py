import abc as abc

class DataRepository(abc.ABC):

    @abc.abstractmethod
    def add(self, **kwargs):
        pass

    @abc.abstractmethod
    def get(self, **kwargs):
        pass