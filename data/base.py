from abc import ABC,abstractmethod
class DatasetProvider(ABC):
    @abstractmethod
    def build(self): ...
