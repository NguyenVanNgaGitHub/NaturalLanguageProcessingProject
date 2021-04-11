from abc import ABC, abstractmethod

class AbstractModel(ABC):
    def __init__(self):
        self.clf = None

    @abstractmethod
    def fit(self, X, y):
        return NotImplementedError

    @abstractmethod
    def transform(self, X):
        return NotImplementedError