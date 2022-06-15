from abc import ABC, abstractmethod
from stubs import Array
import numpy as np


class Model(ABC):
    @abstractmethod
    def design_matrix(self, x: Array) -> Array:
        ...

    @abstractmethod
    def prior_transform(self, cube: Array) -> Array:
        ...

    @abstractmethod
    def prefix(self) -> str:
        ...

    @abstractmethod
    def parameters(self) -> list[str]:
        ...

    @staticmethod
    @abstractmethod
    def mean(parameters: Array, X: Array) -> Array:
        ...


class LogpolyModel(Model):
    def __init__(self, order: int, bias: bool = True,
                 sigma: float = 1):
        self.order = order
        self.bias = bias
        self.sigma = sigma

    def design_matrix(self, x: Array) -> Array:
        N = x.shape[0]
        X = np.empty((N, self.order + 1))
        X[:, 0] = 1.0
        for i in range(1, self.order + 1):
            X[:, i] = (np.log10(x))**i

        X = X if self.bias else X[:, 1:]
        return X

    def prior_transform(self, cube: Array) -> Array:
        for i in range(self.order+1):
            cube[i] = 200*cube[i] - 100
        cube[-1] = self.sigma * cube[-1]
        return cube

    @staticmethod
    def mean(parameters: Array, X: Array) -> Array:
        mu = X @ np.reshape(parameters[:-1], (-1, 1))
        return mu.flatten()

    def prefix(self) -> str:
        return f'logpoly{self.order}/mn_'

    def parameters(self) -> list[str]:
        return [f'C{i}' for i in range(self.order+1)] + ['sigma']


