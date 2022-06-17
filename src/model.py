from abc import ABC, abstractmethod
from stubs import Array
import numpy as np
from typing import Callable, TypeAlias

LoglikeFn: TypeAlias = Callable[[Array], float]
Prior: TypeAlias = Callable[[Array], Array]


class Model(ABC):
    def __call__(self, x: Array, p: Array) -> Array:
        return self.eval(x, p)

    @abstractmethod
    def eval(self, x: Array, p: Array) -> Array:
        ...

    @abstractmethod
    def design_matrix(self, x: Array) -> Array:
        ...

    @abstractmethod
    def prior_transform(self, cube: Array) -> Array:
        ...

    @abstractmethod
    def parameters(self) -> list[str]:
        ...

    @abstractmethod
    def prefix_(self) -> str:
        ...

    def prefix(self) -> str:
        return self.prefix_() + "/mn_"

    @property
    @abstractmethod
    def num_parameters(self) -> int:
        ...

    def __add__(self, other: 'Noise') -> 'DataModel':
        # For some reason the isinstance doesn't work.
        #if issubclass(type(other), Noise):
        return DataModel(self, other)
        #raise ValueError(f"Can not add type {type(other)}")


class LogpolyModel(Model):
    def __init__(self, order: int, bias: bool = True):
        self.order = order
        self.bias = bias

    @property
    def num_parameters(self) -> int:
        return self.order + 1

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
        return cube

    @staticmethod
    def eval(parameters: Array, X: Array) -> Array:
        mu = X @ parameters
        return mu

    def prefix_(self) -> str:
        return f'logpoly{self.order}'

    def parameters(self) -> list[str]:
        return [f'C{i}' for i in range(self.order+1)]


class Gsf3Model(Model):
    """

    e = EXP{ [ (A+B*x+C*x^2 + ...)**(-G) + (D+E*y+F*y^2 + ...)**(-G) ]**(-1/G) },
    where x = log(EG/E1) and y = log(EG/E2), med E1=100keV og E2=1MeV
    By convention, C should be close to 0, so higher orders are discouraged
    """
    def __init__(self, order: int = 2):
        self.order = order
        self.E1 = 1e2
        self.E2 = 1e3

    @property
    def num_parameters(self) -> int:
        return 2*(self.order + 1) + 1

    def design_matrix(self, x: Array) -> Array:
        N = x.shape[0]
        l = self.order+1
        h = 2*l
        X = np.empty((N, h))
        # A, B, C, ...
        X[:, 0] = 1.0
        for i in range(1, l):
            X[:, i] = (np.log10(x / self.E1))**i
        # D, E, F, ...
        X[:, l] = 1.0
        for i in range(l+1, h):
            X[:, i] = (np.log10(x / self.E2))**i
        # G

        return X

    def prior_transform(self, cube: Array) -> Array:
        for i in range(2*(self.order+1)):
            cube[i] = 200*cube[i] - 100
        #cube[2] = 0*cube[i]
        #cube[-2] = 0*cube[i]
        cube[-1] = 200*cube[i] - 100
        return cube

    def eval(self, parameters: Array, X: Array) -> Array:
        l = self.order+1
        h = 2*l
        low = X[:, :l] @ parameters[:l]
        high = X[:, l:h] @ parameters[l:h]
        G = parameters[-1]
        mu = (low ** (-G) + high ** (-G)) ** (-1/G)
        return mu

    def prefix_(self) -> str:
        return f'gsf3_{self.order}'

    def parameters(self) -> list[str]:
        low = [f'L{i}' for i in range(self.order+1)]
        high = [f'H{i}' for i in range(self.order+1)]
        return low + high + ['G']

class Noise(ABC):
    @abstractmethod
    def prefix_(self) -> str:
        ...

    @abstractmethod
    def parameters(self) -> list[str]:
        ...

    @abstractmethod
    def prior_transform(self, cube: Array) -> Array:
        ...

    @abstractmethod
    def likelihood(self, p: Array, y: Array, y_hat: Array) -> float:
        ...

    @property
    @abstractmethod
    def num_parameters(self) -> int:
        ...

class EmpiricalPoisson(Noise):
    def likelihood(self, p: Array, y: Array, y_hat: Array) -> float:
        return -0.5 * (((y - y_hat)/y)**2).sum()

    @property
    def num_parameters(self) -> int:
        return 0

    def prefix_(self) -> str:
        return 'EP'

    def parameters(self) -> list[str]:
        return []

    def prior_transform(self, cube: Array) -> Array:
        return cube


class FitNoise(Noise):
    def __init__(self, prior: Prior):
        self.prior = prior

    def likelihood(self, p: Array, y: Array, y_hat: Array) -> float:
        return -0.5 * ((y - y_hat)**2/p[0]).sum()

    @property
    def num_parameters(self) -> int:
        return 1

    def prefix_(self) -> str:
        return 'FN'

    def parameters(self) -> list[str]:
        return ['sigma']

    def prior_transform(self, cube: Array) -> Array:
        cube[-1] = self.prior(cube[-1])
        return cube


class DataModel:
    """ Wrapper that combines a Model and a Noise

    """
    def __init__(self, model: Model, noise: Noise):
        self.model = model
        self.noise = noise

    def design_matrix(self, x: Array) -> Array:
        return self.model.design_matrix(x)

    def prior_transform(self, cube: Array) -> Array:

        model_cube, noise_cube = self.split_parameters(cube)
        print(len(model_cube))
        print(len(noise_cube))

        self.model.prior_transform(model_cube)
        self.noise.prior_transform(noise_cube)

        return cube

    def mean(self, parameters: Array, X: Array) -> Array:
        return self.model.mean(parameters, X)

    def prefix_(self) -> str:
        s = self.model.prefix_()
        s += self.noise.prefix_()
        return s

    def prefix(self) -> str:
        return self.prefix_() + "/mn_"

    def parameters(self) -> list[str]:
        return self.model.parameters() + self.noise.parameters()

    def likelihoodfn(self, design_matrix: Array,
                     y_observations: Array) -> LoglikeFn:
        ylog = np.log10(y_observations)
        def fn(params: Array):
            modelp, noisep = self.split_parameters(params)
            yhat = self.model(modelp, design_matrix)
            loglike = self.noise.likelihood(noisep, ylog, yhat)
            if not np.isfinite(loglike):
                loglike = 1e-12
            return loglike

        return fn

    def split_parameters(self, params: Array) -> tuple[Array, Array]:
        if self.num_parameters != len(params):
            raise RuntimeError("Model #parameters != #parameters")
        model = params[:self.model.num_parameters]
        noise = params[self.model.num_parameters:]
        return model, noise

    def split_samples(self, samples: Array) -> tuple[Array, Array, Array]:
        if self.num_parameters != samples.shape[1] - 1:
            raise RuntimeError("Model #parameters != samples")
        model_params = samples[:, :self.model.num_parameters]
        noise_params = samples[:, self.model.num_parameters:-1]
        likelihood = samples[:, -1]
        return model_params, noise_params, likelihood

    @property
    def num_parameters(self) -> int:
        return self.model.num_parameters + self.noise.num_parameters

    def __call__(self, x: Array, p: Array) -> Array:
        return self.model(x, p)
