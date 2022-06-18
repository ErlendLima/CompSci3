from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data import ResponseData
from stubs import Array, Axes
from typing import Literal

try:
    from tqdm.autonotebook import tqdm
except ImportError:
    tqdm = lambda x: x

SampleType = Literal['predictions', 'samples']


class Posterior:
    importance_evidence: list[float]
    evidence: list[float]
    mean: Array
    std: Array
    ml: Array
    MAP: Array

    def __init__(self, data: ResponseData, prefix: str | None = None):
        prefix = data.prefix() if prefix is None else prefix
        self.samples = np.loadtxt(prefix + 'post_equal_weights.dat')
        self.data = data
        self.mu: Array = np.array([])
        self.sigma: Array = np.array([])
        self.predictions: Array = np.array([])
        mp, np_, lh = data.model.split_samples(self.samples)
        self.model_params = mp
        self.noise_params = np_
        self.likelihood = lh
        self._load_stats()
        self._load_samples()

    def predict(self, X: Array | None = None,
                parameters: Array | None = None) -> Array:
        if X is None:
            X, _ = self.data.get_data()
        if parameters is None:
            parameters = self.mean
        if len(parameters) != X.shape[1]:
            parameters, _ = self.data.model.split_parameters(parameters)
        return self.data.model(parameters, X)

    def _load_samples(self, x: Array | None = None,
                      y: Array | None = None,
                      design_matrix: Array | None = None,
                      nsamples: int | None = None):
        if (x is None and y is not None) or (x is not None and y is None):
            raise ValueError("x and y must be both None or both not None")
        x = self.data.x if x is None else x
        y = self.data.y if y is None else y
        if design_matrix is None:
            design_matrix, _ = self.data.get_data()
        nsamples = len(self) if nsamples is None else nsamples
        if nsamples > len(self):
            raise ValueError(f"Too many samples. {nsamples} > {len(self)}")

        N = len(x)
        mu = np.zeros((nsamples, N))
        sigma = np.zeros_like(mu)
        for i in range(nsamples):
            mu[i, :] = self.data.model(self.model_params[i], design_matrix)
            sigma[i, :] = self.data.model.noise(self.noise_params[i], y, mu[i, :])

        self.mu = mu
        self.sigma = sigma

    def _load_stats(self) -> None:
        N = self.data.model.num_parameters
        with Path(self.data.prefix() + 'stats.dat').open('r') as f:
            def next(i: int = 1) -> str:
                for i in range(i):
                    s = f.readline()
                return s

            sampling = next().split(':')[1].split('+/-')
            importance_sampling = next().split(':')[1].split('+/-')
            sampling: list[float] = [float(x) for x in sampling]
            importance_sampling: list[float] = [float(x) for x in importance_sampling]
            next(2)
            mean = [next().split()[1:] for _ in range(N)]
            mean = [(float(mu), float(std)) for mu, std in mean]
            mean: list[tuple[float, float]] = list(zip(*mean))
            next(3)
            ml = [next().split()[1] for _ in range(N)]
            ml: list[float] = [float(x) for x in ml]
            next(3)
            map = [next().split()[1] for _ in range(N)]
            map: list[float] = [float(x) for x in map]
        self.MAP = np.asarray(map)
        self.ml = np.asarray(ml)
        self.mean = np.asarray(mean[0])
        self.std = np.asarray(mean[1])
        self.evidence = sampling
        self.importance_evidence = importance_sampling

        if not np.allclose(self.MAP, self.ml):
            print("ML is not equal to MAP")

    def sample_predict(self, nsamples: int | None = None,
                       size: int = 1) -> Array:
        if not len(self.mu):
            raise ValueError("Call load_samples() first")
        N = self.mu.shape[0]
        nsamples = N if nsamples is None else nsamples
        if nsamples > N:
            raise ValueError(f"Too many samples. {nsamples} > {N}")

        predictions = np.zeros_like(self.mu)
        m = self.mu.shape[1]
        for i in tqdm(range(nsamples)):
            predictions[i, :] = np.random.normal(self.mu[i, :], self.sigma[i, :], (size, m))

        # We save in the special case of a single sample from each point
        if size == 1:
            self.predictions = predictions
        return predictions

    def sample_predictions(self, n, which: str = 'mean'):
        pass

    def plot(self, x: Array | None = None,
             ax: Axes | None = None,
             quantile: float | None = 0.05,
             **kwargs) -> Axes:
        if ax is None:
            fig, ax = plt.subplots()
        if x is None:
            x = self.data.x
        self.plot_mean(ax=ax, **kwargs)
        if quantile is not None:
            self.plot_quantiles(quantile, ax)
        return ax

    def plot_quantiles(self, alpha_: float = 0.05,
                       ax: Axes | None = None,
                       **kwargs) -> Axes:
        if ax is None:
            fig, ax = plt.subplots()
        if False:# len(self.predictions) < 1:
            print("Call predict() to plot quantiles.")
        else:
            kwargs = {'alpha': 0.3} | kwargs
            x = self.data.x
            low, high = self.quantiles(alpha_)
            ax.fill_between(x, y1=low, y2=high, **kwargs)
        return ax

    def quantiles(self, alpha: float = 0.05,
                  which: SampleType = 'samples') -> tuple[Array, Array]:
        if which == 'samples':
            y = self.mu
        elif which == 'predictions':
            y = self.predictions
        else:
            raise ValueError(f"Unknown sample type: {which}")

        low = np.quantile(10 ** y, q=alpha / 2, axis=0)
        high = np.quantile(10 ** y, q=1 - alpha / 2, axis=0)
        return low, high

    def plot_mus(self, x: Array | None = None,
                 nsamples: int | None = None,
                 ax: Axes | None = None,
                 **kwargs) -> Axes:
        if ax is None:
            fig, ax = plt.subplots()
        if x is None:
            x = self.data.x
        nsamples = len(self) if nsamples is None else nsamples
        if nsamples > len(self):
            raise ValueError(f"Too many samples. {nsamples} > {len(self)}")

        kwargs = {'color': 'r', 'alpha': 0.1} | kwargs

        if len(self.mu) < 1:
            print("Call sample() to plot quantiles.")
        else:
            for y in self.mu[:nsamples]:
                ax.plot(x, 10 ** y, **kwargs)

        return ax

    def plot_predictions(self, n: int | None = None, ax: Axes | None = None,
                         **kwargs) -> Axes:
        if ax is None:
            fig, ax = plt.subplots()
        _n = self.predictions.shape[0]
        n = _n if n is None else n
        if n > _n:
            raise ValueError(f"Too many samples. {n} > {_n}")
        for i in range(n):
            ax.scatter(self.data.x, 10 ** self.predictions[i, :], **kwargs)
        return ax

    def plot_samples(self, n: int | None = None, ax: Axes | None = None,
                     **kwargs) -> Axes:
        if ax is None:
            fig, ax = plt.subplots()
        _n = self.predictions.shape[0]
        n = _n if n is None else n
        if n > _n:
            raise ValueError(f"Too many samples. {n} > {_n}")
        for i in range(n):
            ax.plot(self.data.x, self.samples[i, :], **kwargs)
        return ax

    def plot_mean(self, ax: Axes | None = None, **kwargs) -> Axes:
        if ax is None:
            fig, ax = plt.subplots()

        x = self.data.x
        design_matrix = self.data.get_data()[0]
        mean = 10 ** self.predict(design_matrix, self.mean)
        kwargs = {'color': 'r'} | kwargs

        ax.plot(x, mean, **kwargs)
        return ax

    def plot_ml(self, ax: Axes | None = None, **kwargs) -> Axes:
        if ax is None:
            fig, ax = plt.subplots()

        x = self.data.x
        design_matrix = self.data.get_data()[0]
        ml = 10 ** self.predict(design_matrix, self.ml)
        kwargs = {'color': 'g'} | kwargs

        ax.plot(x, ml, **kwargs)
        return ax

    def summary(self) -> str:
        s = ''

        def put(x):
            nonlocal s
            s += x + '\n'

        labels = self.data.model.parameters()
        # this is the plus minus symbol
        pm = '\u00b1'
        # Mean
        put(f'Posterior for {self.data.model.name}')
        put(f'Nested sampling global log evidence:            {self.evidence[0]:.8f} {pm} {self.evidence[1]:.8f}')
        put(f'Nested importance sampling global log evidence: {self.importance_evidence[0]:.8f} {pm} {self.importance_evidence[1]:.8f}')
        put('')
        put(f'Param        Mean               Std')
        for label, mu, sigma in zip(labels, self.mean, self.std):
            put(f'{label:<13}{mu:<13.8f} {pm} {sigma:.8f}')
        put('')

        mlequalmap = np.allclose(self.ml, self.MAP)
        put("Param         Maximum Likelihood" + " \ MAP" if mlequalmap else "")
        for label, ml in zip(labels, self.ml):
            put(f'{label:<13}{ml:.8e}')
        if not mlequalmap:
            put('')
            put("MAP")
            for map in self.MAP:
                put(f'{map:.8e}')
        # CL
        return s

    def __len__(self) -> int:
        return self.samples.shape[0]
