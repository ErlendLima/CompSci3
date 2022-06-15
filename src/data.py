import numpy as np
import pandas as pd
from typing import Iterable
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.scale import LogScale
from stubs import Axes, Array
from utils import maybe_set_xlabel, maybe_set_ylabel
from model import Model


class SynteticData():

    def __init__(self,
                 x_range: Iterable[float] = [0, 1],
                 y_range: Iterable[float] = [0, 1],
                 z_range: Iterable[float] = [0, 1],
                 Nx: int = 20,
                 Ny: int = 20,
                 Nz: int = 20,
                 N: int = 1000,
                 noise_sigma: float = 0.1,
                 grid_type: str = 'random'):
        self.grid_type = grid_type
        if self.grid_type == 'random':
            self.N = N
            self.X = self.make_random_design_matrix(x_range, y_range, z_range)
        else:
            self.Nx = Nx
            self.Ny = Ny
            self.Nz = Nz
            self.X = self.make_eqdist_design_matrix(x_range, y_range, z_range)
        self.labels = self.set_labels(noise_sigma)

    def make_random_design_matrix(self, x_range, y_range, z_range):
        X = np.zeros((self.N, 3))
        X[:, 0] = np.random.uniform(low=x_range[0],
                                    high=x_range[-1],
                                    size=self.N)
        X[:, 1] = np.random.uniform(low=y_range[0],
                                    high=y_range[-1],
                                    size=self.N)
        X[:, 2] = np.random.uniform(low=z_range[0],
                                    high=z_range[-1],
                                    size=self.N)
        return X

    def make_eqdist_design_matrix(self, x_range, y_range, z_range):
        X = np.zeros((self.N, 3))

        x = np.linspace(x_range[0], x_range[-1], self.Nx, endpoint=True)
        y = np.linspace(y_range[0], y_range[-1], self.Ny, endpoint=True)
        z = np.linspace(z_range[0], z_range[-1], self.Nz, endpoint=True)

        x_, y_, z_ = np.meshgrid(x, y, z)
        X[:, 0] = x_.flatten()
        X[:, 1] = y_.flatten()
        X[:, 2] = z_.flatten()
        return X

    def set_labels(self, noise_sigma):
        x = self.X[:, 0]
        y = self.X[:, 1]
        z = self.X[:, 2]

        func = x**2
        noise = np.random.normal(loc=0.0, scale=noise_sigma, size=self.N)
        return (func + noise).reshape(-1, 1)


class ResponseData():

    def __init__(self,
                 model: Model,
                 path: str = '../data/response_p.csv',
                 label: str = 'DE',
                 log_fit: bool = True):
        self.model = model
        self.data = pd.read_csv(path)
        self.label = label
        self.log_fit = log_fit

        self.x = self.data['Eg'].values
        self.x[self.x < 1.0e-12] = 1e-12
        self.y = self.data[self.label].values
        self.y[self.y < 1.0e-12] = 1e-12

    def cut(self, E_low=None, E_high=None):
        if E_low is not None:
            bool_arr = self.x > E_low
            self.x = self.x[bool_arr]
            self.y = self.y[bool_arr]
        if E_high is not None:
            bool_arr = self.x < E_high
            self.x = self.x[bool_arr]
            self.y = self.y[bool_arr]

    def create_design_matrix(self):
        X = np.zeros((self.N, self.model_order + 1))
        X[:, 0] = 1.0
        for i in range(1, self.model_order + 1):
            X[:, i] = (np.log10(self.x))**i

        self.X = X if self.bias_term else X[:, 1:]
        return self.X

    def split(self, test_size=0.3):
        X, Y = self.get_data()
        return train_test_split(X, Y,
                                test_size=test_size,
                                random_state=100)

    def get_data(self) -> tuple[Array, Array]:
        X = self.model.design_matrix(self.x)
        X[:, 1:] = self.standardize(X[:, 1:])
        return X, self.y

    @staticmethod
    def standardize(X_train, X_test=None):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        if X_test is None:
            return X_train
        else:
            X_test = scaler.transform(X_test)
            return X_train, X_test

    def plot(self, ax: Axes | None = None,
             log=True, **kwargs):
        if ax is None:
            _, ax = plt.subplots()

        kwargs |= {'s': 0.75}
        ax.scatter(self.x, self.y, **kwargs)
        ax.maybe_set_xlabel(r"$E_{\gamma}$")
        ax.maybe_set_ylabel(f"P({self.label})")
        if log:
            ax.set_xscale('log')
            ax.set_yscale('log')
        #ax.set_xlabel('$\mathrm{log}_{10} E_g$' if log_plot else '$E_g$')
        #ax.set_ylabel('$\mathrm{log}_{10}$' +
        #              self.labels_name if log_plot else self.labels_name)

        return ax

    @property
    def N(self) -> int:
        return len(self.x)
