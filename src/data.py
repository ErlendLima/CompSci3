import numpy as np
import torch
from typing import Iterable

class DataSet():
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
        X[:, 0] = np.random.uniform(low=x_range[0], high=x_range[-1], size=self.N)
        X[:, 1] = np.random.uniform(low=y_range[0], high=y_range[-1], size=self.N)
        X[:, 2] = np.random.uniform(low=z_range[0], high=z_range[-1], size=self.N)
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



        


