import json
import sys
import numpy as np
from scipy.stats import multivariate_normal
import pymultinest
import matplotlib.pyplot as plt
from data import DataSet
from sklearn.preprocessing import PolynomialFeatures




def run_multinest(X, labels, parameters, n_params):

    def prior(cube, ndim, nparams):
        pass

    def loglike(cube, ndim, nparams): 
        return -0.5*((cube@X - labels)**2 / cube[-1]**2).sum()

    # run MultiNest
    pymultinest.run(loglike, prior, n_params, outputfiles_basename='test', resume = False, verbose = True)
    json.dump(parameters, open('test_params.json', 'w')) # save parameter names

ds = DataSet(N=1000, noise_sigma=0.1, grid_type='random')
X = ds.X
labels = ds.labels

poly = PolynomialFeatures(2, include_bias=True)
X_ = poly.fit_transform(X)

parameters = [f'beta {i}' for i in range(X_.shape[1])] + ['sigma']
n_params = len(parameters)
run_multinest(X_, labels, parameters[:-1], n_params)


