from tkinter.tix import X_REGION
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
from scipy.stats import norm, invgamma, lognorm, multivariate_normal
from pymultinest.solve import solve
import corner
from data import ResponseData
from pathlib import Path
from stubs import Array, Axes

try:
    from tqdm.autonotebook import tqdm
except ImportError:
    tqdm = lambda x: x


def model_mean(params, X):
    mu = X @ np.reshape(params[:-1], (-1, 1))
    return mu.flatten()

def run_multinest(parameters, X, y, prefix, verbose=False, **kwargs):
    ndim = len(parameters)
    def prior_transform(cube):
        params = cube.copy()
        params[0] = 200*cube[0] - 100
        params[1] = 200*cube[1] - 100
        params[2] = 200*cube[2] - 100
        #params[3] = 200*cube[3] - 100
        params[-1] = 500*cube[-1]

        return params

    def loglike(params):
        mu = model_mean(params, X)
        loglike = -0.5*(((np.log10(y) - mu)/params[-1])**2).sum()
        if not np.isfinite(loglike):
            loglike = -1e12
        return loglike

    results = solve(LogLikelihood=loglike, Prior=prior_transform, n_dims=ndim,
                    n_live_points=1000, outputfiles_basename=prefix, resume=False,
                    verbose=verbose, **kwargs)
    json.dump(parameters, open(prefix + 'params.json', 'w'))
    return results


def posterior_predictions(path_post_equal_weights, X, N):
    post_samples = np.loadtxt(path_post_equal_weights)
    nsamples = post_samples.shape[0]
    ypreds = np.zeros((nsamples, N))

    for i in tqdm(range(nsamples)):
        mu_i = model_mean(post_samples[i, :-1], X)
        for j in range(N):
            ypreds[i, j] = norm.rvs(loc=mu_i[j], scale=post_samples[i, -2], size=1)

    ypreds_mean = X @ np.reshape(np.mean(post_samples[:, :-2], axis=0), (-1, 1))
    return ypreds, ypreds_mean.flatten()


if __name__ == '__main__':
    prefix = 'tests/test2_'

    model_order = 2
    data = ResponseData(model_order=model_order)
    data.cut_data(E_low=2000)
    data.create_design_matrix()
    X = data.X
    y = data.y

    X[:, 1:] = data.standardize(X[:, 1:])
    ax = data.plot_data()

    parameters = [f'C{i}' for i in range(model_order+1)] + ['sigma']
    results = run_multinest(parameters, X, y, prefix=prefix)
    corner.corner(results['samples'], labels=parameters)

    ypreds, ypreds_mean = posterior_predictions(prefix+'post_equal_weights.dat', X, data.N)
    ax.plot(np.log10(data.x), ypreds_mean, color='tab:red')
    plt.show()
