from tkinter.tix import X_REGION
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
from scipy.stats import norm, invgamma, lognorm, multivariate_normal
from pymultinest.solve import solve
import corner
from data import ResponseData


def model_mean(params, X):
    mu = X @ np.reshape(params[:-1], (-1, 1))
    return mu.flatten()
def run_multinest(parameters, X, y):
    ndim = len(parameters)
    def prior_transform(cube):
        params = cube.copy()
        
        params[0] = 200*cube[0] - 100
        params[1] = 200*cube[1] - 100
        params[2] = 200*cube[2] - 100
        params[3] = 200*cube[3] - 100
        params[4] = 1*cube[4]

        return params

    def loglike(params):
        mu = model_mean(params, X)
        loglike = -0.5*(((np.log10(y) - mu)/params[-1])**2).sum()
        if not np.isfinite(loglike):
            loglike = -1e12
        return loglike

    results = solve(LogLikelihood=loglike, Prior=prior_transform, n_dims=ndim,
        n_live_points=1000, outputfiles_basename=prefix, verbose=True, resume=False)
    json.dump(parameters, open(prefix + 'params.json', 'w'))
    print(results)
    return results

def posterior_predictions(path_post_equal_weights, X, N):
    post_samples = np.loadtxt(path_post_equal_weights)
    nsamples = post_samples.shape[0]
    ypreds = np.zeros((nsamples, N))
    for i in range(nsamples):
        print(i)
        mu_i = model_mean(post_samples[i, :-1], X)
        #cov_i = (post_samples[i, -2]**2)*np.identity(N)
        for j in range(N):
            ypreds[i, j] = norm.rvs(loc=mu_i[j], scale=post_samples[i, -2], size=1)
    return ypreds


if __name__ == '__main__':
    prefix = 'tests/test1_'

    model_order = 3
    data = ResponseData(model_order=model_order)
    data.cut_data(E_low=2000)
    data.create_design_matrix()
    X = data.X
    y = data.y

    X[:, 1:] = data.standardize(X[:, 1:])
    ax = data.plot_data()

    parameters = [f'C{i}' for i in range(model_order+1)] + ['sigma']
    results = run_multinest(parameters, X, y)
    corner.corner(results['samples'], labels=parameters)

    ypreds = posterior_predictions('tests/test1_post_equal_weights.dat', X, data.N)
    ypreds_mean = np.mean(ypreds, axis=0)
    ax.plot(np.log10(data.x), ypreds_mean)
    plt.show()
