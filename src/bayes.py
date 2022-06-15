from stubs import Array, Axes
import numpy as np
import matplotlib.pyplot as plt
from model import Model
from pymultinest.solve import solve
import json
from pathlib import Path


def run_multinest(model: Model,
                  X,
                  y,
                  prefix: str | None = None,
                  verbose: bool = False,
                  **kwargs):
    parameters = model.parameters()
    ndim = len(parameters)
    prefix = model.prefix() if prefix is None else prefix
    path, _ = prefix.rsplit('/')
    Path(path).mkdir(exist_ok=True)

    Y = np.log10(y)
    def loglike(params: Array) -> float:
        mu = model.mean(params, X)
        loglike = -0.5 * (((Y - mu) / params[-1])**2).sum()
        if not np.isfinite(loglike):
            loglike = -1e12
        return loglike

    results = solve(LogLikelihood=loglike,
                    Prior=model.prior_transform,
                    n_dims=ndim,
                    n_live_points=1000,
                    outputfiles_basename=prefix,
                    resume=False,
                    verbose=verbose,
                    **kwargs)

    with open(prefix + 'params.json', 'w') as outfile:
        json.dump(parameters, outfile)

    return results


class Posterior:
    def __init__(self, model: Model, prefix: str | None = None):
        prefix = model.prefix if prefix is None else prefix
        self.samples = np.loadtxt(prefix + 'post_equal_weights.dat')

    def predict(self, X: Array, N: int):
        #ypreds = np.zeros((len(self), N))
        ypreds_mean = X @ np.reshape(np.mean(post_samples[:, :-2], axis=0), (-1, 1))

    def scatter(self, ax: Axes | None = None) -> Axes:
        if ax is None:
            fig, ax = plt.subplots()
        ax.scatter()
        return ax

    def __len__(self) -> int:
        return self.samples.shape[0]


if __name__ == '__main__':
    from model import LogpolyModel
    from data import ResponseData
    from corner import corner
    from bayesian_regression import posterior_predictions

    model = LogpolyModel(order=2)
    data = ResponseData(model)
    data.cut(E_low=2000, E_high=10000)
    data.plot(log=False)
    X, y = data.get_data()

    results = run_multinest(model, X, y)
    corner.corner(results['samples'], labels=model.parameters())

    ypreds, ypreds_mean = posterior_predictions(model.prefix()+'post_equal_weights.dat', X, data.N)

    ax = data.plot()
    ax.plot(data.x, 10**ypreds_mean, color='tab:red')
    plt.show()
