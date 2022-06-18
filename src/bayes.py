from stubs import Array, Axes
import numpy as np
import matplotlib.pyplot as plt
from model import Model
from pymultinest.solve import solve
import json
from pathlib import Path
from data import Data, ResponseData
from scipy.stats import norm, invgamma, lognorm, multivariate_normal, poisson

try:
    from tqdm.autonotebook import tqdm
except ImportError:
    tqdm = lambda x: x


def run_multinest(data: ResponseData,
                  prefix: str | None = None,
                  verbose: bool = False,
                  **kwargs):
    parameters = data.model.parameters()
    ndim = len(parameters)
    prefix = data.prefix() if prefix is None else prefix
    path, _ = prefix.rsplit('/')
    Path(path).mkdir(exist_ok=True)

    loglike = data.likelihoodfn()
    kwargs = {'n_live_points': 1000} | kwargs

    results = solve(LogLikelihood=loglike,
                    Prior=data.model.prior_transform,
                    n_dims=ndim,
                    outputfiles_basename=prefix,
                    resume=False,
                    verbose=verbose,
                    **kwargs)

    with open(prefix + 'params.json', 'w') as outfile:
        json.dump(parameters, outfile)

    return results
