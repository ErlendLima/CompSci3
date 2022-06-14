from locale import MON_1
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
from scipy.stats import norm, invgamma, lognorm, multivariate_normal
from pymultinest.solve import solve
import corner

df = pd.read_csv('../data/response.csv')
x = df.iloc[:, 1].values
y = df.iloc[:, 6].values
N = len(x)

plt.figure()
plt.scatter(np.log(x), np.log(y), s=0.5)
plt.xlabel('$E_g$')
plt.ylabel('DE')
prefix = 'tests/test1_'

def model_mean(params):
    order = len(params)-2
    #log_sum = params[0]*np.ones(N)
    sum_ = params[0]*np.ones(N)
    for i in range(1, order+1):
        #log_sum += params[i]*(np.log(x))**i
        sum_ += params[i]*(x**i)

    #return np.exp(log_sum)
    return sum_

def run_multinest(parameters):
    ndim = len(parameters)
    def prior_transform(cube):
        params = cube.copy()
        
        #params[0] = 0.2*cube[0] - 0.1
        #params[1] = 0.2*cube[0] - 0.1
        #params[2] = 0.2*cube[0] - 0.1
        #params[3] = 0.2*cube[0] - 0.1
        #params[4] = 0.2*cube[0] - 0.1

        return params

    def loglike(params):
        mu = model_mean(params)
        loglike = -0.5*(((y-mu)/params[-1])**2).sum()
        if not np.isfinite(loglike):
            loglike = -1e12
        return loglike


    results = solve(LogLikelihood=loglike, Prior=prior_transform, n_dims=ndim,
        n_live_points=1000, outputfiles_basename=prefix, verbose=True, resume=False)
    json.dump(parameters, open(prefix + 'params.json', 'w'))
    print(results)
    return results




class Inference():
    def __init__(self,
                 path_post_equal_weights,
                 ):
        self.post_samples = np.loadtxt(path_post_equal_weights)
        self.nsamples = self.post_samples.shape[0]
        

    def posterior_predictions(self):
        ypreds = np.zeros((self.nsamples, N))
        for i in range(self.nsamples):
            mu_i = model_mean(self.post_samples[i])
            cov = (self.post_samples[i,-2]**2)*np.identity(N)
            ypreds[i] = multivariate_normal(mean=mu_i, cov=cov)

        return ypreds

    


order = 3
parameters = [f'C{i}' for i in range(order+1)] + ['sigma']
results = run_multinest(parameters)
corner.corner(results['samples'], labels=parameters)

post_inf = Inference('tests/test1_post_equal_weights.dat')



plt.show()
