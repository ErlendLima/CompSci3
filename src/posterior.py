from data import ResponseData
from stubs import Array, Axes
import numpy as np
import matplotlib.pyplot as plt

try:
    from tqdm.autonotebook import tqdm
except ImportError:
    tqdm = lambda x: x


class Posterior:
    def __init__(self, data: ResponseData, prefix: str | None = None):
        prefix = data.prefix() if prefix is None else prefix
        self.samples = np.loadtxt(prefix + 'post_equal_weights.dat')
        self.data = data
        self.ypreds: Array = np.array([])
        mp, np_, lh = data.model.split_samples(self.samples)
        self.model_params = mp
        self.noise_params = np_
        self.likelihood = lh

    def parameter_mean(self) -> Array:
        return np.mean(self.model_params, axis=0)

    def parameter_best(self) -> Array:
        i = np.argmax(self.likelihood)
        return self.model_params[i]

    def predict(self, X: Array | None = None,
                parameters: Array | None = None) -> Array:
        if X is None:
            X, _ = self.data.get_data()
        if parameters is None:
            parameters = self.parameter_mean()
        print(parameters.shape)
        print(X.shape)
        return self.data.model(parameters, X)

    def sample(self, x: Array | None = None,
               design_matrix: Array | None = None,
               nsamples: int | None = None):
        x = self.data.x if x is None else x
        if design_matrix is None:
            design_matrix, _ = self.data.get_data()
        nsamples = len(self) if nsamples is None else nsamples
        if nsamples > len(self):
            raise ValueError(f"Too many samples. {nsamples} > {len(self)}")

        N = len(x)
        ypreds = np.zeros((nsamples, N))
        for i in tqdm(range(nsamples)):
            mu_i = self.data.model(self.model_params[i], design_matrix)
            ypreds[i, :] = mu_i
            #for j in range(N):
                #ypreds[i, j] = norm.rvs(loc=mu_i[j], scale=self.samples[i, -2], size=1)
                #ypreds[i, j] = norm.rvs(loc=mu_i[j], scale=x, size=1)
                #ypreds[i, j] = poisson.rvs(mu=mu_i[j], size=1)

        # Samples are expensive, hence saved
        self.ypreds = ypreds
        return ypreds

    def plot(self, x: Array | None = None,
             design_matrix: Array | None = None,
             ax: Axes | None = None,
             quantile: float | None = 0.05,
             **kwargs) -> Axes:
        if ax is None:
            fig, ax = plt.subplots()
        if x is None:
            x = self.data.x
        mean = 10**self.predict(design_matrix, self.parameter_best())
        kwargs = {'color': 'r'} | kwargs

        ax.plot(x, mean, **kwargs)
        if quantile is not None:
            if len(self.ypreds) < 1:
                print("Call sample() to plot quantiles.")
            else:
                low = np.quantile(10**self.ypreds, q=quantile/2, axis=0)
                high = np.quantile(10**self.ypreds, q=1 - quantile/2, axis=0)
                ax.fill_between(x, y1=low, y2=high, alpha=0.3)
        return ax

    def plot_samples(self, x: Array | None = None,
                     nsamples: int | None = None,
                     ax: Axes | None = None,
                     **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        if x is None:
            x = self.data.x
        nsamples = len(self) if nsamples is None else nsamples
        if nsamples > len(self):
            raise ValueError(f"Too many samples. {nsamples} > {len(self)}")

        kwargs = {'color': 'r', 'alpha': 0.1} | kwargs

        if len(self.ypreds) < 1:
            print("Call sample() to plot quantiles.")
        else:
            for y in self.ypreds[:nsamples]:
                ax.plot(x, 10**y, **kwargs)

        return ax

    def plot_pred

    def plot_mean

    def plot_ml

    def plot_map
    def __len__(self) -> int:
        return self.samples.shape[0]
