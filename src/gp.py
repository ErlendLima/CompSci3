from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from data import DataSet
import matplotlib.pyplot as plt


fixed_noise = True
gaussian_noise_sigma=0.1
optimizer_runs = 1

if fixed_noise:
    data_noise_variance = WhiteKernel(noise_level=gaussian_noise_sigma**2, 
                                    noise_level_bounds='fixed')
else:
    data_noise_variance = WhiteKernel(noise_level=gaussian_noise_sigma**2)

kernel = RBF()
gp = GaussianProcessRegressor(kernel=kernel + data_noise_variance, 
                                    n_restarts_optimizer=optimizer_runs-1)


ds = DataSet(N=1000, noise_sigma=0.1, grid_type='random')
X = ds.X
labels = ds.labels

gp.fit(X, labels)

means, stds = gp.predict(X, return_std=True)

# ql = 0.025 quantile
# qu = 0.975 quantile 
# 1.96 due to the posterior being normal
ql = means-1.96*stds
qu = means+1.96*stds

x_test = X[:, 0]
p = x_test.argsort()

plt.title('GP solution')
plt.xlabel(r'$x$')
plt.ylabel(r'$f(x)$')
plt.scatter(x_test[p], labels[p], color='k', s=2) 
plt.plot(x_test[p], means[p], 'r', linewidth=1, label='mean')
plt.fill_between(x_test[p], y1=ql[p], y2=qu[p], color='blue', alpha=0.5)

plt.show()


                










