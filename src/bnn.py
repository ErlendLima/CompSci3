import torch
from torch.nn import Module, MSELoss, ModuleList
from torch.functional import F
from torch.optim import Adam
import numpy as np
from typing import Iterable, Optional
from bayesian_layer import BayesianLayer
from data import SynteticData
import matplotlib.pyplot as plt

class BayesianNet(Module):
    def __init__(self,
                 num_inputs: int,
                 num_outputs: int,
                 hidden: Iterable[int],
                 activation = F.relu,
                 prior_mean: float = 0.0,
                 prior_std: float = 0.1):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden = len(hidden)
        self.activation = activation

        self.inlayer = BayesianLayer(self.num_inputs, hidden[0])
        self.outlayer = BayesianLayer(hidden[-1], self.num_outputs)

        if self.num_hidden > 1:
            self.hlayers = ModuleList([BayesianLayer(hidden[i], hidden[i+1], prior_mean, prior_std) \
                                  for i in range(0, self.num_hidden-1)])

    def forward(self, X):
        X = self.activation(self.inlayer(X))
        if self.num_hidden > 1:
            for hl in self.hlayers:
                X = self.activation(hl(X))
        return self.outlayer(X)

    def kl_loss_function(self):
        kl_loss_total = 0
        n = 0
        for module in self.modules():
            if isinstance(module, (BayesianLayer)):
                kl_loss_total += module.kl_loss 
        return kl_loss_total

    def train_single_step(self, opt, X, ytrue, kl_factor):
        mse = MSELoss()
        yhat = self(X)
        kl_loss = self.kl_loss_function()
        loss =  mse(yhat, ytrue) + kl_factor*(kl_loss/X.shape[0])
        opt.zero_grad()
        loss.backward()
        opt.step()
        return loss

    def make_predictions(self,
              X,
              samples = 1000):
        preds = np.asarray([self.forward(X).data.numpy().flatten() for i in range(samples)])
        means = np.mean(preds, axis=0)
        ql = np.quantile(preds, q=0.025, axis=0)
        qu = np.quantile(preds, q=0.975, axis=0)
        return preds, means, ql, qu

if __name__ == '__main__':
    ds = SynteticData(N=1000, noise_sigma=0.1, grid_type='random')
    X = torch.from_numpy(ds.X).float()
    labels = torch.from_numpy(ds.labels).float()

    num_inputs = 3
    num_outputs = 1
    hidden = [100, 25]
    bnn = BayesianNet(num_inputs, num_outputs, hidden, prior_mean=0.0, prior_std=1.0)
    opt = Adam(bnn.parameters(), lr=0.01)

    for epoch in range(5000):
        loss = bnn.train_single_step(opt, X, labels, kl_factor=0.01)
        if (epoch+1) % 100 == 0:
            print(epoch+1, f'Loss: {loss.item():.8f}')

    X_test = torch.from_numpy(ds.X).float()
    labels_test = torch.from_numpy(ds.labels).float()

    preds, means, ql, qu = bnn.make_predictions(X_test)
    x_test = X_test[:, 0]
    p = x_test.argsort()

    plt.title('BNN solution')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$f(x)$')
    plt.scatter(x_test[p], labels_test.data.numpy().flatten()[p], color='k', s=2)
    plt.plot(x_test[p], means[p], 'r', linewidth=1, label='mean')
    plt.fill_between(x_test.flatten()[p], y1=ql.flatten()[p], y2=qu.flatten()[p], 
                    color='blue', alpha=0.5)
    plt.show()


