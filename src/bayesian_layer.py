import torch
import math
from torch.nn import Module, Parameter
from torch.functional import F
import numpy as np

class BayesianLayer(Module):
    def __init__(self,
                 num_inputs: int,
                 num_outputs: int,
                 prior_mean: float=0.0,
                 prior_std: float=0.1):
        super().__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.prior_mean = prior_mean
        self.prior_log_std = math.log(prior_std)

        self.weight_mean = Parameter(torch.Tensor(num_outputs, num_inputs))
        self.weight_log_std = Parameter(torch.Tensor(num_outputs, num_inputs))

        self.bias_mean = Parameter(torch.Tensor(num_outputs))
        self.bias_log_std = Parameter(torch.Tensor(num_outputs))

        self.reset_parameters()
    
    def reset_parameters(self):
        # Initialization method of Adv-BNN
        stdv = 1. / math.sqrt(self.weight_mean.size(1))
        self.weight_mean.data.uniform_(-stdv, stdv)
        self.weight_log_std.data.fill_(self.prior_log_std)
       
        self.bias_mean.data.uniform_(-stdv, stdv)
        self.bias_log_std.data.fill_(self.prior_log_std)


    def forward(self, X):
        w = self.weight_mean + torch.exp(self.weight_log_std) * torch.randn_like(self.weight_log_std)
        b = self.bias_mean + torch.exp(self.bias_log_std) * torch.randn_like(self.bias_log_std)

        self.kl_loss = BayesianLayer.kl_divergence(self.weight_mean, self.weight_log_std,
                               self.prior_mean, self.prior_log_std) \
                     + BayesianLayer.kl_divergence(self.bias_mean, self.bias_log_std,
                               self.prior_mean, self.prior_log_std)
    
        return F.linear(X, w, b)

    @staticmethod
    def kl_divergence(post_mean, post_log_std, prior_mean, prior_log_std):
        kl = prior_log_std - post_log_std \
        + (torch.exp(post_log_std)**2 + (post_mean - prior_mean)**2) / (2*math.exp(prior_log_std)**2) \
        - 0.5
        return kl.sum()

if __name__ == '__main__':
    blayer = BayesianLayer(10)

        