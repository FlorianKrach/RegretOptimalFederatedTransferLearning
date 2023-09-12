"""
author: Florian Krach
"""

import math
import time

import numpy as np
import matplotlib.pyplot as plt
from fbm import FBM
import scipy.special as scispe
import torch

import joblib


# ==============================================================================
# Global variables

ACTIVATION_FUNCTIONS = {
  "relu": torch.nn.ReLU(),
  "sigmoid": torch.nn.Sigmoid(),
  "tanh": torch.nn.Tanh(),
  "leaky_relu": torch.nn.LeakyReLU(),

}




# ==============================================================================
def init_weights(m):  # initialize weights for model for linear NN
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            try:
                torch.nn.init.xavier_uniform_(m.bias)
            except Exception:
                torch.nn.init.uniform_(m.bias, -0.1, 0.1)


class RandomizedNNDataset:
    def __init__(self, n_samples, n_features, input_size, activation, **kwargs):
        self.n_samples = n_samples
        self.n_features = n_features
        self.input_size = input_size
        layers = [
            torch.nn.Linear(input_size, n_features, bias=True),
            ACTIVATION_FUNCTIONS[activation],
            torch.nn.Linear(n_features, 1, bias=True),
        ]
        self.randNN = torch.nn.Sequential(*layers)
        self.randNN.apply(init_weights)

    def gen_samples(self, n_samples=None):
        if n_samples is None:
            n_samples = self.n_samples
        X = torch.randn(n_samples, self.input_size)
        Y = self.randNN(X)
        samples = (X.detach().numpy(), Y.detach().numpy())
        return samples



# ==============================================================================
DATA_GENERATORS = {
    "RandomizedNNDataset": RandomizedNNDataset,
}



if __name__ == '__main__':
    pass
