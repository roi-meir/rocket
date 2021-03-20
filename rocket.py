import random
from typing import List
from functools import partial

import pandas as pd
import numpy as np
import torch
from sklearn.linear_model import RidgeClassifier
from torch.nn import functional as F


class Rocket:
    def __init__(self,
                 input_length: int,
                 number_of_kernels=10_000,
                 possible_kernel_lengths=(7, 9, 11),
                 weights='normal',
                 centering='always',
                 bias='uniform',
                 dilation='exponential',
                 padding='random',
                 max_pooling=True,
                 ppv_pooling=True,
                 classifier=None
                 ):
        self.input_length = input_length
        self.number_of_kernels = number_of_kernels
        self.possible_kernel_lengths = possible_kernel_lengths
        self.kernel_lengths = random.choices(possible_kernel_lengths, k=self.number_of_kernels)

        self.weights = weights
        if self.weights == 'normal':
            self.weights_distribution = partial(np.random.normal, loc=0, scale=1)
        elif self.weights == 'uniform':
            self.weights_distribution = partial(np.random.uniform, low=-1, high=1)
        elif self.weights == 'integer':
            self.weights_distribution = partial(np.random.choice, (-1, 0, 1))
        else:
            ValueError(f"Unrecognized weights option {self.weights}")

        self.centering = centering
        if centering == 'always':
            self.centering_distribution = partial(np.random.choice, (True,))
        elif centering == 'never':
            self.centering_distribution = partial(np.random.choice, (False,))
        elif centering == 'random':
            self.centering_distribution = partial(np.random.choice, (False, True))
        else:
            ValueError(f"Unrecognized centering option {self.centering}")

        self.bias = bias
        if self.bias == 'normal':
            self.bias_distribution = partial(np.random.normal, loc=0, scale=1)
        elif self.bias == 'uniform':
            self.bias_distribution = partial(np.random.uniform, low=-1, high=1)
        elif self.bias == 'zero':
            self.bias_distribution = partial(np.random.choice, (0,))
        else:
            ValueError(f"Unrecognized bias option {self.bias}")

        self.dilation = dilation

        self.padding = padding
        if padding == 'always':
            self.padding_distribution = partial(np.random.choice, (True,))
        elif padding == 'never':
            self.padding_distribution = partial(np.random.choice, (False,))
        elif padding == 'random':
            self.padding_distribution = partial(np.random.choice, (False, True))
        else:
            ValueError(f"Unrecognized padding option {self.padding}")

        self.max_pooling = max_pooling
        self.ppv_pooling = ppv_pooling

        if classifier is None:
            classifier = RidgeClassifier()

        self.classifier = classifier

        self.kernels = self._sample_kernels()

    def fit(self, X, y):
        X = self.apply_kernels(X)
        return self.classifier.fit(X, y)

    def predict(self, X, y=None):
        X = self.apply_kernels(X)
        return self.classifier.predict(X)

    def score(self, X, y):
        X = self.apply_kernels(X)
        return self.classifier.score(X, y)

    def apply_kernels(self, X):
        if isinstance(X, pd.DataFrame):
            X = torch.from_numpy(X.values)
        elif isinstance(X, np.ndarray):
            X = torch.from_numpy(X)

        X = X.view(X.shape[0], 1, X.shape[-1])

        features = []
        for kernel in self.kernels:
            max_pool, ppv = kernel.apply_kernel(X)
            if self.max_pooling:
                features.append(max_pool)

            if self.ppv_pooling:
                features.append(ppv)

        features = torch.stack(features, -1).view(X.shape[0], len(features))
        return features.detach().cpu().numpy()

    def _sample_kernels(self) -> List:
        kernels = []
        for i in range(self.number_of_kernels):
            kernel_length = self.kernel_lengths[i]

            weights = self.weights_distribution(size=kernel_length)
            should_center = self.centering_distribution()
            if should_center:
                weights = weights - weights.mean()

            if self.dilation == 'exponential':
                exponent = np.random.uniform(0, np.log2((self.input_length - 1) / (kernel_length - 1)))
                dilation = 2 ** exponent
                dilation = np.int32(dilation)
            elif self.dilation == 'uniform':
                dilation = np.random.uniform(1, (self.input_length - 1) / (kernel_length - 1))
                dilation = np.int32(dilation)
            elif isinstance(self.dilation, int):
                dilation = self.dilation
            else:
                raise ValueError(f"Unknown option to dilation {self.dilation}")

            should_pad = self.padding_distribution()
            if should_pad:
                padding = ((kernel_length - 1) * dilation)
            else:
                padding = 0

            kernels.append(Kernel(weights,
                           bias=self.bias_distribution(),
                           padding=padding,
                           dilation=dilation))

        return kernels


class Kernel:
    def __init__(self,
                 weights,
                 bias,
                 padding,
                 dilation):

        if isinstance(weights, np.ndarray):
            weights = torch.from_numpy(weights)

        if isinstance(bias, np.ndarray):
            bias = torch.from_numpy(bias)
        elif isinstance(bias, (float, int)):
            bias = torch.tensor([bias])

        self.weights = weights.view(1, 1, len(weights))
        self.bias = bias
        self.padding = padding
        self.dilation = dilation

    def apply_kernel(self, X):
        output = F.conv1d(input=X, weight=self.weights, bias=self.bias,
                          padding=self.padding, dilation=self.dilation)

        m, _ = output.max(dim=-1)
        ppv = (output > 0).sum(axis=-1) / output.shape[-1]

        return m, ppv

    def __repr__(self):
        return f"Kernel with length of {self.weights.shape[-1]} and dilation {self.dilation}"

    def __str__(self):
        return repr(self)
