import scipy
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from collections import Counter


class BetaFitter:
    def __init__(self, data):
        self.og_data = data
        self.data = OrderedDict(sorted(Counter(data).items()))
        self.x = np.asarray(list(self.data.keys()), dtype=np.float32)
        self.y = np.cumsum(np.asarray(list(self.data.values()), dtype=np.float32) / len(data))

        if self.x[0] == 0:
            self.y[0] = 0

        if self.x[-1] == 1:
            self.y[-1] = 1

        self.output = None
        self.params = None
        self.params_cov = None
        self.last_guess = None

    @staticmethod
    def func_beta(x, a):
        return scipy.stats.beta.cdf(x, a, a)

    def func_beta_residuals(self, a):
        return self.y - self.func_beta(self.x, a)

    def fit(self, p0, method):
        self.last_guess = p0[0]
        self.output = scipy.optimize.least_squares(self.func_beta_residuals, p0, method=method)
        self.params = self.output.x
        print(self.params)
        print(self.output.message)

    def plot(self, bins):
        plt.plot(self.x, self.y)
        plt.plot(self.x, self.func_beta(self.x, *self.params), label='fit')
        plt.plot(self.x, self.func_beta(self.x, self.last_guess), label='guess')
        plt.legend(loc='best')
