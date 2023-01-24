import scipy
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from collections import Counter


class BetaFitter:
    def __init__(self, data):
        self.og_data = data
        self.data = OrderedDict(sorted(Counter(data).items()))
        self.x = np.asarray(list(self.data.keys()), dtype=np.float32)[1:-1]
        self.y = np.asarray(list(self.data.values()), dtype=np.float32)[1:-1]

        self.params = None
        self.params_cov = None

    @staticmethod
    def func_beta(x, a):
        return scipy.stats.beta.pdf(x, a, a)

    def fit(self, p0, method):
        self.params, self.params_cov = scipy.optimize.curve_fit(self.func_beta, self.x, self.y, p0,
                                                                method=method)
        print(self.params)

    def plot(self, bins):
        plt.hist(self.og_data, bins=bins, density=True)
        plt.plot(self.x, self.func_beta(self.x, *self.params), label='fit')
        plt.legend(loc='best')
        plt.show()
