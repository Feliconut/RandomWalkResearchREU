import scipy
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict


class BetaFitter:
    def __init__(self, data):
        self.data = OrderedDict(sorted(data.items()))
        self.x = np.asarray(list(self.data.keys()), dtype=np.float64)[1:-1]
        self.y = np.asarray(list(self.data.values()), dtype=np.float64)[1:-1]

        self.params = None
        self.params_cov = None

    @staticmethod
    def func_beta(x, a):
        return scipy.stats.beta.pdf(x, a, a)

    def fit(self, p0):
        self.params, self.params_cov, _, msg, _ = scipy.optimize.curve_fit(self.func_beta, self.x, self.y, p0,
                                                                           full_output=True)
        print(msg)

    def plot(self):
        plt.bar(self.x, self.y, width=0.01, color=['r'])
        plt.plot(self.x, self.func_beta(self.x, *self.params), label='fit')
        plt.legend(loc='best')
        plt.show()
