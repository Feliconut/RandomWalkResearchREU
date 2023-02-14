import walk
import scipy
import numpy as np
import pandas as pd
import multiprocessing
import matplotlib.pyplot as plt
from collections import OrderedDict
from collections import Counter
from experiment import MultipleExperiment


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

    def plot(self, bins):
        plt.plot(self.x, self.y)
        plt.plot(self.x, self.func_beta(self.x, *self.params), label='fit')
        plt.plot(self.x, self.func_beta(self.x, self.last_guess), label='guess')
        plt.legend(loc='best')


def find_beta(alpha):
    def wf(n):
        return 1 / (1 + n) ** alpha

    exp = MultipleExperiment(walk.SelfInteractingRandomWalk, n_trials=5000, length=1000, weight_function=wf)
    exp.run(store_data=False)
    fitter = BetaFitter(exp.stats['ta0'])
    fitter.fit([0.5], 'lm')
    print(f'Run with Alpha: {alpha} completed.')
    return alpha, fitter.params[0]


if __name__ == '__main__':
    alphas = np.linspace(1, 10, 100)
    pool = multiprocessing.Pool(processes=10)
    output = [0] * len(alphas)

    for i, alpha in enumerate(alphas):
        output[i] = (pool.apply_async(find_beta, args=[alpha]))

    pool.close()
    pool.join()

    output = [list(p.get()) for p in output]

    df = pd.DataFrame(output, columns=['alpha', 'beta'])
    df.to_csv('betas.csv', index=False)
