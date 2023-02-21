import walk
import scipy
import numpy as np
import pandas as pd
import multiprocessing
import matplotlib.pyplot as plt
from math import pi
from collections import OrderedDict
from collections import Counter
from experiment import MultipleExperiment


def wc(x, a):
    return scipy.stats.wrapcauchy.cdf(x, a, scale=0.5 / pi)


def beta(x, a):
    return scipy.stats.beta.cdf(x, a, a)


class MultiFitter:
    def __init__(self, data):
        self.distributions = {
            'beta': beta,
            'wc': wc,
        }

        self.og_data = data
        self.data = OrderedDict(sorted(Counter(data).items()))
        self.x = np.asarray(list(self.data.keys()), dtype=np.float32)
        self.y = np.cumsum(np.asarray(list(self.data.values()), dtype=np.float32) / len(data))

        if self.x[0] == 0:
            self.y[0] = 0

        if self.x[-1] == 1:
            self.y[-1] = 1

        self.current_cdf = None
        self.output = {}
        self.params = {}

    def calc_residuals(self, a):
        return self.y - self.current_cdf(self.x, a)

    def fit(self, method):
        for key, dist in self.distributions.items():
            self.current_cdf = dist
            self.output[key] = scipy.optimize.least_squares(self.calc_residuals, 0.5, method=method)
            self.params[key] = self.output[key].x[0]

    def plot(self):
        plt.plot(self.x, self.y)

        for key, dist in self.distributions.items():
            plt.plot(self.x, dist(self.x, self.params[key]), label=key)

        plt.legend()


def find_beta(alpha):
    def wf(n):
        return 1 / (1 + n) ** alpha

    exp = MultipleExperiment(walk.SelfInteractingRandomWalk, n_trials=5000, length=1000, weight_function=wf)
    exp.run(store_data=False)
    fitter = MultiFitter(exp.stats['ta0'])
    fitter.fit('lm')
    print(f'Run with Alpha: {alpha} completed.')
    return alpha, fitter.params['beta']


if __name__ == '__main__':
    alphas = np.linspace(0, 20, 500)
    pool = multiprocessing.Pool(processes=12)
    output = [0] * len(alphas)

    for i, alpha in enumerate(alphas):
        output[i] = (pool.apply_async(find_beta, args=[alpha]))

    pool.close()
    pool.join()

    output = [list(p.get()) for p in output]

    df = pd.DataFrame(output, columns=['alpha', 'beta'])
    df.to_csv('betas.csv', index=False)
