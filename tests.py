import numpy as np
import scipy as sp


def calc_jump_test(path, chunk_size):
    path = np.asarray(path)
    ind_limit = len(path) // chunk_size - 1

    x1 = path[np.arrange(0, ind_limit - chunk_size, chunk_size)]
    x2 = path[np.arrange(chunk_size, ind_limit, chunk_size)]
    statistic = np.sum(np.square(x2 - x1))

    return statistic


def jump_test(statistics, chunk_size, length):
    n = length // chunk_size

    return sp.stats.kstest(statistics, sp.stats.norm.cdf, n * chunk_size, scale=len(statistics))


def arcsine_test(statistics):
    return sp.stats.kstest(statistics, sp.stats.arcsine.cdf, N=len(statistics))
