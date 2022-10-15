import numpy as np
import scipy as sp


def calc_jump_test(path, n):
    length = len(path)
    path = np.asarray(path)
    chunk_size = int(length / n)
    ind_limit = length - (length % n) - 1

    x1 = path[np.arrange(0, ind_limit - chunk_size, chunk_size)]
    x2 = path[np.arrange(chunk_size, ind_limit, chunk_size)]
    statistic = np.sum(np.square(x2 - x1))

    return statistic


def jump_test(statistics, n):
    return sp.stats.kstest(statistics, sp.stats.norm.cdf, (1 / (n - 1), 3 / (n - 1)), scale=len(statistics))


def arcsine_test(data):
    return sp.stats.kstest(data, sp.stats.arcsine.cdf, N=len(data))
