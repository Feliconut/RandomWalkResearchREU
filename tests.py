import numpy as np
import scipy as sp


def calc_jump_test(path, chunk_size):
    path = np.asarray(path)
    ind_limit = (len(path) // chunk_size) * chunk_size

    x1 = path[np.arange(0, ind_limit - chunk_size, chunk_size)]
    x2 = path[np.arange(chunk_size, ind_limit, chunk_size)]
    statistic = np.sum(np.square(x2 - x1))

    return statistic


def jump_test(statistics, chunk_size, length):
    n = length // chunk_size
    norm_stats = (np.asarray(statistics) - (n * chunk_size)) / (2 * n * (chunk_size ** 2))

    return sp.stats.normaltest(norm_stats)


def arcsine_test(statistics):
    return sp.stats.kstest(statistics, sp.stats.arcsine.cdf, N=len(statistics))
