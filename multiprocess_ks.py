import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


rng = np.random.default_rng()
def sample_from_ssrw(log_n, log_length):
    n, length = int(10**log_n), int(10**log_length)
    samples = list()
    if length > 1e8: # do one sample at a time
        for _ in range(n):
            length_remain = length
            where = 0
            while length_remain > 1e8:
                where += (rng.integers(2, size=(int(1e8))).sum()*2-int(1e8))
                length_remain -= 1e8
            where += (rng.integers(2, size=(int(length_remain))).sum()*2-int(length_remain))
            samples.append(where)
    else: # do multiple samples at a time
        n_remain = n
        n_each_step = int(10**(8 - log_length))
        while n_remain > n_each_step:
            samples.extend(rng.integers(2, size=(n_each_step, length)).sum(axis=1)* 2 - length)
            n_remain -= n_each_step
        samples.extend(rng.integers(2, size=(n_remain, length)).sum(axis=1)* 2 - length)
    return np.array(samples) / length**0.5

def piecewise_linearized_normal_cdf(x, dx):
    # linear between data points
    # decompose: x = q * dx + r
    # q = x // dx
    # r = x - q * dx
    # # linear interpolation of normal.cdf between q*dx and (q+1)*dx
    # return (1- r / dx) * stats.norm.cdf(q * dx) + (r / dx) * stats.norm.cdf((q + 1) * dx)

    # linear at data points
    # decompose: x = (q + 1/2) * dx + r
    q = (x - dx / 2) // dx
    r = x - (q + 1/2) * dx
    # linear interpolation of normal.cdf between q*dx and (q+1)*dx
    return (1- r / dx) * stats.norm.cdf((q+1/2) * dx) + (r / dx) * stats.norm.cdf((q + 1 + 1/2) * dx)
    

def adapted_ks_pvalue(log_n, log_length):
    samples = sample_from_ssrw(log_n, log_length)
    
    piecewise_cdf = lambda x: piecewise_linearized_normal_cdf(x, 2/10**(0.5*log_length)) # resolution formulae; see note

    # ks test
    return stats.kstest(samples, piecewise_cdf)[1]


def test_pvalue(log_n, log_length):
    print("n = 10^", log_n, ", length = 10^", log_length)
    samples = sample_from_ssrw(log_n, log_length)

    resolution = 2/10**(0.5*log_length) # resolution formulae; see note
    piecewise_cdf = lambda x: piecewise_linearized_normal_cdf(x, resolution) 

    # add random noise to samples
    samples_noised = samples + rng.uniform(-0.5* resolution, 0.5*resolution, size=samples.shape)

    # ks test
    stat, p =  stats.kstest(samples_noised, piecewise_cdf)
    # print(f"stat = {stat}, p = {p}")
    return p, samples, samples_noised, piecewise_cdf
# p, samples, samples_noised, piecewise_cdf = test_pvalue(log_n = 1,log_length = 0.8)

x = np.linspace(1, 5,9)
y = np.linspace(1, 5,9)
def pvalue(A):
        return test_pvalue(A[0], A[1])[0]

if __name__ == '__main__':
    # freeze_support()

    from multiprocessing import Pool
    x = np.linspace(1, 6,11)
    y = np.linspace(1, 6,11)
    xx,yy = np.meshgrid(x,y)
    mesh = list(zip(xx.flatten(), yy.flatten()))

    # Make the Pool of workers
    pool = Pool(10)
    # Open the urls in their own threads
    # and return the results
    
    results = pool.map(pvalue, mesh)
    #close the pool and wait for the work to finish
    pool.close()
    pool.join()
    results = np.array(results).reshape(xx.shape)
    np.save("results.npy", results)
    print('success')