# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
# %% Simple Random Walk
n = 10000
p = 0.5
A = np.random.rand(n, n) > p
W = np.cumsum(
    (A.astype('int') - 
    (~A).astype('int')
    ).flatten())
np.save('ssrw-1e8.npy',W)
# plt.plot(W)
# plt.xlabel('step')
# plt.title('simple random walk, p = 0.5, n = 1e6')


# %% Uniform random walk
n = 10000
A = np.random.rand(n, n)* 2 - 1
W = np.cumsum(A).flatten()
np.save('uniform-rw-1e8.npy',W)

# %% Standard normal random walk
n = 10000
A = np.random.standard_normal((n, n))
W = np.cumsum(A).flatten()
np.save('stdnorm-rw-1e8.npy',W)

# %% Exponential random walk
n = 10000
A = np.multiply(
    np.random.standard_exponential(size=(n, n)),
    np.random.choice([-1, 1], size=(n, n)))
    # We make the distribution symmetric
W = np.cumsum(A).flatten()
# plt.plot(W)
np.save('exp-rw-1e8.npy',W)
# %% Cauchy random walk
n = 10000
A = np.random.standard_cauchy(size=(n, n))
W = np.cumsum(A).flatten()
np.save('cauchy-rw-1e8.npy',W)
# plt.plot(W)
# %%
