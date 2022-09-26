# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
# %% Simple Random Walk
n = 100000
p = 0.5
A = np.random.rand(n, n) > p
W = np.cumsum(
    (A.astype('int') - 
    (~A).astype('int')
    ).flatten())
np.save('ssrw-1e10.npy',W,)
# plt.plot(W)
# plt.xlabel('step')
# plt.title('simple random walk, p = 0.5, n = 1e6')

# %% Simple Random Walk, Scaled
n = 1000
p = 0.5
A = np.random.rand(n, n) > p
W = np.cumsum(
    (A.astype('int') - 
    (~A).astype('int')
    ).flatten())
plt.plot(W/n)
plt.xlabel('step')
plt.title('simple random walk, y-Scaled, p = 0.5, n = 1e6')

stats.normaltest(W)
# %% Simple Random Walk, slice normal test
n = 1000
length = 1000
p = 0.5
A = np.random.rand(n, length, 5) > p
W = np.sum(
    (A.astype('int') - 
    (~A).astype('int')
    ), axis=1)
*_, pvalue = stats.normaltest(W)
# %% p value for different n and length
def pvalue(n, length, p):
    print(f"n = {n}, length = {length}")
    n, length = int(10**n), int(10**length)
    #TODO this piece of code will crash kernel if both > 4.5
    A = np.random.rand(n, length, 10) > p
    W = np.sum(
        (A.astype('int') - 
        (~A).astype('int')
        ), axis=1)
    return stats.normaltest(W)[-1].mean()
x = np.linspace(1, 4,15)
y = np.linspace(1, 4,15)
X, Y = np.meshgrid(x, y)
# %%
Z = np.vectorize(pvalue)(X, Y, 0.5)
np.save('data/pvalue.npy', Z)
print('saved')
plt.pcolormesh(X, Y, Z)
plt.colorbar()
plt.xlabel('log10(n)')
plt.ylabel('log10(length)')
plt.title('p-value for normal test')

# %%
Z = np.load('data/pvalue.npy')
# %%
Z.shape
# %%
plt.pcolormesh(X, Y, Z)
plt.colorbar()
plt.xlabel('log10(n)')
plt.ylabel('log10(length)')
plt.title('p-value for normal test')

# %%
