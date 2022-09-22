# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
# %% Simple Random Walk
n = 1000
p = 0.5
A = np.random.rand(n, n) > p
W = np.cumsum(
    (A.astype('int') - 
    (~A).astype('int')
    ).flatten())
plt.plot(W)

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
    A = np.random.rand(n, length, 10) > p
    W = np.sum(
        (A.astype('int') - 
        (~A).astype('int')
        ), axis=1)
    return stats.normaltest(W)[-1].mean()
x = np.linspace(1, 4.5,15)
y = np.linspace(1, 4.5,15)
X, Y = np.meshgrid(x, y)
# %%
Z = np.vectorize(pvalue)(X, Y, 0.5)
np.save('pvalue.npy', Z)
print('saved')
plt.pcolormesh(X, Y, Z)
plt.colorbar()
plt.xlabel('log10(n)')
plt.ylabel('log10(length)')
plt.title('p-value for normal test')

# %%
Z = np.load('pvalue.npy')
# %%
Z.shape
# %%
plt.pcolormesh(X, Y, Z)
plt.colorbar()
plt.xlabel('log10(n)')
plt.ylabel('log10(length)')
plt.title('p-value for normal test')

# %%
