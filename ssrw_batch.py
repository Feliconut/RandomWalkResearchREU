# %%
import numpy as np
import matplotlib.pyplot as plt
# %%
n = 100
p = 0.5
A = np.random.rand(n, n) > p
# %%
W = np.cumsum(
    (A.astype('int') - 
    (~A).astype('int')
    ).flatten())
# %%
plt.plot(W)
# %%
