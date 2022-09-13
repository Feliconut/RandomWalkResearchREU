# %%
import numpy as np
import scipy
from matplotlib import pyplot as plt

import walk
from experiment import MultipleExperiment, SingleExperiment

# %%
test = SingleExperiment(walk.SimpleAsymmetricRandomWalk(0.8))
test.run(100)

# %%
test.data

# %%
test.plot()

# %%
test = MultipleExperiment(
    walk.LinearEdgeReinforcedRandomWalk,
    n_trials=1000,
    length=1000)
test.run()
test.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# %%
test.data

# %%

test.data.iloc[-1, :].hist(bins=50)
# %%

((test.data.iloc[-1, :]
    - test.data.iloc[-1, :].mean())
    / (test.data.iloc[-1, :].std())
 ).hist(
    bins=min(30, int(test.n_trials**(1/2))),
    density=True)
plt.title(f'K-squared test for n = {test.length}')
range = np.linspace(-3, 3, 100)
plt.plot(range, scipy.stats.norm.pdf(range, 0, 1))

# %%
test.test()
# %%
test = SingleExperiment(walk.SimpleSymmetricRandomWalk())
# %%
test.run(length = 100000)
# %%
test.plot()
    # %%
