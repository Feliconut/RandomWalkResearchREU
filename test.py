# %%
import walk
from experiment import SingleExperiment
# %%
test = SingleExperiment(walk.SimpleAsymmetricRandomWalk(0.8))
test.generate(5, 20)

# %%
test.data
# %%
test.plot(n_trials=[1, 2, 3, 4, 5])

# %%
