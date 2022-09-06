# %%
import walk
from experiment import Experiment
# %%
test = Experiment(walk.SimpleAsymmetricRandomWalk(0.8))
test.generate(1, 20)
# %%
test.data
# %%
test.plot()
# %%
