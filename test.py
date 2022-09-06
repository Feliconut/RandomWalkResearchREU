# %%
import walk
from experiment import MultipleExperiment, SingleExperiment
# %%
test = SingleExperiment(walk.SimpleAsymmetricRandomWalk(0.8))
test.run(20)

# %%
test.data
# %%
test.plot()

# %%
test = MultipleExperiment(
    walk.LinearEdgeReinforcedRandomWalk, 
    n_trials = 10, 
    length = 100000)
test.run()
test.plot([0,1,2,3,4,5,6,7,8,9])

# %%
