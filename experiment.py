from typing import Any, Callable, List
import walk
import pandas as pd


class SingleExperiment():
    def __init__(self, walk: walk.RandomWalk):
        self.walk = walk
        self.data = pd.DataFrame()

    def run(self, length: int) -> List[int]:
        path = []

        for i in range(length):
            step = self.walk.choose_step()
            self.walk.take_step(step)
            path.append(float(step.position))

        self.data[1]=path

    def plot(self, ):
        self.data.plot(legend=True)


class MultipleExperiment():
    def __init__(self, walk_cls: type, n_trials: int = None, length:int = None, *args, **kwargs):
        self.walk_cls = walk_cls
        self.n_trials = n_trials
        self.length = length
        self.data = pd.DataFrame()
        self.args = args
        self.kwargs = kwargs
    
    def new_walk(self) -> walk.RandomWalk:
        return self.walk_cls(*self.args, **self.kwargs)

    def run(self):
        for i in range(self.n_trials):
            path = self.single_walk()
            self.data[i] = path

    def single_walk(self) -> List[int]:
        path = []

        walk = self.new_walk()
        for _ in range(self.length):
            step = walk.choose_step()
            walk.take_step(step)
            path.append(float(step.position))

        return path

    def plot(self, trial_indices: List[int]):
        self.data[trial_indices].plot(
            legend=True,
            title=f"{self.walk_cls.__name__}({self.n_trials} trials, {self.length} steps)",
        )
