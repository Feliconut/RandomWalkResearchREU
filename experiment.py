from typing import List
import walk
import pandas as pd


class SingleExperiment():
    def __init__(self, walk: walk.RandomWalk):
        self.walk = walk
        self.data = pd.DataFrame()

    def generate_walk(self, length: int) -> List[int]:
        path = []

        for i in range(length):
            step = self.walk.choose_step()
            self.walk.take_step(step)
            path.append(float(step.position))

        return path

    def generate(self, n_trials: int, length: int):
        for j in range(n_trials):
            self.data[j + 1] = self.generate_walk(length)

    def plot(self, n_trials: int = None):
        if n_trials is None:
            self.data.plot(legend=True)

        else:
            self.data.plot(y=n_trials, legend=True)
