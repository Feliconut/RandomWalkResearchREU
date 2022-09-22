from typing import Any, Callable, List
import walk
import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt


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

        self.data[1] = path

    def plot(self, ):
        self.data.plot(legend=True)

    def test(self):
        statistic, pvalue = stats.normaltest(self.data)
        print(f"Statistic: {statistic}, p-value: {pvalue}")


class MultipleExperiment():
    def __init__(self, walk_cls: type, n_trials: int = None, length: int = None, *args, **kwargs):
        self.walk_cls = walk_cls
        self.n_trials = n_trials
        self.length = length
        self.data = None
        self.args = args
        self.kwargs = kwargs

    def new_walk(self) -> walk.RandomWalk:
        return self.walk_cls(*self.args, **self.kwargs)

    def run(self):
        self.data = np.asarray(self.single_walk())[np.newaxis, :].T
        for _ in range(1, self.n_trials):
            path = self.single_walk()
            self.data = np.concatenate((self.data, np.asarray(path)[np.newaxis, :].T), axis=1)

        self.data = pd.DataFrame(self.data)

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
            alpha=0.7,
            figsize=(20, 6)
        )

    def test(self):
        print(f"Test Results for {self.n_trials} trials of length {self.length}")
        statistic, pvalue = stats.normaltest(self.data.iloc[-1, :])
        print(f"(K-Squared) Statistic: {statistic}, p-value: {pvalue}")
        statistic, pvalue = stats.chisquare(self.data.iloc[-1, :])
        print(f"(Chi-Square) Statistic: {statistic}, p-value: {pvalue}")
        statistic, pvalue = stats.kstest(self.data.iloc[-1, :], stats.norm.cdf)
        print(f"(K-S) Statistic: {statistic}, p-value: {pvalue}")

    def hist_plot(self):
        self.data.iloc[-1, :].hist()
