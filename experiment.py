from typing import Any, Callable, List
import walk
import pandas as pd
import numpy as np
from scipy import stats
import tests
from matplotlib import pyplot as plt


class MultipleExperiment():
    def __init__(self, walk_cls: type, n_trials: int = None, length: int = None, chunk_size: int = 10, *args, **kwargs):
        self.walk_cls = walk_cls
        self.n_trials = n_trials
        self.length = length
        self.chunk_size = chunk_size
        self.data = [0] * self.n_trials
        self.args = args
        self.kwargs = kwargs
        self.stats = {
            'jump': [],
            'ta0': []
        }

    def new_walk(self) -> walk.RandomWalk:
        return self.walk_cls(*self.args, **self.kwargs)

    def run(self):
        for i in range(self.n_trials):
            path = self.single_walk()
            self.accumulate_statistics(path)
            self.data[i] = path

        self.data = np.asarray(self.data).T
        self.data = pd.DataFrame(self.data)

    def single_walk(self) -> List[int]:
        path = [0] * self.length
        walk = self.new_walk()

        for i in range(self.length):
            step = walk.choose_step()
            walk.take_step(step)
            path[i] = (float(step.position))

        return path

    def accumulate_statistics(self, path):
        self.stats['jump'].append(tests.calc_jump_test(path, self.chunk_size))
        time_above_one = 0

        for pos in path:
            if pos > 0:
                time_above_one += 1

        self.stats['ta0'].append(time_above_one / self.length)

        return

    def bw_test(self):
        print(f"Brownian Motion Test Results for {self.n_trials} trials of length {self.length}")
        statistic, pvalue = tests.arcsine_test(self.stats['asin'])
        print(f"Arcsine Test Statistic: {statistic}, p-value:{pvalue}")
        statistic, pvalue = tests.jump_test(self.stats['jump'], self.chunk_size, self.length)
        print(f"Jump Test Statistic: {statistic}, p-value:{pvalue}")

    def norm_test(self):
        slc = self.data.iloc[-1, :]

        print(f"Test Results for {self.n_trials} trials of length {self.length}")
        statistic, pvalue = stats.normaltest(slc)
        print(f"(K-Squared) Statistic: {statistic}, p-value: {pvalue}")
        statistic, pvalue = stats.chisquare(slc)
        print(f"(Chi-Square) Statistic: {statistic}, p-value: {pvalue}")

        # Normalize Data
        mean = np.mean(slc)
        std = np.std(slc)
        slc = (slc - mean) / std

        statistic, pvalue = stats.kstest(slc, stats.norm.cdf, N=len(slc))
        print(f"(K-S) Statistic: {statistic}, p-value: {pvalue}")

    def plot(self, trial_indices: List[int]):
        self.data[trial_indices].plot(
            legend=True,
            title=f"{self.walk_cls.__name__}({self.n_trials} trials, {self.length} steps)",
            alpha=0.7,
            figsize=(20, 6)
        )

    def hist_plot(self):
        self.data.iloc[-1, :].hist()


''' Not Needed, Deprecate?
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
'''