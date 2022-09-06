import walk
import pandas as pd


class Experiment():
    def __init__(self, walk):
        self.walk = walk
        self.data = pd.DataFrame()

    def generate_walk(self, length):
        trial = []

        for i in range(length):
            step = self.walk.choose_step()
            self.walk.take_step(step)
            trial.append(float(step.position))

        return trial

    def generate(self, trials :int, length :int):
        for j in range(trials):
            self.data[str(j + 1)] = self.generate_walk(length)

    def test(self):
        pass

    def plot(self, num_trial=None):
        if num_trial is None:
            self.data.plot(legend=True)

        else:
            self.data.plot(y=num_trial, legend=True)
