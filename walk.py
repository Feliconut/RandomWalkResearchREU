
import random
from typing import Dict


class RandomWalk():
    class EdgeProperty():
        '''
        Sometimes we give each edge some property that may be subject to change.

        Examples of edge properties are:
        1. In simple random walk there is no edge property
        2. In Edge Reinforced RW we need to store the number of time each edge is visited
        '''
        @classmethod
        def default(cls, position: int):
            return None

    class Step():
        '''
        Describes the change to system state when one action is taken.

        weight: Used to determine probability to choosing this step
        age: system age after this step
        modified_edges: change to edge properties after this step
        position: new position after this step
        '''

        def __init__(self, weight: float, t: int, modified_edges: dict, position: int) -> None:
            self.weight: float = weight
            self.t: int = t
            self.modified_edges: Dict[int,
                                      RandomWalk.EdgeProperty] = modified_edges
            self.position: int = position

    def __init__(self, _random_seed: str = None):
        # steps of simulation so far
        self.t: int = 0
        # current edge properties that are not default
        self.modified_edges: Dict[int, RandomWalk.EdgeProperty] = dict()
        # current step of the walker
        self.position: int = 0
        # used to make pseudoranodm choices.
        self._random_seed: str = _random_seed

    def __reset_random(self):
        if self._random_seed is not None:
            random.seed(self._random_seed)

    def get_edge_property(self, position: int) -> EdgeProperty:
        if position in self.modified_edges:
            return self.modified_edges[position]
        else:
            return self.__class__.EdgeProperty.default(position)

    def choose_step(self) -> Step:
        # get steps
        steps = list(self.available_steps())
        steps.sort(key=lambda x: x.position)

        # choose a step
        self.__reset_random()
        step = random.choices(steps, weights=[x.weight for x in steps])[0]

        return step

    def take_step(self, step: Step):
        # update
        self.t = step.t
        self.modified_edges.update(step.modified_edges)
        self.position = step.position

    def available_steps(self):
        raise NotImplementedError()


class SimpleSymmetricRandomWalk(RandomWalk):
    '''
    In this type of random walk, the walker walk left or right with equal probability.
    '''

    def available_steps(self):
        Step = RandomWalk.Step
        yield from [
            Step(
                weight=1,
                t=self.t + 1,
                modified_edges=dict(),  # no edge property
                position=position
            ) for position in [self.position-1, self.position+1]]


class SimpleAsymmetricRandomWalk(RandomWalk):
    pass


class LinearEdgeReinforcedRandomWalk(RandomWalk):
    class EdgeProperty(RandomWalk.EdgeProperty, int):
        '''
        This is the number of times this edge is visited.
        '''
        @classmethod
        def default(cls, position: int):
            return cls(1)

    def available_steps(self):
        Step = RandomWalk.Step
        n = self.position
        left, right = self.get_edge_property(n-1), self.get_edge_property(n)

        # left
        yield Step(
            weight=left,
            t=self.t + 1,
            modified_edges={n-1: left+1},
            position=n-1)

        # right
        yield Step(
            weight=right,
            t=self.t + 1,
            modified_edges={n: right+1},
            position=n+1)


# %%
if __name__ == "__main__":
    rw = LinearEdgeReinforcedRandomWalk()
    for i in range(20):
        step = rw.choose_step()
        rw.take_step(step)
        print(step.position, step.weight, step.t, step.modified_edges)
    print([rw.get_edge_property(i).__repr__() for i in range(-10, 10)])
