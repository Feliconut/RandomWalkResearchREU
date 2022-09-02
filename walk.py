
import random
from typing import Dict


class RandomWalk():
    class EdgeProperty():
        '''Sometimes we give each edge some property that may be subject to change.
        Examples of edge properties are:
        1. In simple random walk there is no edge property
        2. In Edge Reinforced RW we need to store the number of time each edge is visited
        '''

    class Step():
        '''Describes the change to system state when one action is taken.
        weight: Used to determine probability to choosing this step
        age: system age after this step
        modified_edges: change to edge properties after this step
        position: new position after this step'''

        def __init__(self, weight: float, t: int, modified_edges: dict, position: int) -> None:
            self.weight = weight
            self.t = t
            self.modified_edges: Dict[int,
                                      RandomWalk.EdgeProperty] = modified_edges
            self.position = position

    def __init__(self, random_seed: str = None):
        self.t = 0  # steps of simulation so far
        self.modified_edges = dict()  # current block properties that are not default
        self.position = 0  # current step of the walker
        self.random_seed = random_seed  # used to make pseudoranodm choices.

    def default_edge_property(self, n) -> EdgeProperty:
        return self.__class__.EdgeProperty()

    def get_edge_property(self, n: int) -> EdgeProperty:
        if n in self.modified_edges:
            return self.modified_edges[n]
        else:
            return self.default_edge_property(n)

    def choose_step(self) -> Step:
        # get moves
        steps = list(self.available_steps())
        steps.sort(key=lambda x: x.position)

        # choose a move
        random.seed(self.random_seed)
        step = random.choices(steps, weights=[x.weight for x in steps])[0]

        return step

    def take_step(self, move: Step):
        # update
        self.t = move.t
        self.modified_edges.update(move.modified_edges)
        self.position = move.position

    def available_steps(self):
        raise NotImplementedError()


class SimpleSymmetricRandomWalk(RandomWalk):
    def available_steps(self):
        Step = RandomWalk.Step
        n = self.position
        for i in [-1, 1]:  # relative motions
            yield Step(
                weight=1,
                t=self.t + 1,
                modified_edges=dict(),  # no block property to update
                position=n+i
            )


class SimpleAsymmetricRandomWalk(RandomWalk):
    pass


class LinearEdgeReinforcedRandomWalk(RandomWalk):
    class BlockProperty(RandomWalk.EdgeProperty, int):
        pass

    def default_edge_property(self, n) -> BlockProperty:
        return self.__class__.BlockProperty(1)

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
