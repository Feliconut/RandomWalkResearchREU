
import random


class RandomWalk():
    class BlockProperty():
        '''Sometimes we give each block some property that may be subject to change'''

    class Move():
        '''Describes the change to system state when one action is taken.
        weight: Used to determine probability to choosing this move
        age: system age after this move
        changed: change to block properties after this move
        position: new position after this move'''

        def __init__(self, weight: float, age: int, changed: dict, position: int) -> None:
            self.weight = weight
            self.age = age
            self.changed = changed
            self.position = position

    def __init__(self, random_seed: str = None):
        self.age = 0  # steps of simulation so far
        self.changed = dict()  # current block properties that are not default
        self.position = 0  # current step of the walker
        self.random_seed = random_seed  # used to make pseudoranodm choices.

    def default_property(self, n) -> BlockProperty:
        return self.__class__.BlockProperty()

    def get_property(self, n: int) -> BlockProperty:
        if n in self.changed:
            return self.changed[n]
        else:
            return self.default_property(n)

    def choose_move(self):
        # get moves
        steps = list(self.available_steps())
        steps.sort(key=lambda x: x.position)

        # choose a move
        random.seed(self.random_seed)
        step = random.choices(steps, weights=[x.weight for x in steps])[0]

        return step

    def do_step(self, step: Move):
        # update
        self.age = step.age
        self.changed.update(step.changed)
        self.position = step.position

    def available_steps(self):
        raise NotImplementedError()


class SimpleSymmetricRandomWalk(RandomWalk):
    def available_steps(self):
        Move = RandomWalk.Move
        n = self.position
        for i in [-1, 1]:  # relative motions
            yield Move(
                weight=1,
                age=self.age + 1,
                changed=dict(),  # no block property to update
                position=n+i
            )


class SimpleAsymmetricRandomWalk(RandomWalk):
    pass


class LinearEdgeReinforcedRandomWalk(RandomWalk):
    pass


if __name__ == "__main__":
    rw = SimpleSymmetricRandomWalk()
    for i in range(10):
        step = rw.choose_move()
        rw.do_step(step)
        print(step.position, step.weight, step.age, step.changed)
    print(rw.changed)
