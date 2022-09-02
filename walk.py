
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

        def __init__(self, weight: float, t: int, changed: dict, position: int) -> None:
            self.weight = weight
            self.t = t
            self.changed = changed
            self.position = position

    def __init__(self, random_seed: str = None):
        self.t = 0  # steps of simulation so far
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

    def choose_move(self) -> Move:
        # get moves
        moves = list(self.available_moves())
        moves.sort(key=lambda x: x.position)

        # choose a move
        random.seed(self.random_seed)
        move = random.choices(moves, weights=[x.weight for x in moves])[0]

        return move

    def do_move(self, move: Move):
        # update
        self.t = move.t
        self.changed.update(move.changed)
        self.position = move.position

    def available_moves(self):
        raise NotImplementedError()


class SimpleSymmetricRandomWalk(RandomWalk):
    def available_moves(self):
        Move = RandomWalk.Move
        n = self.position
        for i in [-1, 1]:  # relative motions
            yield Move(
                weight=1,
                t=self.t + 1,
                changed=dict(),  # no block property to update
                position=n+i
            )


class SimpleAsymmetricRandomWalk(RandomWalk):
    pass


class LinearEdgeReinforcedRandomWalk(RandomWalk):
    class BlockProperty(RandomWalk.BlockProperty, int):
        pass

    def default_property(self, n) -> BlockProperty:
        return self.__class__.BlockProperty(1)

    def available_moves(self):
        Move = RandomWalk.Move
        n = self.position
        left, right = self.get_property(n-1), self.get_property(n)

        # left
        yield Move(
            weight=left,
            t=self.t + 1,
            changed={n-1: left+1},
            position=n-1)

        # right
        yield Move(
            weight=right,
            t=self.t + 1,
            changed={n: right+1},
            position=n+1)


if __name__ == "__main__":
    rw = SimpleSymmetricRandomWalk()
    for i in range(10):
        move = rw.choose_move()
        rw.do_move(move)
        print(move.position, move.weight, move.t, move.changed)
    print(rw.changed)
