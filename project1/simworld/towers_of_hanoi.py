import time

from prettytable import PrettyTable

from simworld.abstract_simworld import Simworld

from simworld.actions import Actions

class TowersOfHanoi(Simworld):
    def __init__(self, n_disks: int = 3, n_pegs: int = 3, n_timesteps: int = 1000):
        self.n_disks = n_disks
        self.n_pegs = n_pegs
        self.n_timesteps = n_timesteps
        self.current_timestep = 0
        self.state = None  # (smallest, ..., largest)
        self.actions = Actions(n_pegs * (n_pegs - 1))
        self.moves = self._get_moves()
        self.state_shape = (n_pegs, ) * n_disks

        self.state_history = []

    def initialize(self):
        self.current_timestep = 0
        self.state = (0, ) * self.n_disks
        self.state_history = [self.state]
        return self.state

    def _get_moves(self):
        moves = []
        for p1 in range(self.n_pegs):
            for p2 in range(self.n_pegs):
                if p1 == p2:
                    continue
                moves.append((p1, p2))
        return moves

    def _is_legal(self, from_peg, to_peg):
        if from_peg not in self.state:
            return False
        from_disk = self.state.index(from_peg)

        if to_peg in self.state[:from_disk]:
            return False

        return True

    def _move_disk(self, from_peg, to_peg):
        state = list(self.state)
        from_disk = state.index(from_peg)
        state[from_disk] = to_peg
        self.state = tuple(state)

    def next(self, action):
        self.current_timestep += 1
        from_peg, to_peg = self.moves[action]
        if self._is_legal(from_peg, to_peg):
            self._move_disk(from_peg, to_peg)
            is_won = self.is_won
            reward = 100 if is_won else 0
            finished = is_won or self.current_timestep >= self.n_timesteps
        else:
            reward = -1
            finished = False

        self.state_history.append(self.state)
        return self.state, reward, finished

    @property
    def is_won(self):
        prev = self.state[0]
        if prev != self.n_pegs - 1:
            return False
        for s in self.state:
            if s != prev:
                return False
            prev = s
        return True

    @classmethod
    def visualize(self, state_history):
        for i, state in enumerate(state_history):
            n_pegs = state_history[-1][0] + 1
            table = PrettyTable([f'Peg {x}' for x in range(n_pegs)])
            for disk, peg in enumerate(state):
                row = [''] * n_pegs
                row[peg] = 'X'*(disk + 1)
                table.add_row(row)
            print(table)
            input('next')

if __name__ == '__main__':
    t = TowersOfHanoi(n_disks=3, n_pegs=3)
    t.initialize()
    a = 3
    print('move:', t.moves[a])
    print(t.next(a))
    #print(t.state, t.state_shape, t.actions.n)
    # for a in range(t.actions.n):
    #     print(t.moves[a])
    #     print(t._is_legal(a))
