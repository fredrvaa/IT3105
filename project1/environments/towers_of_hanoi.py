import time

from prettytable import PrettyTable

from environments.environment import Environment


class TowersOfHanoi(Environment):
    def __init__(self, n_disks: int = 3, n_pegs: int = 3, *args, **kwargs):
        """
        :param n_disks: Number of disks.
        :param n_pegs: Number of pegs.
        """

        super().__init__(*args, **kwargs)
        self.n_disks = n_disks
        self.n_pegs = n_pegs
        self.current_timestep = 0
        self.state = None  # (smallest, ..., largest)
        self.moves = self._get_moves()

        self.state_history = []

    def _get_moves(self) -> list:
        """Get all possible moves for the towers of hanoi game given number of disks and pegs.

        :return: A list of all moves (from_peg, to_peg).
        """

        moves = []
        for p1 in range(self.n_pegs):
            for p2 in range(self.n_pegs):
                if p1 == p2:
                    continue
                moves.append((p1, p2))
        return moves

    def _move_disk(self, from_peg, to_peg) -> None:
        """Moves disk from a peg to a different peg.

        :param from_peg: The peg that a disk should be moved from.
        :param to_peg: The peg that a disk should be moved to.
        """

        state = list(self.state)
        from_disk = state.index(from_peg)
        state[from_disk] = to_peg
        self.state = tuple(state)

    def _is_won(self) -> bool:
        """Checks whether the current state is won.

        :return: Whether or not the current state is won.
        """

        prev = self.state[0]
        if prev != self.n_pegs - 1:
            return False
        for s in self.state:
            if s != prev:
                return False
            prev = s
        return True

    def initialize(self) -> tuple:
        """Initializes environment/state and returns the initialized state.

        :return: The initial state.
        """

        self.current_timestep = 0
        self.state = (0, ) * self.n_disks
        self.state_history = []
        if self.store_states:
            self.state_history.append(self.state)
        return self.state

    def next(self, action: int) -> tuple[tuple, float, bool]:
        """Applies action to the environment, moving it to the next state.

        :param action: The action to perform
        :return: (next_state, reward, finished)
                    next_state: the current state of the environment after applying the action
                    reward: a numerical reward for moving to the state
                    finished: boolean specifying if the environment has reached some terminal condition
        """

        self.current_timestep += 1
        if self.action_legal_in_state(action, self.state):
            from_peg, to_peg = self.moves[action]
            self._move_disk(from_peg, to_peg)
            is_won = self._is_won()
            reward = 100 if is_won else 0
            finished = is_won or self.current_timestep >= self.n_timesteps
        else:
            reward = -1
            finished = False

        if self.store_states:
            self.state_history.append(self.state)
        return self.state, reward, finished

    def action_legal_in_state(self, action: int, state: tuple):
        """Checks whether an action is legal in a given state.

        :param action: Action to check.
        :param state: State to check.
        :return: Whether the action is legal in the given state.
        """

        from_peg, to_peg = self.moves[action]
        if from_peg not in state:
            return False
        from_disk = state.index(from_peg)

        if to_peg in state[:from_disk]:
            return False

        return True

    @property
    def state_shape(self) -> tuple:
        """The shape of the state space.

        :return: A tuple describing the shape of the state space.
        """

        return (self.n_pegs, ) * self.n_disks

    @property
    def actions(self) -> int:
        """The actions that can be performed.

        :return: Number of total actions
        """

        return self.n_pegs * (self.n_pegs - 1)

    def visualize(self, vis_sleep: float = 1.0) -> None:
        """Visualizes the state history."""
        print(f'Epsilon=0 run in {self.__class__.__name__}')
        for i, state in enumerate(self.state_history):
            n_pegs = self.state_history[-1][0] + 1
            table = PrettyTable([f'Peg {x}' for x in range(n_pegs)])
            for disk, peg in enumerate(state):
                row = [''] * n_pegs
                row[peg] = 'X'*(disk + 1)
                table.add_row(row)
            print(table)
            time.sleep(vis_sleep)
