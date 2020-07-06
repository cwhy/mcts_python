import numpy as np
from gym import spaces


# bad: -2, empty: -1, players: 0, 1, 2...

def get_win_patterns(h: int):
    patterns = ([np.arange(h * i, h * i + h) for i in range(h)] +
                [np.arange(i, (h - 1) * h + i + 1, h) for i in range(h)] +
                [np.arange(0, h ** 2, h + 1),
                 np.arange(h - 1, (h - 1) * h + 1, h - 1)])
    state_patterns = []
    for pattern in patterns:
        base_vec = np.full(h * h, False)
        base_vec[pattern] = True
        state_patterns.append(base_vec)
    return state_patterns


class TicTacToeEnv:
    def __init__(self, player_symbols, board_size=3, win_size=3):
        super().__init__()
        self.win_size = win_size
        self.board_size = board_size
        self.symbols = player_symbols
        n_players = len(player_symbols)
        self.action_space = spaces.Discrete(self.board_size ** 2)
        self.rewards = {
            "still_in_game": 0,
            "draw": 10,
            "win": 20,
            "bad_position": -20
        }
        self.curr_player_ = 0
        self.get_next_player = lambda: (self.curr_player_ + 1) % n_players
        self.win_patterns = get_win_patterns(self.board_size)
        self.state_vector = np.full(self.board_size ** 2, -1)

    def reset(self):
        self.state_vector = np.full(self.board_size ** 2, -1)
        return self.state_vector

    # ------------------------------------------ GAME STATE CHECK ----------------------------------------
    def is_win(self, player):
        p_pos = self.state_vector == player
        for pattern in self.win_patterns:
            if all(p_pos & pattern == pattern):
                return True
        else:
            return False

    # ------------------------------------------ ACTIONS ----------------------------------------
    def step(self, action, curr_player):
        assert curr_player == self.curr_player_
        is_position_already_used = False

        if self.state_vector[action] != -1:
            is_position_already_used = True

        if is_position_already_used:
            self.state_vector[action] = -2
            reward_type = 'bad_position'
            done = True
        else:
            self.state_vector[action] = curr_player

            if self.is_win(curr_player):
                reward_type = 'win'
                done = True
            elif not any(self.state_vector == -1):
                reward_type = 'draw'
                done = True
            else:
                reward_type = 'still_in_game'
                done = False

        message = f"{self.symbols[self.curr_player_]} {reward_type}"
        self.curr_player_ = next_player = self.get_next_player()
        return (self.state_vector,
                self.rewards[reward_type],
                done,
                {'already_used_position': is_position_already_used,
                 'next_player': next_player,
                 'message': message})

    # ------------------------------------------ DISPLAY ----------------------------------------
    def get_state_vector_to_display(self):
        new_state_vector = []
        for value in self.state_vector:
            if value == -1:
                new_state_vector.append(' ')
            else:
                new_state_vector.append(self.symbols[value])
        return new_state_vector

    def print_grid_line(self, grid, offset=0):
        print(" " + "-" * (self.board_size * 4 + 1))
        for i in range(self.board_size):
            if grid[i + offset] == 0:
                print(" | " + " ", end='')
            else:
                print(" | " + str(grid[i + offset]), end='')
        print(" |")

    def display_grid(self, grid):
        for i in range(0, self.board_size * self.board_size, self.board_size):
            self.print_grid_line(grid, i)

        print(" " + "-" * (self.board_size * 4 + 1))
        print()

    def render(self, **kwargs):
        self.display_grid(self.get_state_vector_to_display())
