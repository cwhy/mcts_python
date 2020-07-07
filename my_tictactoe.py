import numpy as np

# bad: -2, empty: -1, players: 0, 1, 2...
State = np.ndarray
Action = int

h = 3
player_symbols = ['x', 'o']
n_players = len(player_symbols)
init_state: State = np.full(h ** 2, -1)


# noinspection PyShadowingNames
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


def get_available_actions(s: State):
    return np.where(s == -1)[0]


win_patterns = get_win_patterns(h)


def rewards_winner_take_all(num: float, player: int):
    rewards = [-num] * 2
    rewards[player] += num * 2
    return tuple(rewards)


def rewards_individual(num: float, player: int):
    rewards = [0] * 2
    rewards[player] += num
    return tuple(rewards)


def rewards_all(num: float):
    rewards = [num] * 2
    return tuple(rewards)


def model(s: State, a: Action, player: int, render: bool = False):
    s = np.copy(s)
    if s[a] != -1:
        reward_type = 'bad_position'
        rewards = rewards_individual(-10, player)
        done = True
    else:
        s[a] = player
        p_pos = s == player
        for pattern in win_patterns:
            if all(p_pos & pattern == pattern):
                reward_type = 'win'
                rewards = rewards_winner_take_all(1, player)
                done = True
                break
        else:
            if not any(s == -1):
                reward_type = 'draw'
                rewards = rewards_all(0)
                done = True
            else:
                reward_type = 'still_in_game'
                rewards = tuple(0 for _ in range(n_players))
                done = False
    next_player = (player + 1) % n_players
    message = f"{player_symbols[player]} {reward_type} " \
              f"with reward {rewards[player]}"
    if render:
        render_(s)
    return s, rewards, done, next_player, message


def render_(s):
    new_state_vector = []
    for value in s:
        if value == -1:
            new_state_vector.append(' ')
        else:
            new_state_vector.append(player_symbols[value])

    for i in range(0, h ** 2, h):
        print_grid_line_(new_state_vector, i)

    print(" " + "-" * (h * 4 + 1))
    print()


def print_grid_line_(grid, offset=0):
    print(" " + "-" * (h * 4 + 1))
    for i in range(h):
        # if grid[i + offset] == 0:
        #     print(" | " + " ", end='')
        # else:
        print(" | " + str(grid[i + offset]), end='')
    print(" |")


class CliAgent:
    def __init__(self, agent_id: int):
        self.ag_id = agent_id

    def find_action(self, s, render=False):
        print("Current Game State:")
        render_(s)
        print("Position numbers:")
        for i in range(0, h ** 2, h):
            print_grid_line_(np.arange(0, h**2), i)

        print(" " + "-" * (h * 4 + 1))
        i = int(input("Enter your next move as position number:"))
        return i
