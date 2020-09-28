import numpy as np
from typing import Hashable
from dataclasses import dataclass, replace
from config import h, State, Action, player_symbols, Env, device, \
    EnvOutput
from gridboard_utils import rewards_all, rewards_winner_take_all, rewards_individual, render_, \
    get_actions, get_symmetries, CliAgent
from networks import BoardNet

env_name = "WuZiQi"


@dataclass
class StateWZQ:
    array: np.ndarray


def _hash(state: StateWZQ, agent_id: int) -> Hashable:
    return hash((state.array.tobytes(), agent_id))


# bad: -2, empty: -1, players: 0, 1, 2...
init_state: State = StateWZQ(np.full(h ** 2, -1))
n_actions = h ** 2
n_players = len(player_symbols)

ttt_net = BoardNet(device, env_name, h ** 2, n_players, h ** 2)


# noinspection PyShadowingNames
def check_win(s_array: np.ndarray) -> bool:
    grid = s_array.reshape((h, h))
    sum_horizontal = np.sum(grid, axis=1)
    sum_vertical = np.sum(grid, axis=0)
    sum_diagonal_1 = np.sum(grid.diagonal())
    sum_diagonal_2 = np.sum(np.flipud(grid).diagonal())
    return (5 in sum_horizontal or 5 in sum_vertical
            or sum_diagonal_1 == 5 or sum_diagonal_2 == 5)


def model(s: StateWZQ, a: Action, player: int,
          render: bool = False) -> EnvOutput:
    s_new = replace(s, array=np.copy(s.array))
    if s_new.array[a] != -1:
        reward_type = 'bad_position'
        rewards = rewards_individual(-10, player)
        done = True
    else:
        s_new.array[a] = player
        p_pos = s_new.array == player
        if check_win(p_pos):
            reward_type = 'win'
            rewards = rewards_winner_take_all(1, player)
            done = True
        else:
            if not any(s_new.array == -1):
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
        render_(s_new)
    return EnvOutput(s_new, rewards, done, next_player, message)


wuziqi_env = Env(
    name=env_name,
    n_agents=n_players,
    n_actions=n_actions,
    init_state=lambda: init_state,
    model=model,
    state_utils=Env.StateUtils(
        hash=_hash,
        get_actions=get_actions,
        get_symmetries=lambda s, a: get_symmetries(s, a, wrapper=StateWZQ),
        render_=render_
    ),
    agent_symbols=player_symbols,
    cli_agent=CliAgent,
)
