import numpy as np
from typing import Hashable
from dataclasses import dataclass, replace
from config import h, State, Action, player_symbols, Env, device, \
    EnvOutput
from games.gridboard_utils import rewards_all, rewards_winner_take_all, rewards_individual, render_, \
    get_actions, get_symmetries, CliAgent
from networks import BasicBoardNet

env_name = "TicTacToe"


@dataclass
class StateTTT:
    array: np.ndarray
    turn: int


def _hash(state: StateTTT, agent_id: int) -> Hashable:
    return hash((state.array.tobytes(), agent_id, state.turn))


# bad: -2, empty: -1, players: 0, 1, 2...
init_state: State = StateTTT(np.full(h ** 2, -1), turn=0)
n_actions = h ** 2
n_players = len(player_symbols)

ttt_net = BasicBoardNet(device, env_name, h ** 2 + 1, n_players, h ** 2)


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


win_patterns = get_win_patterns(h)


def model(s: StateTTT, a: Action, player: int,
          render: bool = False) -> EnvOutput:
    s_new = replace(s, array=np.copy(s.array))
    if s_new.array[a] != -1:
        reward_type = 'bad_position'
        rewards = rewards_individual(-10, player)
        done = True
    else:
        s_new.array[a] = player
        p_pos = s_new.array == player
        for pattern in win_patterns:
            if all(p_pos & pattern == pattern):
                reward_type = 'win'
                rewards = rewards_winner_take_all(1, player)
                done = True
                break
        else:
            if not any(s_new.array == -1):
                reward_type = 'draw'
                rewards = rewards_all(0)
                done = True
            else:
                reward_type = 'still_in_game'
                rewards = tuple(0 for _ in range(n_players))
                done = False
    if h == 3 or s_new.turn == 1:
        next_player = (player + 1) % n_players
        s_new.turn = 0
    else:
        next_player = player
        s_new.turn = 1
    message = f"{player_symbols[player]} {reward_type} " \
              f"with reward {rewards[player]}"
    if render:
        render_(s_new)
    return EnvOutput(s_new, rewards, done, next_player, message)


ttt_env = Env(
    name=env_name,
    n_agents=n_players,
    n_actions=n_actions,
    init_state=lambda: init_state,
    model=model,
    state_utils=Env.StateUtils(
        hash=_hash,
        get_actions=get_actions,
        get_symmetries=lambda s, a: get_symmetries(s, a,
                                                   wrapper=lambda na: replace(s, array=na)),
        render_=render_
    ),
    agent_symbols=player_symbols,
    cli_agent=CliAgent,
)
