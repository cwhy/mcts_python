from dataclasses import dataclass, replace
import numpy as np
from typing import Hashable
from config import h, State, Action, player_symbols, Env, device, \
    EnvOutput
from games.gridboard_utils import rewards_all, rewards_winner_take_all, rewards_individual, \
    get_actions, move_along_in_dirs, GridBoard
from networks import BasicBoardNet

env_name = "Reversi"
board = GridBoard(h, h)
render_ = board.render_


@dataclass
class StateReversi:
    array: np.ndarray

    @property
    def get_array(self) -> np.ndarray:
        return self.array


def _hash(state: StateReversi, agent_id: int) -> Hashable:
    return hash((state.array.tobytes(), agent_id))


# bad: -2, empty: -1, players: 0, 1, 2...
init_state: State = StateReversi(np.full(h ** 2, -1))
n_actions = h ** 2
n_players = len(player_symbols)

reversi_net = BasicBoardNet(device, env_name, h ** 2, n_players, h ** 2)


def update_array_(s_array: np.ndarray, action: Action, player: int):
    action_idx = (action // h, action % h)
    grid = s_array.reshape((h, h))
    to_eat = []
    for move_fn in move_along_in_dirs:
        pending = []
        pos = action_idx
        while True:
            pos = move_fn(*pos)
            if not board.check_bound(pos):
                break
            else:
                if grid[pos] == player:
                    to_eat += pending
                    break
                elif grid[pos] != -1:
                    pending.append(pos)
    s_array[[board.pos_to_arr_idx(i) for i in to_eat]] = player
    s_array[action] = player


def model(s: StateReversi, a: Action, player: int,
          render: bool = False) -> EnvOutput:
    s_new = replace(s, array=np.copy(s.array))
    if s_new.array[a] != -1:
        reward_type = 'bad_position'
        rewards = rewards_individual(-10, player)
        done = True
    else:
        update_array_(s_new.array, a, player)
        if not any(s_new.array == -1):
            count = np.sum(s_new.array == player)
            if count > h ** 2 / 2:
                reward_type = 'win'
                rewards = rewards_winner_take_all(1, player)
            elif count > h ** 2 / 2:
                reward_type = 'lose'
                rewards = rewards_winner_take_all(1, (player + 1) % n_players)
            else:
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


reversi_env = Env(
    name=env_name,
    n_agents=n_players,
    n_actions=n_actions,
    init_state=lambda: (init_state, 0),
    model=model,
    state_utils=Env.StateUtils(
        hash=_hash,
        get_actions=get_actions,
        get_symmetries=lambda s, a: board.get_symmetries_4(s, a, wrapper=StateReversi),
        render_=render_
    ),
    agent_symbols=player_symbols,
    cli_agent=board.get_actor,
)
