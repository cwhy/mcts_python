import numpy as np
from typing import Callable, NamedTuple, Optional, List, Type, Tuple, Hashable, TypeVar, Generic
from typing_extensions import Protocol

## Types
Action = int
StateID = int
Actions = np.ndarray
Rewards = np.ndarray

State = TypeVar('State')


class EnvOutput(NamedTuple, Generic[State]):
    next_state: State
    rewards: np.ndarray
    done: bool
    next_agent_id: int
    message: str


class Env(NamedTuple, Generic[State]):
    class StateUtils(NamedTuple):
        hash: Callable[[State, int], Hashable]
        get_actions: Callable[[State, int], Actions]
        render_: Optional[Callable[[State], None]] = None
        get_symmetries: Optional[Callable[[State, Actions],
                                          Tuple[List[State], List[
                                              Actions]]]] = None

    class Actor(Protocol):
        def __call__(self, s: State, render: bool) -> Action:
            pass

    class EnvModel(Protocol):
        def __call__(self, s: State, a: Action, player: int,
                     render: bool) -> 'EnvOutput[State]':
            pass

    name: str
    n_agents: int
    n_actions: int
    init_state: Callable[[], Tuple[State, int]]
    model: EnvModel
    state_utils: StateUtils

    agent_symbols: Optional[List[str]] = None
    cli_agent: Optional[Callable[[int], Actor]] = None
    web_agent: Optional[Callable[[int], Actor]] = None
    cycle_reward: Optional[Rewards] = None
    timeout_reward: Optional[Rewards] = None


## Environment
env_name = 'Reversi'
player_symbols = ['*', 'o']
h = 8

## MCTS setting
n_iters = 0
n_eps = 32
n_mcts = 256
max_depth = 100

## Bandit setting
c_puct = 1.0

## Neural Network setting
max_batch_size = 1024
lr = 0.01

## Experiment setting
exp_name = f'{env_name}_{h}_n_mcts_{n_mcts}_net1'
train_from_last = True
device = 'cpu'
n_pools = 8
