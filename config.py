import numpy as np
from typing import Callable, NamedTuple, Optional, List, Type, Tuple, Hashable, TypeVar, Generic
from typing_extensions import Protocol

## Types
Action = int
StateID = int
Actions = np.ndarray

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

    class Agent(Protocol):
        def find_action(self, s: State, render: bool) -> Action:
            pass

    class EnvModel(Protocol):
        def __call__(self, s: State, a: Action, player: int,
                     render: bool) -> 'EnvOutput[State]':
            pass

    name: str
    n_agents: int
    n_actions: int
    init_state: Callable[[], State]
    model: EnvModel
    state_utils: StateUtils

    agent_symbols: Optional[List[str]] = None
    cli_agent: Optional[Type[Agent]] = None


## Environment
player_symbols = ['x', 'o']
h = 3

## MCTS setting
n_iters = 2
n_eps = 2
n_mcts = 128
max_depth = 100

## Bandit setting
c_puct = 1.0

## Neural Network setting
max_batch_size = 1024
lr = 0.005
device = 'cpu'

## Experiment setting
exp_name = f'ttt_{h}_n_mcts_{n_mcts}_net1'
