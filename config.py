from typing import Callable, NamedTuple, Optional, List, Type, Tuple
from typing_extensions import Protocol
import numpy as np

## Types
Action = int
State = np.ndarray
StateID = int


class EnvOutput(NamedTuple):
    next_state: State
    rewards: np.ndarray
    done: bool
    next_agent_id: int
    message: str


class Actor(Protocol):
    def __call__(self, s: State, render: bool) -> Action:
        pass


class Agent(Protocol):
    def find_action(self, s: State, render: bool) -> Action:
        pass


class EnvModel(Protocol):
    def __call__(self, s: State, a: Action, player: int, render: bool) -> EnvOutput:
        pass


class Env(NamedTuple):
    name: str
    n_agents: int
    n_actions: int
    init_state: State
    agent_symbols: Optional[List[str]]
    get_actions: Callable[[State], np.ndarray]
    model: EnvModel
    render_: Optional[Callable[[State], None]]
    get_symmetries: Optional[Callable[[State, np.ndarray],
                                      Tuple[List[State], List[np.ndarray]]]]
    cli_agent: Optional[Type[Agent]]


## Environment
player_symbols = ['x', 'o']
h = 3

## MCTS setting
n_iters = 64
n_eps = 8
n_mcts = 32

## Bandit setting
c_puct = 1.0

## Neural Network setting
max_batch_size = 1024
lr = 0.001
device = 'cuda'

## Experiment setting
exp_name = f'ttt_{h}_n_mcts_{n_mcts}_net1'
