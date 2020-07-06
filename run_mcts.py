from mcts import self_play_
from my_tictactoe import h, init_state, get_available_actions, model, n_players

curr_player_ = 0
n_actions = h ** 2
n_iters = 10
n_eps = 10
n_mcts = 10
state_ = init_state
env_get_actions, env_model = get_available_actions, model

self_play_(env_get_actions, env_model, init_state,
           n_players, n_actions, n_iters, n_eps, n_mcts)
