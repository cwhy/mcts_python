from env_utils import pit, RandomAgent
from mcts import Mcts
from my_tictactoe import h, init_state, get_available_actions, model, n_players, \
    CliAgent, player_symbols

n_actions = h ** 2
n_iters = 10
n_eps = 16
n_mcts = 128
env_get_actions, env_model = get_available_actions, model

mcts = Mcts(n_actions, n_players, n_mcts, env_get_actions)
print("Learning...")
mcts.self_play_(env_model, init_state, n_eps, n_iters)

print("Testing Against Random...")
print(f"as {player_symbols[0]}")
for _ in range(5):
    pit(env_model, init_state,
        [mcts.get_agent_decision_fn(0, env_model),
         RandomAgent(1, get_available_actions).find_action])
    print('-----------------------------------------')

print(f"as {player_symbols[1]}")
for _ in range(5):
    pit(env_model, init_state,
        [RandomAgent(0, get_available_actions).find_action,
         mcts.get_agent_decision_fn(1, env_model)])
    print('-----------------------------------------')

print("Testing Against Cli/Human...")
print(f"as {player_symbols[0]}")
pit(env_model, init_state, [mcts.get_agent_decision_fn(0, env_model),
                            CliAgent(1).find_action])
print(f"as {player_symbols[1]}")
pit(env_model, init_state, [CliAgent(0).find_action,
                            mcts.get_agent_decision_fn(1, env_model)])
