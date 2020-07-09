import torch
from config import n_mcts, n_eps, n_iters, device, Env, h
from env_utils import pit, RandomAgent
from mcts import Mcts
from memory_utils import NNMemory
from my_tictactoe import ttt_env as env, ttt_net, CliAgent

memory = NNMemory(ttt_net, env.n_actions, device=device)
mcts = Mcts(n_mcts, memory, env)
print("Learning...")
mcts.self_play_(n_eps, n_iters)

print("Testing Against Random...")
print(f"as {env.agent_symbols[0]}")
mcts_agent0 = Mcts(n_mcts, memory, env).get_agent_decision_fn(0)
for _ in range(5):
    pit(env, [mcts_agent0, RandomAgent(1, env).find_action])
    print('-----------------------------------------')

print(f"as {env.agent_symbols[1]}")
for _ in range(5):
    mcts_agent1 = Mcts(n_mcts, memory, env).get_agent_decision_fn(1)
    pit(env, [RandomAgent(0, env).find_action, mcts_agent1])
    print('-----------------------------------------')

exp_name = f'ttt_{h}_nmcts_{n_mcts}.pth'
torch.save(ttt_net.state_dict(), f'saved_models/{exp_name}')
print("Testing Against Cli/Human...")
print(f"as {env.agent_symbols[0]}")
mcts_agent0 = Mcts(n_mcts, memory, env).get_agent_decision_fn(0)
pit(env, [mcts_agent0, CliAgent(1).find_action], render=True)
mcts_agent1 = Mcts(n_mcts, memory, env).get_agent_decision_fn(1)
print(f"as {env.agent_symbols[1]}")
pit(env, [CliAgent(0).find_action, mcts_agent1], render=True)

