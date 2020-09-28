from multiprocessing import Pool
import numpy as np
import torch
from config import n_mcts, n_eps, n_iters, exp_name
from tqdm import tqdm
from env_utils import pit, RandomAgent
from gridboard_utils import CliAgent
from mcts import Mcts
from memory_utils import NNMemoryAnyState
from my_tictactoe import ttt_env as env, ttt_net as neural_net_

print("Learning...")

for _ in tqdm(range(n_iters)):
    memory_ = NNMemoryAnyState(neural_net_, env)
    mcts = Mcts(n_mcts, env, max_depth=100)


    def do_episode_(i):
        return mcts.self_play(memory_, i)


    p = Pool(16)
    exps = p.map(do_episode_, range(n_eps))
    exps_arrays = [np.concatenate([ex[i] for ex in exps], axis=0) for i in range(4)]
    neural_net_.train_(*exps_arrays)

print("Testing Against Random...")
memory_ = NNMemoryAnyState(neural_net_, env)
print(f"as {env.agent_symbols[0]}")
mcts_agent0 = Mcts(n_mcts, env).get_agent_decision_fn(memory_, 0)
for _ in range(5):
    pit(env, [mcts_agent0, RandomAgent(1, env).find_action])
    print('-----------------------------------------')

print(f"as {env.agent_symbols[1]}")
for _ in range(5):
    mcts_agent1 = Mcts(n_mcts, env).get_agent_decision_fn(memory_, 1)
    pit(env, [RandomAgent(0, env).find_action, mcts_agent1])
    print('-----------------------------------------')

torch.save(neural_net_.state_dict(), f'saved_models/{exp_name}_{n_iters}.pth')
while True:
    print("Testing Against Cli/Human...")
    print(f"as {env.agent_symbols[0]}")
    mcts_agent0 = Mcts(n_mcts, env).get_agent_decision_fn(memory_, 0)
    pit(env, [mcts_agent0, CliAgent(1).find_action], render=True)
    mcts_agent1 = Mcts(n_mcts, env).get_agent_decision_fn(memory_, 1)
    print(f"as {env.agent_symbols[1]}")
    pit(env, [CliAgent(0).find_action, mcts_agent1], render=True)
