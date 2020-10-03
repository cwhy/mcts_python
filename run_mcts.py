import numpy as np
import torch
from config import n_mcts, n_eps, n_iters, exp_name, env_name, n_pools, train_from_last
from tqdm import tqdm
from env_utils import pit, RandomAgent
from mcts import Mcts
from memory_utils import NNMemoryAnyState
from games import envs, nets, cli_agents
from glob import glob
import ray


ray.init()

env = envs[env_name]
neural_net_ = nets[env_name]
if train_from_last:
    saves = glob(f'saved_models/{exp_name}_*.pth')
    if len(saves) > 0:
        save_iters = [int(s.split('.')[0].split('_')[-1]) for s in saves]
        last_iter = max(save_iters)
        last_path = saves[save_iters.index(last_iter)]
        neural_net_.load_state_dict(torch.load(last_path))
        iters_ = 0
    else:
        print("Loading model failed: no saved model found, reinitilize neural network")
        iters_ = 0
else:
    iters_ = 0
CliAgent = cli_agents[env_name]
print("Learning...")

for _ in tqdm(range(n_iters)):
    iters_ += 1
    memory_ = NNMemoryAnyState(neural_net_, env)
    mcts = Mcts(n_mcts, env, max_depth=100)


    @ray.remote
    def do_episode_(i):
        return mcts.self_play(memory_, i)


    if n_pools > 0:
        # p = Pool(16)
        # exps = p.map(do_episode_, range(n_eps))
        exps = ray.get([do_episode_.remote(i) for i in range(n_eps)])
    else:
        exps = [do_episode(i) for i in range(n_eps)]
    exps_arrays = [np.concatenate([ex[i] for ex in exps], axis=0) for i in range(4)]
    neural_net_.train_(*exps_arrays)

net_save_path = f'saved_models/{exp_name}_{iters_}.pth'

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

torch.save(neural_net_.state_dict(), net_save_path)
while True:
    print("Testing Against Cli/Human...")
    print(f"as {env.agent_symbols[0]}")
    mcts_agent0 = Mcts(n_mcts, env).get_agent_decision_fn(memory_, 0)
    pit(env, [mcts_agent0, CliAgent(1).find_action], render=True)
    mcts_agent1 = Mcts(n_mcts, env).get_agent_decision_fn(memory_, 1)
    print(f"as {env.agent_symbols[1]}")
    pit(env, [CliAgent(0).find_action, mcts_agent1], render=True)
