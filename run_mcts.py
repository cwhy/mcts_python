from env_utils import pit, RandomAgent
from mcts import self_play
from mcts_agent import MctsAgent
from my_tictactoe import h, init_state, get_available_actions, model, n_players, CliAgent
from table_utils import TablePolicy

n_actions = h ** 2
n_iters = 10
n_eps = 16
n_mcts = 128
state_ = init_state
env_get_actions, env_model = get_available_actions, model

mcts_agents = []
memory = TablePolicy(n_actions)
for i in range(n_players):
    agent = MctsAgent(i, n_actions, memory, get_available_actions, model, n_mcts)
    mcts_agents.append(agent)
for agent in mcts_agents:
    agent.assign_next_(mcts_agents[(agent.ag_id + 1) % n_players])

memory, mcts_agents = self_play(env_model, init_state,
                                 mcts_agents, memory, n_iters, n_eps)



for _ in range(5):
    pit(env_model, init_state, [mcts_agents[0], RandomAgent(1, env_model, get_available_actions)])
    print('-----------------------------------------')

for _ in range(5):
    pit(env_model, init_state, [RandomAgent(0, env_model, get_available_actions), mcts_agents[1]])
    print('-----------------------------------------')

pit(env_model, init_state, [mcts_agents[0], CliAgent(1, None, None)])
