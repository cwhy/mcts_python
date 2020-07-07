import itertools

from mcts import self_play
from mcts_agent import MctsAgent
from my_tictactoe import h, init_state, get_available_actions, model, n_players, player_symbols, render_
from table_utils import TablePolicy
import numpy as np

curr_player_ = 0
n_actions = h ** 2
n_iters = 10
n_eps = 16
n_mcts = 12
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

def pit(env_model, init_state, agents):
    s = init_state
    curr_agent_ = agents[0]
    while True:
        print(f'agent {player_symbols[curr_agent_.ag_id]} turn')
        # render_(s)
        policy = curr_agent_.find_policy(s, render=False)
        action = np.random.choice(len(policy), p=policy)
        next_s, rewards, done, next_id, message = env_model(s, action, curr_agent_.ag_id)
        curr_agent_ = agents[next_id]
        if done:
            print("done")
            render_(s)
            print(action)
            render_(next_s)
            print(rewards)
            print(message)
            return rewards
        else:
            s = next_s

class RandomAgent:
    def __init__(self, agent_id: int, env_model, get_actions):
        self.model = env_model
        self.get_actions = get_actions
        self.ag_id = agent_id
        self.combined_rewards = lambda v, v_next: v - v_next

    def find_policy(self, s, render=False):
        avail_a = self.get_actions(s)
        policy = np.zeros(n_actions)
        policy[avail_a] += 1 / len(avail_a)
        return policy


for _ in range(5):
    pit(env_model, init_state, [mcts_agents[0], RandomAgent(1, env_model, get_available_actions)])
    print('-----------------------------------------')

for _ in range(5):
    pit(env_model, init_state, [RandomAgent(0, env_model, get_available_actions), mcts_agents[1]])
    print('-----------------------------------------')
