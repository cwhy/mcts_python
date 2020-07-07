from my_tictactoe import player_symbols, render_
import numpy as np


def pit(env_model, init_state, agents):
    s = init_state
    curr_agent_ = agents[0]
    while True:
        print(f'agent {player_symbols[curr_agent_.ag_id]} turn')
        # render_(s)
        action = curr_agent_.find_action(s, render=False)
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
    def __init__(self, agent_id: int, _, get_actions):
        self.get_actions = get_actions
        self.ag_id = agent_id

    def find_action(self, s, render=False):
        avail_a = self.get_actions(s)
        return np.random.choice(avail_a)

