from my_tictactoe import player_symbols, render_
import numpy as np


def pit(env_model, init_state, actors):
    s = init_state
    ag_id = 0
    current_actor_ = actors[ag_id]
    while True:
        print(f'agent {player_symbols[ag_id]} turn')
        # render_(s)
        action = current_actor_(s, render=False)
        next_s, rewards, done, next_id, message = env_model(s, action, ag_id)
        current_actor_ = actors[next_id]
        ag_id = next_id
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
    def __init__(self, agent_id: int, get_actions):
        self.get_actions = get_actions
        self.ag_id = agent_id

    def find_action(self, s, render=False):
        avail_a = self.get_actions(s)
        return np.random.choice(avail_a)
