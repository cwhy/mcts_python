from typing import List
import numpy as np
from config import Actor, Env


def pit(env: Env, actors: List[Actor], render:bool=False):
    s = env.init_state
    ag_id = 0
    current_actor_ = actors[ag_id]
    done = False
    env_output = action = None
    while not done:
        if render:
            print(f'agent {env.agent_symbols[ag_id]} turn')
        # render_(s)
        action = current_actor_(s, render)
        env_output = env.model(s, action, ag_id, render=False)
        current_actor_ = actors[env_output.next_agent_id]
        ag_id = env_output.next_agent_id
        done = env_output.done
        if not done:
            s = env_output.next_state

    if render:
        print("done")
        env.render_(s)
        print(action)
    env.render_(env_output.next_state)
    print(env_output.rewards)
    print(env_output.message)
    return env_output.rewards


class RandomAgent:
    def __init__(self, agent_id: int, env: Env):
        self.get_actions = env.get_actions
        self.ag_id = agent_id

    def find_action(self, s, render=False):
        avail_a = self.get_actions(s)
        return np.random.choice(avail_a)
