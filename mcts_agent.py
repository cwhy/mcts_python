from collections import defaultdict
import itertools
from typing import Dict, List

import numpy as np

from math_calc import ucb_all

Action = int
State = np.ndarray
StateID = int
c_puct = 1.0


class MctsAgent:
    def __init__(self, agent_id: int, n_actions: int, memory):
        self.ag_id = agent_id
        self.memory_ = memory
        self.states_ = []
        self.state_ids: Dict[bytes, StateID] = {}
        self.visits: Dict[StateID, np.ndarray] \
            = defaultdict(lambda: np.zeros(n_actions))
        self.qs: Dict[StateID, np.ndarray] \
            = defaultdict(lambda: np.zeros(n_actions))
        self.combined_rewards = lambda v, v_next: v - v_next

    def search(self, s: State, env_get_actions, env_model, agents_gen):
        sb = s.tobytes()
        if sb not in self.state_ids:
            v = 0  # self.memory_.get_v(s)
            self.states_.append(s)
            self.state_ids[sb] = len(self.states_)
            return v
        else:
            s_id = self.state_ids[sb]
            ucbs = ucb_all(qs=self.qs[s_id],
                           c_puct=c_puct,
                           ps=self.memory_.get_p(s_id),
                           nas=self.visits[s_id])
            action = np.argmax(ucbs[env_get_actions(s)])
            next_s, rewards, done, next_id, message = env_model(s, action, self.ag_id)
            if done:
                return rewards[self.ag_id]
            next_agent = next(agents_gen)
            print(next_id, next_agent.ag_id)
            assert next_id == next_agent.ag_id
            v_next = next_agent.search(next_s, env_get_actions, env_model, agents_gen)
            v = self.combined_rewards(rewards[self.ag_id], v_next)  # TODO  Figure out how to forward multiagent rewards
            n_sa = self.visits[s_id][action]
            self.qs[s_id][action] = (n_sa * self.qs[s_id][action] + v) / (n_sa + 1)
            self.visits[s_id][action] += 1
            return v

    def find_policy(self, env_get_actions, env_model, s: State, agents, n_mcts):
        print(f"agent {self.ag_id} move")
        for _ in range(n_mcts):
            agents_gen = itertools.cycle(agents[self.ag_id+1:] + agents[:self.ag_id+1])
            self.search(s, env_get_actions, env_model, agents_gen)
        sb = s.tobytes()
        assert sb in self.state_ids
        s_id = self.state_ids[sb]
        assert s_id in self.visits
        print(self.visits[s_id])
        return self.visits[s_id]/self.visits[s_id].sum()
