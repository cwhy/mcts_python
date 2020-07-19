from collections import defaultdict
from typing import Dict, List
import numpy as np

from config import StateID, c_puct, State, Env
from math_calc import ucb_all


class MctsAgent:
    def __init__(self, agent_id: int,
                 states: List[State],
                 state_ids: Dict[bytes, StateID],
                 env: Env):
        self.ag_id = agent_id
        self.visits: Dict[StateID, np.ndarray] \
            = defaultdict(lambda: np.zeros(env.n_actions))
        self.qs: Dict[StateID, np.ndarray] \
            = defaultdict(lambda: np.zeros(env.n_actions))
        self.states_ = states
        self.state_ids = state_ids
        self.n_actions = env.n_actions

        self.get_actions = env.get_actions

    def selection(self, s_id: int, ps: np.ndarray) -> int:
        s = self.states_[s_id]
        avail_a = self.get_actions(s)
        if len(avail_a) == 1:
            return avail_a[0]
        else:
            ucbs = ucb_all(qs=self.qs[s_id],
                           c_puct_normed_by_sum=c_puct * np.sqrt(
                               self.visits[s_id][avail_a].sum()),
                           ps=ps/ps[avail_a].sum(),
                           n_as=self.visits[s_id])
            # print(self.visits[s_id])
            # print(self.qs[s_id])
            # print(ucbs)
            # print("--")
            return avail_a[np.argmax(ucbs[avail_a])]

    def update_qn_(self, s_id, action, v):
        n_sa = self.visits[s_id][action]
        self.qs[s_id][action] = (n_sa * self.qs[s_id][action] + v) / (n_sa + 1)
        self.visits[s_id][action] += 1
        return v

    def find_policy(self, s: State, render=False):
        sb = s.tobytes()
        assert sb in self.state_ids
        s_id = self.state_ids[sb]
        assert s_id in self.visits
        policy_count = self.visits[s_id]
        if render:
            print("Q: ", self.qs[s_id])
            print("N: ", self.visits[s_id])

        if not policy_count.any():
            avail_a = self.get_actions(s)
            policy_count[avail_a] += 1 / len(avail_a)
            return policy_count
        else:
            return policy_count / policy_count.sum()

    def find_action(self, s: State, render=False):
        policy = self.find_policy(s, render=render)
        action = np.argmax(policy)
        return action
