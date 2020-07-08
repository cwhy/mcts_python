from collections import defaultdict
from typing import Dict, List
import numpy as np
from math_calc import ucb_all

Action = int
State = np.ndarray
StateID = int
c_puct = 1.


class MctsAgent:
    def __init__(self, agent_id: int, n_actions: int, memory,
                 env_get_actions):
        self.ag_id = agent_id
        self.memory_ = memory
        self.states_: List[np.ndarray] = []
        self.state_ids: Dict[bytes, StateID] = {}
        self.visits: Dict[StateID, np.ndarray] \
            = defaultdict(lambda: np.zeros(n_actions))
        self.qs: Dict[StateID, np.ndarray] \
            = defaultdict(lambda: np.zeros(n_actions))
        self.combined_rewards = lambda v, v_next: v - v_next
        self.next_agent = None

        self.get_actions = env_get_actions

    def assign_next_(self, agent: "MctsAgent"):
        self.next_agent = agent

    def selection(self, s_id):
        s = self.states_[s_id]
        avail_a = self.get_actions(s)
        if len(avail_a) == 1:
            return avail_a[0]
        else:
            ucbs = ucb_all(qs=self.qs[s_id],
                           c_puct_normed_by_sum=c_puct * np.sqrt(
                               self.visits[s_id][avail_a].sum()),
                           ps=self.memory_.get_p(s, self.ag_id),
                           nas=self.visits[s_id])
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
        if s_id not in self.visits:
            print(s)  # TODO: some bugs here, idk
        policy_count = self.visits[s_id]
        if render:
            print(self.qs[s_id])
            print(self.visits[s_id])

        if not policy_count.any():
            avail_a = self.get_actions(s)
            policy_count[avail_a] += 1 / len(avail_a)
            return policy_count
        else:
            return policy_count / policy_count.sum()

    def find_action(self, s: State, render=False):
        policy = self.find_policy(s, render=render)
        action = np.random.choice(len(policy), p=policy)
        return action
