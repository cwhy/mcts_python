from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np
import torch

from config import Action, StateID, State, Env
from mcts_agent import MctsAgent
from memory_utils import NNMemory


class Mcts:
    def __init__(self,
                 n_mcts: int,
                 env: Env):
        self.n_mcts = n_mcts
        self.states_: List[State] = []
        self.state_ids: Dict[bytes, StateID] = {}
        self.env = env
        self.agents: List[MctsAgent] = [
            MctsAgent(i, self.states_, self.state_ids, env)
            for i in range(env.n_agents)
        ]

    def search_(self, memory: NNMemory, start_state: State, ag_id: int):
        env_model = self.env.model
        curr_agent_ = self.agents[ag_id]
        n_agents = len(self.agents)
        s = start_state
        all_vs: Dict[StateID, np.ndarray] = defaultdict(lambda: np.zeros(n_agents))
        history: List[Tuple[int, StateID, Action]] = []
        done = False
        while not done:
            sb = s.tobytes()
            if sb not in self.state_ids:
                v = np.zeros(n_agents)
                v[curr_agent_.ag_id] += memory.get_v(s, curr_agent_.ag_id)
                self.states_.append(s)
                s_id = self.state_ids[sb] = len(self.states_) - 1
                all_vs[s_id] += v
                break
            else:
                s_id = self.state_ids[sb]
                action = curr_agent_.selection(s_id,
                                               memory.get_p(s, curr_agent_.ag_id))
                env_output = env_model(s, action, curr_agent_.ag_id, render=False)
                all_vs[s_id] += env_output.rewards
                history.append((curr_agent_.ag_id, s_id, action))
                curr_agent_ = self.agents[env_output.next_agent_id]
                s = env_output.next_state
                done = env_output.done

        s_vs = np.zeros(n_agents)
        for ag_id, s_id, a in reversed(history):
            s_vs += all_vs[s_id]
            self.agents[ag_id].update_qn_(s_id, a, s_vs[ag_id])

    def get_agent_decision_fn(self, memory_: NNMemory, ag_id: int):
        def decision(s, render=False):
            for _ in range(self.n_mcts):
                self.search_(memory_, s, ag_id)
            if render:
                np.set_printoptions(precision=3, suppress=True)
                torch.set_printoptions(precision=3)
                print("P: ", memory_.get_p(s, ag_id))
                print("V: ", memory_.get_v(s, ag_id))
            return self.agents[ag_id].find_action(s, render)
        return decision

    def self_play(self, memory_: NNMemory):
        ag_ids = []
        states = []
        policies = []
        values = []
        s = self.env.init_state
        curr_agent_ = self.agents[0]
        done = False
        total_rewards = np.zeros(len(self.agents))
        while not done:
            for _ in range(self.n_mcts):
                self.search_(memory_, s, curr_agent_.ag_id)

            policy = curr_agent_.find_policy(s)
            action = np.random.choice(len(policy), p=policy)
            env_output = self.env.model(s, action, curr_agent_.ag_id, render=False)
            total_rewards += env_output.rewards

            _states, _policies = self.env.get_symmetries(s, policy)
            states += _states
            policies += _policies
            ag_ids += len(states) * [curr_agent_.ag_id]

            curr_agent_ = self.agents[env_output.next_agent_id]
            s = env_output.next_state
            done = env_output.done

        values.extend(total_rewards for _ in range(len(states)))
        return ag_ids, states, policies, values
