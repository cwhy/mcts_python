import itertools
from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np
import torch
from tqdm import tqdm

from config import Action, StateID, State, Env
from mcts_agent import MctsAgent
from memory_utils import FlatMemory, NNMemory


class Mcts:
    def __init__(self,
                 n_mcts: int,
                 memory: NNMemory,
                 env: Env):
        self.n_mcts = n_mcts
        self.agents: List[MctsAgent] = []
        self.states_: List[State] = []
        self.state_ids: Dict[bytes, StateID] = {}
        self.memory_ = memory
        self.env = env
        for i in range(env.n_agents):
            agent = MctsAgent(i, self.states_, self.state_ids, env)
            self.agents.append(agent)

        for agent in self.agents:
            agent.assign_next_(self.agents[(agent.ag_id + 1) % env.n_agents])

    def reset_(self):
        self.states_.clear()
        self.state_ids.clear()
        for agent in self.agents:
            agent.reset_()

    def self_play_(self, n_eps, n_iters):
        for _ in tqdm(range(n_iters)):
            self.memory_.clear_()
            for _ in range(n_eps):
                self.reset_()
                s = self.env.init_state
                agents_gen = itertools.cycle(self.agents)
                curr_agent_ = next(agents_gen)
                done = False
                total_rewards = np.zeros(len(self.agents))
                while not done:
                    for _ in range(self.n_mcts):
                        self.search_(s, curr_agent_.ag_id)

                    policy = curr_agent_.find_policy(s)
                    action = np.random.choice(len(policy), p=policy)
                    env_output = self.env.model(s, action, curr_agent_.ag_id, render=False)
                    total_rewards += env_output.rewards
                    self.memory_.add_with_symmetry(curr_agent_.ag_id,
                                                   s,
                                                   policy,
                                                   self.env.get_symmetries)
                    curr_agent_ = next(agents_gen)
                    assert env_output.next_agent_id == curr_agent_.ag_id
                    s = env_output.next_state
                    done = env_output.done
                self.memory_.assign_values_(total_rewards)
            self.memory_.train_()

    def get_agent_decision_fn(self, ag_id: int):
        def decision(s, render=False):
            for _ in range(self.n_mcts):
                self.search_(s, ag_id)
            if render:
                np.set_printoptions(precision=3, suppress=True)
                torch.set_printoptions(precision=3)
                print("P: ", self.memory_.get_p(s, ag_id))
                print("V: ", self.memory_.get_v(s, ag_id))
            return self.agents[ag_id].find_action(s, render)

        return decision


    def search_(self, start_state: State, ag_id: int):
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
                v[curr_agent_.ag_id] += self.memory_.get_v(s, curr_agent_.ag_id)
                self.states_.append(s)
                s_id = self.state_ids[sb] = len(self.states_) - 1
                all_vs[s_id] += v
                break
            else:
                sb = s.tobytes()
                s_id = self.state_ids[sb]
                action = curr_agent_.selection(s_id,
                                               self.memory_.get_p(s, curr_agent_.ag_id))
                env_output = env_model(s, action, curr_agent_.ag_id, render=False)
                all_vs[s_id] += env_output.rewards
                history.append((curr_agent_.ag_id, s_id, action))
                curr_agent_ = curr_agent_.next_agent
                s = env_output.next_state
                done = env_output.done

        s_vs = np.zeros(n_agents)
        for ag_id, s_id, a in reversed(history):
            s_vs += all_vs[s_id]
            self.agents[ag_id].update_qn_(s_id, a, s_vs[ag_id])
