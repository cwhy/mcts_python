from collections import defaultdict
from typing import List, Tuple, Dict, Optional, NamedTuple

import numpy as np
import torch

from config import Action, StateID, State, Env
from mcts_agent import MctsAgent
from memory_utils import NNMemory


class Mcts:
    def __init__(self,
                 n_mcts: int,
                 env: Env,
                 max_depth: Optional[int] = None):
        self.n_mcts = n_mcts
        self.states_: List[State] = []
        self.state_ids: Dict[bytes, StateID] = {}
        self.env = env
        self.max_depth = max_depth
        self.agents: List[MctsAgent] = [
            MctsAgent(i, self.states_, self.state_ids, env)
            for i in range(env.n_agents)
        ]

    def search_(self, memory: NNMemory,
                start_state: State,
                ag_id: int):
        env_model = self.env.model
        curr_agent_ = self.agents[ag_id]
        n_agents = len(self.agents)
        # TODO fix all_vs to list
        all_vs: Dict[StateID, np.ndarray] = defaultdict(lambda: np.zeros(n_agents))
        history_: List[Tuple[int, StateID, Action]] = []

        s_ = start_state
        done_ = False
        depth_ = 0
        while not (done_ or (self.max_depth is not None and depth_ > self.max_depth)):
            sb = s_.tobytes()
            if sb not in self.state_ids:
                v = np.zeros(n_agents)
                v[curr_agent_.ag_id] += memory.get_v(s_, curr_agent_.ag_id)
                self.states_.append(s_)
                s_id = self.state_ids[sb] = len(self.states_) - 1
                all_vs[s_id] = v
                break
            else:
                s_id = self.state_ids[sb]
                action = curr_agent_.selection(s_id,
                                               memory.get_p(s_, curr_agent_.ag_id))
                env_output = env_model(s_, action, curr_agent_.ag_id, render=False)
                all_vs[s_id] = env_output.rewards
                history_.append((curr_agent_.ag_id, s_id, action))
                curr_agent_ = self.agents[env_output.next_agent_id]
                s_ = env_output.next_state
                depth_ += 1
                done_ = env_output.done

        s_vs = np.zeros(n_agents)
        for ag_id, s_id, a in reversed(history_):
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

    def self_play(self, memory_: NNMemory) -> Tuple[np.ndarray,
                                                    np.ndarray,
                                                    np.ndarray,
                                                    np.ndarray]:
        ag_ids = []
        states = []
        policies = []
        s_ = self.env.init_state
        curr_agent_ = self.agents[0]
        total_rewards = np.zeros(len(self.agents))
        done_ = False
        depth_ = 0
        while not (done_ or (self.max_depth is not None and depth_ > self.max_depth)):
            for _ in range(self.n_mcts):
                self.search_(memory_, s_, curr_agent_.ag_id)

            policy = curr_agent_.find_policy(s_)
            action = np.random.choice(len(policy), p=policy)
            env_output = self.env.model(s_, action, curr_agent_.ag_id, render=False)
            total_rewards += env_output.rewards

            if self.env.get_symmetries is not None:
                _states, _policies = self.env.get_symmetries(s_, policy)
                states += _states
                policies += _policies
                ag_ids += len(_states) * [curr_agent_.ag_id]
            else:
                states.append(s_)
                policies.append(policy)
                ag_ids.append(curr_agent_.ag_id)

            curr_agent_ = self.agents[env_output.next_agent_id]
            s_ = env_output.next_state
            depth_ += 1
            done_ = env_output.done
        return (np.stack(ag_ids, axis=0)[:, np.newaxis],
                np.stack(states, axis=0),
                np.stack(policies, axis=0),
                np.repeat(total_rewards[np.newaxis, :], len(states), axis=0))

# agent_ids: np.ndarray  int, (length,)
# states: np.ndarray  int/float, (length, dim_states)
# policies: np.ndarray  float, (length, n_possible_policies)
# total_rewards: np.ndarray  float, (length, n_agents)
