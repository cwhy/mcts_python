import itertools
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from mcts_agent import MctsAgent
from mcts_agent import StateID, Action
from memory_utils import NNPolicy
from my_tictactoe import ttt_net, get_symmetries

device = 'cpu'


class Mcts:
    def __init__(self,
                 n_actions: int,
                 n_agents: int,
                 n_mcts: int,
                 get_available_actions):
        self.n_mcts = n_mcts
        self.agents = []
        self.memory_ = NNPolicy(ttt_net, device=device)
        for i in range(n_agents):
            agent = MctsAgent(i, n_actions, self.memory_, get_available_actions)
            self.agents.append(agent)

        for agent in self.agents:
            agent.assign_next_(self.agents[(agent.ag_id + 1) % n_agents])

    def self_play_(self, env_model, init_state, n_eps, n_iters):
        self.memory_.clear_()
        for _ in tqdm(range(n_iters)):
            for _ in range(n_eps):
                s = init_state
                agents_gen = itertools.cycle(self.agents)
                curr_agent_ = next(agents_gen)
                done = False
                total_rewards = np.zeros(len(self.agents))
                while not done:
                    for _ in range(self.n_mcts):
                        search_(env_model, s, self.agents, curr_agent_.ag_id)

                    policy = curr_agent_.find_policy(s)
                    action = np.random.choice(len(policy), p=policy)
                    next_s, rewards, done, next_id, message = env_model(s,
                                                                        action,
                                                                        curr_agent_.ag_id)
                    total_rewards += np.array(
                        [rewards[i] for i, _ in enumerate(self.agents)])
                    self.memory_.add_with_symmetry(curr_agent_.ag_id, s, policy, get_symmetries)
                    curr_agent_ = next(agents_gen)
                    assert next_id == curr_agent_.ag_id
                    s = next_s
                self.memory_.assign_values_(total_rewards)
            self.memory_.train_()

    def get_agent_decision_fn(self, ag_id: int, env_model):
        def decision(s, render=False):
            for _ in range(self.n_mcts):
                search_(env_model, s, self.agents, ag_id)
            return self.agents[ag_id].find_action(s, render)

        return decision


def search_(env_model, state, agents: List[MctsAgent], ag_id: int):
    curr_agent_ = agents[ag_id]
    n_agents = len(agents)
    s = state
    all_vs = []
    history: List[Tuple[int, StateID, Action]] = []
    while True:
        sb = s.tobytes()
        if sb not in curr_agent_.state_ids:
            v = np.zeros(n_agents)
            v[curr_agent_.ag_id] += curr_agent_.memory_.get_v(s, curr_agent_.ag_id)
            curr_agent_.states_.append(s)
            curr_agent_.state_ids[sb] = len(curr_agent_.states_) - 1
            all_vs.append(v)
            break
        else:
            sb = s.tobytes()
            s_id = curr_agent_.state_ids[sb]
            action = curr_agent_.selection(s_id)
            s, rewards, done, next_id, _ = env_model(s,
                                                     action,
                                                     curr_agent_.ag_id)
            if done:
                all_vs.append(np.array([rewards[i] for i in range(n_agents)]))
                break
            else:
                all_vs.append(np.array([rewards[i] for i in range(n_agents)]))
                history.append(
                    (curr_agent_.ag_id, s_id, action))
                curr_agent_ = curr_agent_.next_agent

    all_v = all_vs.pop(-1)
    for ag_id, s_id, a in reversed(history):
        all_v += all_vs.pop(-1)
        agents[ag_id].update_qn_(s_id, a, all_v[ag_id])
    assert len(all_vs) == 0
