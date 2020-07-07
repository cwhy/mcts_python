import itertools
from typing import List, Tuple

import numpy as np

from mcts_agent import MctsAgent
from mcts_agent import StateID, Action
from table_utils import TablePolicy


class Mcts:
    def __init__(self, n_actions: int, n_players: int, n_mcts: int,
                 get_available_actions):
        self.n_mcts = n_mcts
        self.agents = []
        self.memory_ = TablePolicy(n_actions)
        for i in range(n_players):
            agent = MctsAgent(i, n_actions, self.memory_, get_available_actions)
            self.agents.append(agent)

        for agent in self.agents:
            agent.assign_next_(self.agents[(agent.ag_id + 1) % n_players])

    def self_play_(self, env_model, init_state, n_iters, n_eps):
        examples = []
        for _ in range(n_iters):
            for _ in range(n_eps):
                s = init_state
                agents_gen = itertools.cycle(self.agents)
                curr_agent_ = next(agents_gen)
                while True:
                    for _ in range(self.n_mcts):
                        search_(env_model, s, self.agents, curr_agent_.ag_id)

                    policy = curr_agent_.find_policy(s)
                    action = np.random.choice(len(policy), p=policy)
                    next_s, rewards, done, next_id, message = env_model(s,
                                                                        action,
                                                                        curr_agent_.ag_id)
                    curr_agent_ = next(agents_gen)
                    assert next_id == curr_agent_.ag_id
                    if done:
                        examples.append(
                            [curr_agent_.ag_id, s, policy, rewards])
                        break
                    else:
                        s = next_s
            self.memory_.train_(examples)

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
            v = np.zeros(n_agents)  # self.memory_.get_v(s)
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
