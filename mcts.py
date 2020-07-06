import numpy as np
from mcts_agent import MctsAgent
import itertools

from my_tictactoe import render_
from table_utils import TablePolicy


def self_play_(env_get_actions, env_model, init_state,
               n_players, n_actions, n_iters, n_eps, n_mcts):
    agents = []
    memory = TablePolicy(n_actions)
    for i in range(n_players):
        agent = MctsAgent(i, n_actions, memory)
        agents.append(agent)
    examples = []
    for i in range(n_iters):
        for e in range(n_eps):
            s = init_state
            agents_gen = itertools.cycle(agents)
            curr_agent_ = next(agents_gen)
            while True:
                print(f'agent {curr_agent_.ag_id} turn')
                render_(s)
                policy = curr_agent_.find_policy(env_get_actions, env_model,
                                                 s, agents, n_mcts)
                action = np.random.choice(len(policy), p=policy)
                next_s, rewards, done, next_id, message = env_model(s, action, curr_agent_.ag_id)
                curr_agent_ = next(agents_gen)
                assert next_id == curr_agent_.ag_id
                if done:
                    print("done")
                    render_(s)
                    print(action)
                    render_(next_s)
                    print(rewards)
                    print(message)
                    examples.append([curr_agent_.ag_id, s, policy, rewards])
                    break
                else:
                    s = next_s
        memory.train_(examples)
    return memory

    #     frac_win = pit(new_nnet, nnet)
    #     if frac_win > threshold:
    #         nnet = new_nnet


