import numpy as np
from mcts_agent import MctsAgent
import itertools


def self_play(env_get_actions, env_model, init_state, n_players, n_actions, n_iters, n_eps, n_mcts):
    agents = []
    memory = 1  # TODO implement memory and NNmemory
    for i in range(n_players):
        agent = MctsAgent(i, n_actions, memory)
        agents.append(agent)
    examples = []
    for i in range(n_iters):
        for e in range(n_eps):
            s = init_state
            curr_agent_ = agents[0]
            agents_gen = itertools.cycle(agents)
            while True:
                policy = curr_agent_.find_policy(env_get_actions, env_model, s, agents_gen, n_mcts)
                action = np.random.choice(len(policy), p=policy)
                next_s, reward, done, next_id, message = env_model(s, action, curr_agent_.ag_id)
                curr_agent_ = next(agents_gen)
                assert next_id == curr_agent_.ag_id
                if done:
                    examples.append([curr_agent_.ag_id, s, policy, reward])  #TODO add multi-agent reward for env_model
                    break
        memory.train_(examples)
    return memory

    #     frac_win = pit(new_nnet, nnet)
    #     if frac_win > threshold:
    #         nnet = new_nnet


