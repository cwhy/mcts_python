import itertools
import numpy as np


def self_play(env_model, init_state,
               agents, memory, n_iters, n_eps):
    examples = []
    for _ in range(n_iters):
        for _ in range(n_eps):
            s = init_state
            agents_gen = itertools.cycle(agents)
            curr_agent_ = next(agents_gen)
            while True:
                policy = curr_agent_.find_policy(s)
                action = np.random.choice(len(policy), p=policy)
                next_s, rewards, done, next_id, message = env_model(s, action, curr_agent_.ag_id)
                curr_agent_ = next(agents_gen)
                assert next_id == curr_agent_.ag_id
                if done:
                    examples.append([curr_agent_.ag_id, s, policy, rewards])
                    break
                else:
                    s = next_s
        memory.train_(examples)
    return memory, agents



