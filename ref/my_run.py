import numpy as np
from my_tictactoe import h, get_available_actions, init_state, model, render_, \
    player_symbols

curr_player_ = 0
actions = h ** 2
state_ = init_state

while True:
    render_(state_)
    action = np.random.choice(get_available_actions(state_))
    state_, rewards, done, curr_player_, message = model(state_, action, curr_player_)

    if done:
        render_(state_)
        print(dict(zip(player_symbols, rewards)))
        print(message)
        break
