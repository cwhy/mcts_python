import numpy as np
from my_tictactoe import h, get_available_actions, init_state, model, render_

curr_player_ = 0
actions = h ** 2
state_ = init_state

while True:
    render_(state_)
    action = np.random.choice(get_available_actions(state_))
    state_, reward, done, curr_player_, message = model(state_, action, curr_player_)

    if done:
        render_(state_)
        print(message)
        break
