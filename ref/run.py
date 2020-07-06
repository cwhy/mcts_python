from ref.tictactoe import TicTacToeEnv
import numpy as np

user = 0
done = False
reward = 0

# Reset the env before playing

player_symbols = ['x', 'o']
h = 3
env = TicTacToeEnv(player_symbols=player_symbols, board_size=h, win_size=h)
state = env.reset()
curr_player_ = 0
actions = h ** 2

while True:
    env.render()
    avail_actions = np.where(state == -1)[0]
    action = np.random.choice(avail_actions)
    state, reward, done, infos = env.step(action, curr_player_)
    curr_player_ = infos["next_player"]

    if done:
        if reward == 10:
            print("draw")
        env.render()
        print(infos)
        break
