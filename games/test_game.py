from env_utils import pit
from games import envs

env_name = 'Connect4'
# env_name = 'TicTacToe'
env = envs[env_name]
CliAgent = env.cli_agent
pit(env, [CliAgent(0), CliAgent(1)], render=True)
