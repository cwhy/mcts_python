from config import h
from env_utils import pit, RandomAgent
from games import envs
from games.web_agent import WebAgent

env_name = 'WuZiQi'
# web_agent0 = WebAgent(h, h).get_actor(0)
web_agent1 = WebAgent(h, h).get_actor(1)
# env_name = 'TicTacToe'
env = envs[env_name]
CliAgent = env.cli_agent
pit(env, [RandomAgent(0, env).find_action, web_agent1], render=True)
