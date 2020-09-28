from games.my_wuziqi import wuziqi_env, wzq_net
from games.my_reversi import reversi_env, reversi_net
from games.my_tictactoe import ttt_env, ttt_net
from games.my_wuziqi_1swap import wuziqi_env as wuziqi_1swap_env, wzq_net as wzq_1swap_net


envs = {
    'TicTacToe': ttt_env,
    'Reversi': reversi_env,
    'WuZiQi': wuziqi_env,
    'WuZiQi_1swap': wuziqi_1swap_env,
}

nets = {
    'TicTacToe': ttt_net,
    'Reversi': reversi_net,
    'WuZiQi': wzq_net,
    'WuZiQi_1swap': wzq_1swap_net
}

cli_agents = {
    'TicTacToe': my_wuziqi.CliAgent,
    'Reversi': my_reversi.CliAgent,
    'WuZiQi': my_wuziqi.CliAgent,
    'WuZiQi_1swap': my_wuziqi_1swap.CliAgent
}
