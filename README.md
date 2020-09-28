## MCTS (actually PUCT) in Python

### Demo Games
* TicTacToe
* Reversi
* Wuziqi (Gomoku)
* Wuziqi-1swap ([五子棋一手交換規則](https://zh.wikipedia.org/wiki/%E4%BA%94%E5%AD%90%E6%A3%8B))
### Supports
* Custom environment with clear APIs
    - Examples are in `/games`
* Arbitrary number of agents with per-agent rewards
### Instructions
* run `run_mcts.py` to start
* look up `config.py` to change game/configurations
### Key parameters of MCTS(PUCT)
* `n_iters`: the larger the more clever neural network will be,
 will increase training time linearly.
* `n_eps`: the larger the more robust the training will be,
 will increase training time linearly
* `n_mcts`: the larger the larger the more brute-force search samples will be,
 will increase training time and testing time polynomially

### Requirement
* Python 3.7 +
* Refer to `requirement.txt`
