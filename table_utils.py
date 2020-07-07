from collections import defaultdict
from enum import Enum, auto
from typing import NamedTuple, Dict, Set
import numpy as np

from math_calc import ucb, ucb_all

Action = int
StateID = int
c_puct = 1.0


class TablePolicy:
    def __init__(self, n_actions: int):
        self._p_inits: Dict[StateID, np.ndarray] \
            = defaultdict(lambda: np.ones(n_actions)/n_actions)

    def train_(self, examples):
        pass

    def get_p(self, s_id: int):
        return self._p_inits[s_id]
