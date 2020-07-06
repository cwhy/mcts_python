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
            = defaultdict(lambda: np.zeros(n_actions))
