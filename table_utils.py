from collections import defaultdict
from typing import Dict

import numpy as np

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
