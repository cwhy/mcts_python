import numpy as np


def ucb(q: float, c_puct: float, p: float, n: float, na: float, temperature=1):
    return q + c_puct * p * np.sqrt(n) / (1 + na)


def ucb_all(qs: np.array,
            c_puct_normed_by_sum: float,
            ps: np.array,
            nas: np.array):
    return qs + c_puct_normed_by_sum * ps / (1 + nas)
