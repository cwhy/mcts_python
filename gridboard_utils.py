import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Protocol, Callable
from config import h, player_symbols, Actions, Action


@dataclass
class StateBoard(Protocol):
    array: np.ndarray


def get_actions(state: StateBoard, agent_id: int) -> Actions:
    return np.where(state.array == -1)[0]


def render_(state: StateBoard) -> None:
    new_state_vector = []
    for value in state.array:
        if value == -1:
            new_state_vector.append(' ')
        else:
            new_state_vector.append(player_symbols[value])

    for i in range(0, h ** 2, h):
        print_grid_line_(new_state_vector, i)

    print(" " + "-" * (h * 4 + 1))
    print()


def print_grid_line_(grid, offset=0):
    print(" " + "-" * (h * 4 + 1))
    for i in range(h):
        # if grid[i + offset] == 0:
        #     print(" | " + " ", end='')
        # else:
        print(" | " + str(grid[i + offset]), end='')
    print(" |")


class CliAgent:
    def __init__(self, agent_id: int):
        self.ag_id = agent_id

    # noinspection PyMethodMayBeStatic
    def find_action(self, s: StateBoard, render: bool = False) -> Action:
        print("Current Game State:")
        render_(s)
        print("Position numbers:")
        for i in range(0, h ** 2, h):
            print_grid_line_(np.arange(0, h ** 2), i)

        print(" " + "-" * (h * 4 + 1))
        i = int(input("Enter your next move as position number:"))
        return i


def get_symmetries(state: StateBoard,
                   actions: Actions,
                   wrapper: Callable[[np.ndarray], StateBoard]) -> Tuple[
                                            List[StateBoard], List[Actions]]:
    # mirror, rotational
    board = state.array.reshape(h, h)
    board_a = actions.reshape(h, h)
    boards = []
    boards_a = []
    for i in range(1, 5):
        for j in True, False:
            new_board = np.rot90(board, i)
            new_board_a = np.rot90(board_a, i)
            if j:
                new_board = np.fliplr(new_board)
                new_board_a = np.fliplr(new_board_a)
            new_board = new_board.flatten()
            boards.append(wrapper(new_board))
            boards_a.append(new_board_a.flatten())
    return boards, boards_a


def rewards_winner_take_all(num: float, player: int):
    rewards = np.full(2, -num)
    rewards[player] += num * 2
    return rewards


def rewards_individual(num: float, player: int):
    rewards = np.full(2, 0)
    rewards[player] += num
    return rewards


def rewards_all(num: float):
    rewards = np.full(2, num)
    return rewards
