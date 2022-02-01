from pommerman.constants import Item
from pommerman import constants
import numpy as np
from enum import Enum


# TODO check dependency problem
class KickDirection(Enum):
    Left = 0
    Right = 1
    Up = 2
    Down = 3


def count_power_ups(board1, board2):
    """ counting number of new power-up between two timesteps of the game"""
    items = {Item.ExtraBomb.value, Item.IncrRange.value, Item.Kick.value}
    pos_board1 = set()
    pos_board2 = set()
    for i in range(constants.BOARD_SIZE):
        for j in range(constants.BOARD_SIZE):
            if board1[i,j] in items:
                pos_board1.add((i,j))
            if board2[i,j] in items:
                pos_board2.add((i,j))
    # difference between the second set and the first set to add only positions where new power-ups have appeared
    new_pos_pow_ups = pos_board2 - pos_board1
    count = len(new_pos_pow_ups)
    return count


def get_target_opponent(opp_pos, distances):
    """return the position of the closest opponent that can be reached by my agent"""
    opponents_dis = {(i,j): distances[i,j] for (i,j) in opp_pos}
    opp = None
    dis_min = np.inf
    for (pos, dis) in opponents_dis.items():
        if dis < dis_min:
            opp = pos
            dis_min = dis
    return opp, dis_min


def get_admissible_pos(ag_pos, board):
    """Return positions where it is possible to place a bomb and kick it toward the opponent"""
    left = []
    right = []
    up = []
    down = []

    # left scan
    i = ag_pos[0]
    j = ag_pos[1] - 1
    while j >= 0 and board[i,j] == Item.Passage.value:
        left.append((i,j))
        j -= 1

    # right scan
    i = ag_pos[0]
    j = ag_pos[1] + 1
    while j < constants.BOARD_SIZE and board[i,j] == Item.Passage.value:
        right.append((i,j))
        j += 1

    # up scan
    i = ag_pos[0] - 1
    j = ag_pos[1]
    while i >= 0 and board[i,j] == Item.Passage.value:
        up.append((i,j))
        i -= 1

    # down scan
    i = ag_pos[0] + 1
    j = ag_pos[1]
    while i < constants.BOARD_SIZE and board[i,j] == Item.Passage.value:
        down.append((i,j))
        i += 1

    # remove last positions in all directions due to the needed space to kick the bomb
    if len(left) > 0:
        left.pop()
    if len(right) > 0:
        right.pop()
    if len(up) > 0:
        up.pop()
    if len(down) > 0:
        down.pop()

    positions = left + right + up + down
    return positions


def get_target_pos(dis_board, agent_pos, positions):
    """Return positions where to place a bomb and direction of kick"""
    distances = []
    for i in range(len(positions)):
        distances.append(dis_board[positions[i][0], positions[i][1]])

    target_pos = positions[np.argmin(distances)]
    # check the kick direction
    if target_pos[0] == agent_pos[0] and target_pos[1] < agent_pos[1]:
        direction = KickDirection.Right.value
    elif target_pos[0] == agent_pos[0] and target_pos[1] > agent_pos[1]:
        direction = KickDirection.Left.value
    elif target_pos[1] == agent_pos[1] and target_pos[0] < agent_pos[0]:
        direction = KickDirection.Down.value
    else:
        direction = KickDirection.Up.value

    return target_pos, direction


def get_close_target(position, board, direction):
    """Return next target position where to place a bomb to place a sequence of bombs"""


