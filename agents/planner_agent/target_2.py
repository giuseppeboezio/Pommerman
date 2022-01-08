from pommerman import constants
import numpy as np
from pommerman.constants import Item
from utility import get_dangerous_positions


def get_positions(obs):

    dang_pos = []
    # collecting positions where there are bombs and positions affected by the blast strength
    for i in range(constants.BOARD_SIZE):
        for j in range(constants.BOARD_SIZE):
            if obs['bomb_life'][i,j] > 0:
                position = (i,j)
                dang_pos.append(position)
                blast_strength = obs['bomb_blast_strength'][position[0],position[1]] - 1
                board = obs['board']
                dang_pos = dang_pos + get_dangerous_positions(board, blast_strength, position)

    return dang_pos


def get_target_pos(distances):
    """Return the position with the minimum distance from the current position of the agent"""
    position = (-1,-1)
    value = np.inf
    for i in range(constants.BOARD_SIZE):
        for j in range(constants.BOARD_SIZE):
            if distances[i,j] < value:
                position = (i,j)
                value = distances[i,j]

    return position
