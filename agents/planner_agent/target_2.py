from pommerman import constants
import numpy as np
from pommerman.constants import Item
from utility import get_dangerous_positions


def get_positions(obs):

    dang_pos = []
    # collecting positions where there are bombs and positions affected by the blast strength
    for i in range(constants.BOARD_SIZE):
        for j in range(constants.BOARD_SIZE):
            if obs['board'][i,j] == Item.Bomb.value:
                position = (i,j)
                blast_strength = int(obs['bomb_blast_strength'][position[0],position[1]] - 1)
                board = obs['board']
                dang_pos = dang_pos + get_dangerous_positions(board, blast_strength, position)

    return dang_pos


def get_target_pos(distances, curr_position):
    """Return the position with the minimum distance from the current position of the agent"""
    dis_without_curr_pos = np.array(distances)
    # the agent is forced to move to reveal on the board the bomb it has just placed
    dis_without_curr_pos[curr_position[0], curr_position[1]] = np.inf
    position = (-1,-1)
    value = np.inf
    for i in range(constants.BOARD_SIZE):
        for j in range(constants.BOARD_SIZE):
            if dis_without_curr_pos[i,j] < value:
                position = (i,j)
                value = dis_without_curr_pos[i,j]

    return position
