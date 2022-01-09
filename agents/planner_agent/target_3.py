from pommerman import constants
from scipy.special import softmax
import numpy as np


def get_target_collect(distances, positions):
    """Target position for the third target of the agent"""
    ordered_pos = []
    dis_items = []
    for i in range(constants.BOARD_SIZE):
        for j in range(constants.BOARD_SIZE):
            # position of a power-up which can be reach by the agent starting from its current position
            if (i,j) in positions and distances[i,j] > 0 and distances[i,j] != np.inf:
                ordered_pos.append((i,j))
                dis_items.append(distances[i,j])

    '''apply softmax to choose the power-up according to the smallest distance
    soft_dis = softmax(dis_items)
    soft_dis = 1 - soft_dis
    soft_dis = softmax(soft_dis)
    idx = np.random.choice(range(len(ordered_pos)), p=soft_dis)
    target_pos = ordered_pos[idx]'''

    # check whether there are any power-up which can be reached without destroying wooden walls
    if len(dis_items) > 0:
        target_pos = ordered_pos[np.argmin(dis_items)]
    else:
        target_pos = None

    return target_pos
