from pommerman import constants
from scipy.special import softmax
import numpy as np


def get_target_collect(distances, positions):
    """Target position for the third target of the agent"""
    ordered_pos = []
    dis_items = []
    for i in range(constants.BOARD_SIZE):
        for j in range(constants.BOARD_SIZE):
            # position of a reachable power-up
            if (i,j) in positions and distances[i,j] > 0:
                ordered_pos.append((i,j))
                dis_items.append(distances[i,j])

    # apply softmax to choose the power-up according to the smallest distance
    soft_dis = softmax(dis_items)
    soft_dis = 1 - soft_dis
    idx = np.random.choice(np.range(len(ordered_pos)), p=soft_dis)
    target_pos = ordered_pos[idx]

    return target_pos
