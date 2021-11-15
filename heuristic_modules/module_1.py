import numpy as np
from playground.pommerman import constants


def get_feasible_pos(state):
    """Positions of the board where the agent can put a bomb"""
    mask = np.zeros((constants.BOARD_SIZE, constants.BOARD_SIZE))
    for i in range(constants.BOARD_SIZE):
        for j in range(constants.BOARD_SIZE):
            if state['board'][i, j] == 0:
                mask[i, j] = 1
    return mask


def destroyed_walls(state, position):
    count = 0
    positions = []
    blast_strength = state['blast_strength'] - 1
    # list of positions to check
    left = [(position[0], position[1] - j) for j in range(1, blast_strength + 1)]
    right = [(position[0], position[1] + j) for j in range(1, blast_strength + 1)]
    up = [(position[0] - j, position[1]) for j in range(1, blast_strength + 1)]
    down = [(position[0] + j, position[1]) for j in range(1, blast_strength + 1)]




def get_destroyed_boxes(state):
    """Board of destroyed walls putting the bomb in a feasible position"""
    bombs = np.zeros((constants.BOARD_SIZE, constants.BOARD_SIZE))
    mask = get_feasible_pos(state)
    for i in range(constants.BOARD_SIZE):
        for j in range(constants.BOARD_SIZE):
            if mask[i, j] == 1:
                bombs[i, j] = destroyed_walls(state, (i,j))
    return bombs


def get_distances(state):
    pass


def combine_masks(bombs, distances, alpha=1, beta=0.5):
    return bombs * alpha - distances * beta
