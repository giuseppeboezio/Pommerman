from pommerman import constants
import numpy as np
from pommerman.constants import Item
from utility import get_dangerous_positions
from utility import ExplosionField


def get_positions(obs):
    """Positions affected by the bomb according to its blast strength"""

    expl_fields = []  # it is a list of ExplosionField objects
    # collecting positions where there are bombs and positions affected by the blast strength
    for i in range(constants.BOARD_SIZE):
        for j in range(constants.BOARD_SIZE):
            if obs['board'][i, j] == Item.Bomb.value:
                position = (i, j)
                blast_strength = int(obs['bomb_blast_strength'][position[0], position[1]] - 1)
                board = obs['board']
                life = obs['bomb_life'][position[0], position[1]]
                dang_pos = get_dangerous_positions(board, blast_strength, position)
                field = ExplosionField(position, life)
                field.set_danger_pos(dang_pos)
                expl_fields.append(field)

    return expl_fields


def get_target_pos(distances, curr_position, board):
    """Return the position with the minimum distance from the current position of the agent"""
    dis_without_curr_pos = np.array(distances)
    # the agent is forced to move to reveal on the board the bomb it has just placed
    dis_without_curr_pos[curr_position[0], curr_position[1]] = np.inf
    position = (curr_position[0], curr_position[1])
    value = np.inf
    # four possible moves
    positions = [(curr_position[0] - 1, curr_position[1]), (curr_position[0], curr_position[1] - 1)
        , (curr_position[0] + 1, curr_position[1]), (curr_position[0], curr_position[1] + 1)]
    positions = [(r,c) for (r,c) in positions if 0 <= r < constants.BOARD_SIZE and 0 <= c < constants.BOARD_SIZE]
    for pos in positions:
        if dis_without_curr_pos[pos[0], pos[1]] < value:
            # it is important to check whether the agent can be stuck or not
            neig_pos = [(pos[0] - 1, pos[1]), (pos[0], pos[1] - 1), (pos[0] + 1, pos[1]), (pos[0], pos[1] + 1)]
            neig_pos = [(r, c) for (r, c) in neig_pos if 0 <= r < constants.BOARD_SIZE and 0 <= c < constants.BOARD_SIZE]
            # items contained in neighbourhood positions
            values = np.array([board[p[0], p[1]] for p in neig_pos])
            # at least one neighbours must be a passage to have a free escape
            if len(np.argwhere(values == Item.Passage.value)) > 0:
                position = tuple(pos)
                value = dis_without_curr_pos[position[0], position[1]]

    return position
