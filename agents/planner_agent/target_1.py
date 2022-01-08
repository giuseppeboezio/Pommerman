import numpy as np
from pommerman.constants import Item
from utility import get_dangerous_positions
from scipy.special import softmax


def get_positions(nodes):
    """Positions where the agent can put a bomb"""

    conn_nodes = [node for node in nodes if node.get_distance() != np.inf]
    positions = []
    for node in conn_nodes:
        positions.append(node.get_position())

    return positions


def destroyed_walls(obs, position):

    """Number of destroyed wooden walls placing a bomb in position"""
    blast_strength = obs['blast_strength'] - 1
    board = obs['board']
    dangerous_pos = get_dangerous_positions(board, blast_strength, position)
    # positions which correspond to wooden walls
    walls_pos = [pos for pos in dangerous_pos if board[pos[0],pos[1]] == Item.Wood.value]
    num = len(walls_pos)

    return num


def get_target_position(obs, positions):
    """Return the target position, i.e. the position where putting a bomb is possible to
    destroy the highest number of wooden walls"""

    num_des_walls = []

    for i in range(len(positions)):
        num_des_walls.append(destroyed_walls(obs, positions[i]))

    # choose the target position
    target_position = positions[np.argmax(num_des_walls)]

    return target_position
