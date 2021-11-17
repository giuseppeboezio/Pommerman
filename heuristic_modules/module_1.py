import numpy as np
from playground.pommerman import constants
from queue import PriorityQueue

def get_feasible_pos(state):
    """Positions of the board where the agent can put a bomb"""
    mask = np.zeros((constants.BOARD_SIZE, constants.BOARD_SIZE))
    for i in range(constants.BOARD_SIZE):
        for j in range(constants.BOARD_SIZE):
            if state['board'][i, j] == 0:
                mask[i, j] = 1
    return mask


# TODO test this function
def destroyed_walls(state, position):
    """Number of walls that can be destroyed putting a bomb in position"""
    count = 0
    positions = []
    blast_strength = state['blast_strength'] - 1
    # list of positions to check
    left = [(position[0], position[1] - j) for j in range(1, blast_strength + 1)]
    right = [(position[0], position[1] + j) for j in range(1, blast_strength + 1)]
    up = [(position[0] - j, position[1]) for j in range(1, blast_strength + 1)]
    down = [(position[0] + j, position[1]) for j in range(1, blast_strength + 1)]
    positions = positions + left + right + up + down
    # Removing positions outside the board
    for pos in positions:
        if pos[0] < 0 or pos[1] < 0:
            positions.remove(pos)
    # positions where there are destructible walls
    board = state['board']
    positions = [pos for pos in positions if board[pos[0],pos[1]] == 2]
    # left direction
    left = [pos for pos in positions if pos[1] < position[1]]
    for elem in left:
        # checking whether the element on the position is a rigid wall
        if board[elem[0], elem[1]] == 1:
            remove_list = [pos for pos in left if pos[1] < elem[1]]
            # removing positions beyond the rigid wall because they cannot be reached by the flames
            for item in remove_list:
                left.remove(item)
    # right direction
    right = [pos for pos in positions if pos[1] > position[1]]
    for elem in right:
        # checking whether the element on the position is a rigid wall
        if board[elem[0], elem[1]] == 1:
            remove_list = [pos for pos in right if pos[1] > elem[1]]
            # removing positions beyond the rigid wall because they cannot be reached by the flames
            for item in remove_list:
                right.remove(item)
    # up direction
    up = [pos for pos in positions if pos[0] < position[0]]
    for elem in up:
        # checking whether the element on the position is a rigid wall
        if board[elem[0], elem[1]] == 1:
            remove_list = [pos for pos in up if pos[0] < elem[0]]
            # removing positions beyond the rigid wall because they cannot be reached by the flames
            for item in remove_list:
                up.remove(item)
    # down direction
    down = [pos for pos in positions if pos[0] > position[0]]
    for elem in down:
        # checking whether the element on the position is a rigid wall
        if board[elem[0], elem[1]] == 1:
            remove_list = [pos for pos in down if pos[0] > elem[0]]
            # removing positions beyond the rigid wall because they cannot be reached by the flames
            for item in remove_list:
                down.remove(item)
    # removing positions that do not correspond to wooden walls
    positions = left + right + up + down
    # keep only positions where there are wooden walls
    positions = [pos for pos in positions if board[pos[0],pos[1]] == 2]
    count = len(positions)
    return count


def get_destroyed_boxes(state):
    """Board of destroyed walls putting the bomb in a feasible position"""
    bombs = np.zeros((constants.BOARD_SIZE, constants.BOARD_SIZE))
    mask = get_feasible_pos(state)
    for i in range(constants.BOARD_SIZE):
        for j in range(constants.BOARD_SIZE):
            if mask[i, j] == 1:
                bombs[i, j] = destroyed_walls(state, (i,j))
    return bombs


# dijkstra's algorithm for grids
def dijkstra(board, start, end):


    pass


class Node:

    def __init__(self, position):
        self.position = position
        self.neighbours = []
        # distance from the root node, -1 means infinity
        self.distance = -1
        self.parent = None

    def add_neighbour(self, neighbour):
        self.neighbours.append(neighbour)

    def set_parent(self, parent):
        self.parent = parent

    def get_unvisited_neighbours(self):

        pass


# graph of the board
class Board:

    def __init__(self, board):

        self.nodes = []

        for i in range(constants.BOARD_SIZE):
            for j in range(len(constants.BOARD_SIZE)):

                # check whether the position corresponds to a passage or to our agent
                if board[i,j] == 0 or board[i,j] == 10:

                    neighbours = [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]

                    # remove positions outside the board or for which there is no passage
                    for elem in neighbours:
                        if elem[0] < 0 or elem[1] < 0 or board[elem[0],elem[1]] != 0:
                            neighbours.remove(elem)

                    node = Node((i,j))

                    # adding neighbours of the node
                    for elem in neighbours:
                        node.add_neighbour(Node((elem[0],elem[1])))

                    self.nodes.append(node)

    def get_nodes(self):
        return self.nodes


def get_distances(state):

    obs = state['board']
    board_graph = Board(obs)


    pass

def combine_masks(bombs, distances, alpha=1, beta=0.5):
    return bombs * alpha - distances * beta
