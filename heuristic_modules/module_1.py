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
    """Number of walls that can be destroyed putting a bomb in position"""
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
        if pos[0] < 0 or pos[1] < 0 or pos[0] > constants.BOARD_SIZE - 1 or pos[1] > constants.BOARD_SIZE - 1:
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


class PriorityQueue:

    def __init__(self):
        self.queue = []

    def enqueue(self, elem):
        """Enqueue an element in the form (priority, node)"""
        self.queue.append(elem)
        sorted(self.queue, key=lambda x: x[0])

    def dequeue(self):
        return self.queue.pop(0)

    def change_priority(self, item, priority):
        items = [elem[1] for elem in self.queue]
        pos = items.index(item)
        del self.queue[pos]
        self.queue.append((priority,item))
        sorted(self.queue, key=lambda x: x[0])


    def is_empty(self):
        return len(self.queue) == 0

    def has(self, item):
        items = [elem[1] for elem in self.queue]
        return item in items


class Node:

    def __init__(self, position):
        self.position = position
        self.parent = None
        self.neighbours = []
        # distance from the root node, infinity if they are not connected
        self.distance = np.inf
        self.discovered = False

    def add_neighbour(self, neighbour):
        self.neighbours.append(neighbour)

    def get_distance(self):
        return self.distance

    def set_distance(self, distance):
        self.distance = distance

    def set_parent(self, node):
        self.parent = node

    def get_parent(self):
        return self.parent

    def set_discovered(self):
        self.discovered = True

    def is_discovered(self):
        return self.discovered

    def get_position(self):
        return self.position

    def get_unvisited_neighbours(self):

        unvisited = [node for node in self.neighbours if not node.is_discovered()]
        return unvisited


# graph of the board
class Board:

    def __init__(self, board):

        self.nodes = []
        self.root = None

        # dictionary to store the nodes
        nodes_dict = {}
        for i in range(constants.BOARD_SIZE):
            for j in range(constants.BOARD_SIZE):
                if board[i,j] == 0 or board[i,j] == 10:
                    nodes_dict[(i,j)] = Node((i,j))

        # creation of nodes and add to each one its neighbours
        for i in range(constants.BOARD_SIZE):
            for j in range(constants.BOARD_SIZE):

                # check whether the position corresponds to a passage or to our agent
                if board[i,j] == 0 or board[i,j] == 10:

                    neighbours = [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]

                    # remove positions outside the board or for which there is no passage
                    valid_neighbours = []
                    for elem in neighbours:

                        if 0 <= elem[0] < constants.BOARD_SIZE and elem[1] >= 0 and \
                                elem[1] < constants.BOARD_SIZE and board[elem[0],elem[1]] == 0:
                            valid_neighbours.append(elem)

                    for elem in valid_neighbours:
                        nodes_dict[(i,j)].add_neighbour(nodes_dict[(elem[0],elem[1])])

                    self.nodes.append(nodes_dict[(i,j)])

                    # if in the position there is our agent it means that this node is the root
                    if board[i,j] == 10:
                        self.root = nodes_dict[(i,j)]

    def get_nodes(self):
        return self.nodes

    def get_root_node(self):
        return self.root


def get_distances(state):

    obs = state['board']
    board_graph = Board(obs)

    # dijkstra's algorithm
    # list of nodes to expand
    priority_q = PriorityQueue()

    start = board_graph.get_root_node()
    start.set_distance(0)

    # adding root to the queue
    priority_q.enqueue((0,start))

    while not priority_q.is_empty():
        current_node = priority_q.dequeue()[1]
        current_node.set_discovered()

        for neighbour in current_node.get_unvisited_neighbours():
            min_distance = min(neighbour.get_distance(), current_node.get_distance() + 1)

            if min_distance != neighbour.get_distance():
                neighbour.set_distance(min_distance)
                neighbour.set_parent(current_node)
                if priority_q.has(neighbour):
                    priority_q.change_priority(neighbour, min_distance)

            if not priority_q.has(neighbour):
                priority_q.enqueue((neighbour.get_distance(), neighbour))

    distances = np.full((constants.BOARD_SIZE,constants.BOARD_SIZE), np.inf)

    nodes = board_graph.get_nodes()
    for node in nodes:
        pos = node.get_position()
        distances[pos[0],pos[1]] = node.get_distance()

    return distances, nodes


def combine_masks(bombs, distances, alpha=1, beta=0.5):
    return bombs * alpha - distances * beta
