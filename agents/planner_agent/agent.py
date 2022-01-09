from pommerman.agents import BaseAgent
from pommerman.constants import Item
from pommerman import constants
from heuristic_modules.module_1 import get_distances
from enum import Enum
import numpy as np
import target_1 as tg_one
import target_2 as tg_two
import target_3 as tg_three
import copy
from utility import show_board


def generate_path(nodes, target_pos):
    """Provide the path to follow to reach the position"""

    path = []

    for node in nodes:
        position = node.get_position()
        if position[0] == target_pos[0] and position[1] == target_pos[1]:

            parent = node.get_parent()
            node_pth = node
            # obtaining the path using results of Dijkstra's algorithm
            while parent is not None:
                path.append(node_pth.get_position())
                node_pth = parent
                parent = node_pth.get_parent()

            break

    # reverse the element to get the path from the root node to the next one
    path.reverse()

    return path


def get_action(ag_pos, target_pos):
    """Return the action to reach target_pos from ag_pos"""

    if ag_pos[0] == target_pos[0] and ag_pos[1] < target_pos[1]:
        action = constants.Action.Right
    elif ag_pos[0] == target_pos[0] and ag_pos[1] > target_pos[1]:
        action = constants.Action.Left
    elif ag_pos[1] == target_pos[1] and ag_pos[0] < target_pos[0]:
        action = constants.Action.Down
    elif ag_pos[1] == target_pos[1] and ag_pos[0] > target_pos[0]:
        action = constants.Action.Up
    else:
        action = constants.Action.Bomb

    return action


def get_positions_power_up(board):
    """Return the positions of the power-up on the board"""
    positions = []
    for i in range(constants.BOARD_SIZE):
        for j in range(constants.BOARD_SIZE):
            if board[i,j] == Item.ExtraBomb.value or board[i,j] == Item.Kick.value or board[i,j] == Item.IncrRange.value:
                positions.append((i,j))

    return positions


def change_board(board, positions, value):
    """Update the board setting to rigid walls position where the agent could be killed"""
    new_board = np.array(board)
    for pos in positions:
        new_board[pos[0],pos[1]] = value

    return new_board


class Target(Enum):
    """Target of the agent"""

    Bomb = 1
    Safe = 2
    Collect = 3


# agent behaves in the following way:
# Target 1: put the bomb in the position where it is possible to destroy the highest number of wooden walls
# Target 2: choose a position where to avoid being killed by a bomb
# Target 3: choose the item to pick up according to a probability which depends on the distance from the object
class PlannerAgent(BaseAgent):

    def __init__(self):
        super(PlannerAgent, self).__init__()
        self.target = Target.Bomb.value  # target to achieve
        self.defined = False  # flag to establish if the target position has been found or not
        self.target_pos = None  # Target position to reach

    def act(self, obs, action_space):

        # print for debugging
        print("Information")
        print("-----------")
        print(f"Target : {self.target}")
        print(f"Target position: {self.target_pos}")
        print("Board")
        print(show_board(obs['board']))

        # the objective is putting a bomb
        if self.target == Target.Bomb.value:
            # the objective is finding the position where to put the bomb
            if not self.defined:
                distances, nodes = get_distances(obs)
                # obtaining position where it could be possible to put a bomb
                positions = tg_one.get_positions(nodes)
                # counting number of destroyed walls for each position
                self.target_pos = tg_one.get_target_position(obs, positions)
                self.defined = True

            # update the path
            distances, nodes = get_distances(obs)
            path = generate_path(nodes, self.target_pos)
            # at each step I check whether there is a path to reach the position or not
            if len(path) > 0:
                next_position = path[0]
                action = get_action(obs['position'], next_position)
            else:
                # the current position of the agent is the target position
                if obs['position'][0] == self.target_pos[0] and obs['position'][1] == self.target_pos[1]:
                    # TODO check whether for action bomb a value instead of an object is requested
                    action = constants.Action.Bomb.value
                    self.defined = False
                    self.target_pos = None
                    self.target = Target.Safe.value
                else:
                    # case in which it is no more possible to reach the target
                    action = constants.Action.Stop

        # looking for a safe position
        elif self.target == Target.Safe.value:
            if not self.defined:
                # find positions where the agent can be possibly killed
                dangerous_pos = tg_two.get_positions(obs)
                dangerous_pos = set(dangerous_pos)
                # create a board which takes into account possible dangerous positions
                new_board = change_board(obs['board'], dangerous_pos, Item.Rigid.value)
                new_obs = copy.copy(obs)
                new_obs['board'] = new_board
                # use Dijkstra's algorithm to find distances and keep the closest position
                distances, nodes = get_distances(new_obs)
                self.target_pos = tg_two.get_target_pos(distances, obs['position'])
                # generate the path to follow to reach that position
                self.defined = True

            # update the path
            distances, nodes = get_distances(obs)
            path = generate_path(nodes, self.target_pos)
            # at each step I check whether there is a path to reach the position or not
            if len(path) > 0:
                next_position = path[0]
                action = get_action(obs['position'], next_position)
            else:
                # the current position of the agent is the target position
                if obs['position'][0] == self.target_pos[0] and obs['position'][1] == self.target_pos[1]:
                    self.defined = False
                    self.target_pos = None
                    self.target = Target.Collect.value

                action = constants.Action.Stop

        # pick up a power-up
        else:
            if not self.defined:
                pow_up_pos = get_positions_power_up(obs['board'])
                # check whether there is at least a power-up to pick
                if len(pow_up_pos) > 0:
                    # change the board allowing to reach power-ups
                    new_board = change_board(obs['board'], pow_up_pos, Item.Passage)
                    new_obs = copy.copy(obs)
                    new_obs['board'] = new_board
                    # execute Dijkstra's algorithm to get distances
                    distances, nodes = get_distances(new_obs)
                    self.target_pos = tg_three.get_target_collect(distances, set(pow_up_pos))
                    self.defined = True
                else:
                    # there are no power-up, the agent waits for one of them
                    return constants.Action.Stop

            # change the board allowing to reach power-ups
            new_board = change_board(obs['board'], [self.target_pos], Item.Passage)
            new_obs = copy.copy(obs)
            new_obs['board'] = new_board
            # execute Dijkstra's algorithm to get distances
            distances, nodes = get_distances(new_obs)
            path = generate_path(nodes, self.target_pos)
            # at each step I check whether there is a path to reach the position or not
            if len(path) > 0:
                next_position = path[0]
                action = get_action(obs['position'], next_position)
            else:
                # the current position of the agent is the target position
                if obs['position'][0] == self.target_pos[0] and obs['position'][1] == self.target_pos[1]:
                    self.defined = False
                    self.target = Target.Bomb.value

                action = constants.Action.Stop

        print(action)

        return action
