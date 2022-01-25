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
    """Provide the path (list of positions) to follow to reach the position"""

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


def is_path_safe(path, explosion_fields):
    """Return true whether the number of steps of the path is less than the minimum life
    of the bombs whose affected positions are in the path, false otherwise"""

    # list of explosion field objects which affect the path
    path_set = set(path)
    obj_list = []
    for i in range(len(explosion_fields)):
        affected_pos = set(explosion_fields[i].get_dang_pos())
        common_pos = path_set.intersection(affected_pos)
        # check whether there are some positions affected by the bomb
        if len(common_pos) > 0:
            obj_list.append(explosion_fields[i])
    # check whether there are bombs affecting the path
    if len(obj_list) == 0:
        safe = True
    else:
        lives = np.array([e.get_life() for e in explosion_fields])
        life = np.min(lives)
        # decide whether the path is safe
        if life > len(path):
            safe = True
        else:
            safe = False
    return safe


def is_pos_safe(target, dang_pos):
    """Return true whether there aren't bombs which could kill the agent in the target position, false otherwise"""
    dang_pos_list = []
    for i in range(len(dang_pos)):
        dang_pos_list += dang_pos[i].get_dang_pos()
    dang_pos_list = set(dang_pos_list)
    if target in dang_pos_list:
        safe = False
    else:
        safe = True
    return safe


def get_immediate_expl_pos(explosion_fields):
    """Return the positions affected by bombs in the next timestep"""
    positions = []
    for i in range(len(explosion_fields)):
        life = explosion_fields[i].get_life()
        if life == 1.0:
            dang_pos = explosion_fields[i].get_dang_pos()
            positions += dang_pos
    positions = set(positions)
    return positions


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

        action = None
        corrective_strategy = True

        # used to switch among more sub strategies before deciding next action
        while corrective_strategy:

            # print for debugging
            print("Information")
            print("-----------")
            print(f"Target : {self.target}")
            print(f"Target position: {self.target_pos}")
            print("Board")
            print(show_board(obs['board']))

            # the objective is putting a bomb
            if self.target == Target.Bomb.value:

                # places the bomb even if the agents does not have ammo waiting for an explosion
                if not self.defined:

                    # use Dijkstra's algorithm to find distances
                    distances, nodes = get_distances(obs)
                    # obtaining position where it could be possible to put a bomb
                    positions = tg_one.get_positions(nodes)
                    # counting number of destroyed walls for each position
                    self.target_pos = tg_one.get_target_position(obs, positions)
                    self.defined = True

                # use Dijkstra's algorithm to find distances
                distances, nodes = get_distances(obs)
                path = generate_path(nodes, self.target_pos)
                # at each step I check whether there is a path to reach the position or not
                if len(path) > 0:
                    # general concept: the bomb is dangerous only if it affects agent path
                    # and if it explodes before the agent reaches the target position
                    # obtaining positions of bomb and related explosion field
                    dangerous_pos = tg_two.get_positions(obs)
                    safe = is_path_safe(path, dangerous_pos)
                    if safe:
                        next_position = path[0]
                        action = get_action(obs['position'], next_position)
                        corrective_strategy = False
                    else:
                        # in case the path is not safe the agent moves toward a safe position
                        self.defined = False
                        self.target_pos = None
                        self.target = Target.Safe.value

                else:
                    # the current position of the agent is the target position
                    if obs['position'][0] == self.target_pos[0] and obs['position'][1] == self.target_pos[1]:
                        # check whether the agent has enough ammo
                        if obs['ammo'] > 0:
                            # the action must be passed as a value
                            action = constants.Action.Bomb.value
                            self.defined = False
                            self.target_pos = None
                            self.target = Target.Safe.value
                            corrective_strategy = False
                        else:
                            # the agent must wait until it has again an ammo if the position is safe, move otherwise
                            dangerous_pos = tg_two.get_positions(obs)
                            safe_pos = is_pos_safe(self.target_pos, dangerous_pos)
                            if safe_pos:
                                action = constants.Action.Stop
                            else:
                                self.target = Target.Safe.value
                                self.defined = False
                                self.target_pos = None
                    else:
                        # case in which it is no more possible to reach the target
                        self.defined = False
                        self.target = Target.Safe.value

            # looking for a safe position
            elif self.target == Target.Safe.value:

                # take into account positions where the agent could be killed in case the bomb explodes at this timestep
                expl_fields = tg_two.get_positions(obs)
                dangerous_pos = get_immediate_expl_pos(expl_fields)
                # exclude the position of the agent from the dangerous position
                if obs['position'] in dangerous_pos:
                    dangerous_pos.remove(obs['position'])
                # create a board which takes into account possible dangerous positions
                new_board = change_board(obs['board'], dangerous_pos, Item.Rigid.value)
                new_obs = copy.copy(obs)
                new_obs['board'] = new_board

                if not self.defined:

                    # use Dijkstra's algorithm to find distances and keep the closest position
                    distances, nodes = get_distances(new_obs)
                    self.target_pos = tg_two.get_target_pos(distances, new_obs['position'], new_obs['board'])
                    # there is not a safe position for the agent (because in our strategy it must move)
                    if self.target_pos[0] == obs['position'][0] and self.target_pos[1] == obs['position'][1]:
                        return constants.Action.Stop
                    else:
                        self.defined = True

                # update the path
                distances, nodes = get_distances(new_obs)
                path = generate_path(nodes, self.target_pos)
                # at each step I check whether there is a path to reach the position or not
                if len(path) > 0:
                    next_position = path[0]
                    action = get_action(obs['position'], next_position)
                    corrective_strategy = False
                else:
                    # the current position of the agent is the target position
                    if obs['position'][0] == self.target_pos[0] and obs['position'][1] == self.target_pos[1]:
                        self.defined = False
                        self.target_pos = None
                        self.target = Target.Collect.value
                    else:
                        action = constants.Action.Stop
                        corrective_strategy = False

            # pick up a power-up
            else:
                if not self.defined:
                    pow_up_pos = get_positions_power_up(obs['board'])
                    # check whether there is at least a power-up to pick
                    if len(pow_up_pos) > 0:
                        # change the board allowing to reach power-ups
                        new_obs = obs.copy()
                        new_board = change_board(obs['board'], pow_up_pos, Item.Passage.value)
                        new_obs['board'] = new_board
                        # execute Dijkstra's algorithm to get distances
                        distances, nodes = get_distances(new_obs)
                        self.target_pos = tg_three.get_target_collect(distances, set(pow_up_pos))
                        # check whether a target has been found
                        if self.target_pos is None:
                            self.target = Target.Bomb.value
                            continue
                        else:
                            self.defined = True

                    # there are no power-up on the board
                    else:
                        self.target = Target.Bomb.value
                        continue

                # check whether on the target position there is still a power-up
                if obs['board'][self.target_pos[0], self.target_pos[1]] == Item.IncrRange.value or \
                    obs['board'][self.target_pos[0], self.target_pos[1]] == Item.Kick.value or \
                    obs['board'][self.target_pos[0], self.target_pos[1]] == Item.ExtraBomb.value:
                    # check whether the agent is on the target position before the generation of the distance matrix
                    if obs['position'][0] == self.target_pos[0] and obs['position'][1] == self.target_pos[1]:
                        self.defined = False
                        self.target_pos = None
                        self.target = Target.Bomb.value
                    else:
                        # change the board allowing to reach power-ups
                        new_board = change_board(obs['board'], [self.target_pos], Item.Passage.value)
                        obs['board'] = new_board
                        # execute Dijkstra's algorithm to get distances
                        distances, nodes = get_distances(obs)
                        path = generate_path(nodes, self.target_pos)
                        # at each step I check whether there is a path to reach the position or not
                        if len(path) > 0:
                            dangerous_pos = tg_two.get_positions(obs)
                            safe = is_path_safe(path, dangerous_pos)
                            if safe:
                                next_position = path[0]
                                action = get_action(obs['position'], next_position)
                                corrective_strategy = False
                            else:
                                # in case the path is not safe the agent tries to place a bomb
                                self.defined = False
                                self.target_pos = None
                                self.target = Target.Bomb.value
                        else:
                            self.target = Target.Safe.value
                            self.defined = False
                else:
                    # the agent must change the power-up target
                    self.defined = False
                    self.target_pos = None

        print(action)

        return action
