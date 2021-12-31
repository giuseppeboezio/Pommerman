from pommerman.agents import BaseAgent
from pommerman import constants
from heuristic_modules.module_1 import get_distances
from enum import Enum
import numpy as np
import target_1 as tg_one
import target_2 as tg_two


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


class Target(Enum):
    """Target of the agent"""

    Bomb = 1
    Safe = 2
    Collect = 3


# agent behaves in the following way:
# Target 1: put the bomb in the position which reveals power up according to softmax probability
# Target 2: choose a position where to avoid being killed by a bomb
# Target 3: choose the item to pick up according to a probability which depends on the distance from the object
class PlannerAgent(BaseAgent):

    def __init__(self):
        super(PlannerAgent, self).__init__()
        self.target = Target.Bomb.value  # target to achieve
        self.defined = False  # flag to establish if the target position has been found or not
        self.position = None  # position to reach from the current one
        self.path = []  # path of nodes to follow to reach target position

    def act(self, obs, action_space):

        # the objective is putting a bomb
        if self.target == Target.Bomb.value:
            # the objective is finding the position where to put the bomb
            if not self.defined:
                distances, nodes = get_distances(obs)
                # obtaining position where it could be possible to put a bomb
                positions = tg_one.get_positions(nodes)
                # forward step
                target_pos = tg_one.forward_step(positions, obs)
                # update the path
                self.path = generate_path(nodes, target_pos)
                self.defined = True

            # There are still positions to cross
            if len(self.path) > 0:

                next_position = self.path[0]
                action = get_action(obs['position'], next_position)
                self.path.pop(0)

            # The agent has reached the target position, it can place the bomb
            else:

                action = constants.Action.Bomb
                self.defined = False
                self.target = Target.Safe.value

        # looking for a safe position
        elif self.target == Target.Safe.Safe:
            if not self.defined:
                # find positions where the agent can be possibly killed
                dangerous_pos = tg_two.get_dangerous_positions(obs)
                dangerous_pos = set(dangerous_pos)
                new_obs = tg_two.change_board(obs['board'], dangerous_pos)
                distances, nodes = get_distances(new_obs)
                target_pos = tg_two.get_target_pos(distances)



        # pick up a power-up
        else:
            pass

        return action
