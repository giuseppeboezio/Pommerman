from pommerman.agents import BaseAgent
from pommerman.forward_model import ForwardModel
from pommerman import constants
from pommerman.constants import Item
from heuristic_modules.module_1 import get_distances
import numpy as np
from scipy.special import softmax


def get_positions(nodes):
    """Positions where the agent can put a bomb"""

    conn_nodes = [node for node in nodes if node.distance != np.inf]  # Taking into account putting the bomb in the curr pos
    positions = []
    for node in conn_nodes:
        positions.append(node.get_position())

    return positions


def get_positions_power_up(board):
    """Return the positions of the power-up on the board"""
    positions = []
    for i in range(constants.BOARD_SIZE):
        for j in range(constants.BOARD_SIZE):
            if board[i,j] == Item.ExtraBomb.value or board[i,j] == Item.Kick.value or board[i,j] == Item.IncrRange.value:
                positions.append((i,j))

    return positions


def get_num_items(pos_pow_ups, board):
    """Return the number of power ups after the blast of the bomb"""
    num = 0
    for i in range(constants.BOARD_SIZE):
        for j in range(constants.BOARD_SIZE):
            if (board[i,j] == Item.ExtraBomb.value or board[i,j] == Item.Kick.value or
                    board[i,j] == Item.IncrRange.value) and (i,j) not in pos_pow_ups:
                num = num + 1
    return num


def forward_step(positions, obs):
    """Produce a forward state for each possible position and return the target position of the agent"""

    num_items = []
    pos_pow_up = get_positions_power_up(obs['board'])
    pow_up_set = set(pos_pow_up)

    for position in positions:

        fwd_model = ForwardModel()
        agent_actions = [constants.Action.Stop.value for i in range(4)]
        obs['board'][position[0],position[1]] = constants.Item.Bomb.value

        # TODO ask Professor for the evolution of the state

        # add the number of found power-up
        num_pow_up = get_num_items(pow_up_set, forward_board)
        num_items.append(num_pow_up)

    # apply softmax and choose randomly a position
    soft_num = softmax(num_items)
    idx = np.random.choice(np.range(len(positions)), p=soft_num)
    target_pos = positions[idx]

    return target_pos


# agent behaves in the following way:
# Target 1: put the bomb in the position which reveals power up according to softmax probability
# Target 2: choose a position where to avoid being killed by a bomb
# Target 3: choose the item to pick up according to a probability which depends on the distance from the object
class PlannerAgent(BaseAgent):

    def __init__(self):
        super(PlannerAgent, self).__init__()
        self.target = 1  # target to achieve
        self.phase = False  # phase which denotes the sub goal of the target
        self.position = None  # position to reach from the current one

    def act(self, obs, action_space):

        # the objective is putting a bomb
        if self.target == 1:
            # the objective is finding the position where to put the bomb
            if self.phase == 1:
                distances, nodes = get_distances(obs)
                # obtaining position where it could be possible to put a bomb
                positions = get_positions(nodes)
                # forward model
