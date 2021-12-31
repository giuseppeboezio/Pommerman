import numpy as np
from pommerman import constants
from pommerman.constants import Item
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
        agent_actions = [constants.Action.Stop for i in range(4)]
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