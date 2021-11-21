import numpy as np
from pommerman import constants
import torch
from torch.nn import BCELoss
from torch.optim import Adam
import network


def preprocess_state_avoidance(state, position):
    """Create the input for the NN
    position: tuple containing the position to evaluate
    passage: 0
    bombs: 3
    my agent's position: 10
    marked position: 14
    other objects: 15"""

    board = state['board']
    input_state = np.array(board)
    for i in range(constants.BOARD_SIZE):
        for j in range(constants.BOARD_SIZE):
            # not relevant objects
            if 0 < input_state[i, j] < 3 or 3 < input_state[i, j] < 10 or input_state[i, j] > 10:
                input_state[i, j] = 15
            # marked position
            if i == position[0] and j == position[1]:
                input_state[i, j] = 14
    torch_state = torch.from_numpy(input_state)
    return torch_state


def train():

    model = network.DiscriminatorNet()
    # setting the optimizer
    optimizer = Adam(model.parameters(), lr=0.07)
    # setting loss function
    criterion = BCELoss()



