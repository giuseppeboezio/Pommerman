import numpy as np
from pommerman import constants
import torch
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from heuristic_modules.network import BoardDataset
from heuristic_modules.network import DiscriminatorNet
from heuristic_modules.network import train_loop, test_loop


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


def train_and_test_nn_avoidance(model_path):

    # loading datasets
    training_data = BoardDataset("boards_train.csv","train", transform=ToTensor())
    test_data = BoardDataset("boards_test.csv", "test", transform=ToTensor())

    # creation of dataloader
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    model = DiscriminatorNet()

    loss = BCELoss()
    optimizer = Adam(model.parameters(), lr=0.07)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss, optimizer)
        test_loop(test_dataloader, model, loss)
    print("Done!")

    # saving the model
    torch.save(model.state_dict(), model_path)



