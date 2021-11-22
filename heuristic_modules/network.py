import torch
from torch import nn
import torch.nn.functional as F
from pommerman import constants


class DiscriminatorNet(nn.Module):

    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3,3), padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3,3), padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3,3), padding=1)
        self.linear = nn.Linear(constants.BOARD_SIZE * constants.BOARD_SIZE, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = F.sigmoid(x)
        return x


def train_loop(dataloader, model, optimizer, loss):

    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss):

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss(pred, y).item()
            if pred > 0.5:
                pred = 1
            else:
                pred = 0
            correct += (pred == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

