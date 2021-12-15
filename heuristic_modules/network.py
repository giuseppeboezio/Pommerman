from pommerman import constants
import pandas as pd
import os
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.data import DataLoader


class DiscriminatorNet(nn.Module):

    def __init__(self, num_channels):
        super(DiscriminatorNet, self).__init__()
        self.conv1 = nn.Conv3d(5, num_channels, kernel_size=(3,3), padding=1)
        self.conv2 = nn.Conv3d(num_channels, num_channels, kernel_size=(3,3), padding=1)
        self.conv3 = nn.Conv3d(num_channels, num_channels, kernel_size=(3,3), padding=1)
        flat_dimension = constants.BOARD_SIZE * constants.BOARD_SIZE * num_channels
        self.linear1 = nn.Linear(flat_dimension, round(flat_dimension / 3))
        self.linear2 = nn.Linear(round(flat_dimension / 3), 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        return x


class BoardDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # reading the array from the file and reshape it in 3d
        image = np.loadtxt(img_path, delimiter=',')
        image = image.reshape((constants.BOARD_SIZE, constants.BOARD_SIZE, 5))
        # create the tensor adjusting dimensions and type
        im_torch = torch.from_numpy(image)
        im_torch = torch.reshape(im_torch, (5,constants.BOARD_SIZE, constants.BOARD_SIZE))
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = torch.from_numpy(image)
        if self.target_transform:
            label = self.target_transform(label)
        return im_torch, label


def train_loop(dataloader, model, optimizer, loss_fun):

    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):

        # Compute prediction and loss
        pred = model(X)

        # reshape of y
        y = torch.reshape(y, (len(y), 1))
        y = y.float()

        loss = loss_fun(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fun):

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:

            pred = model(X)

            # reshape of y
            y = torch.reshape(y, (len(y), 1))
            y = y.float()

            test_loss += loss_fun(pred, y).item()

            # conversion of probability to survive with binary classes values
            pred_c = torch.zeros(pred.shape)

            for i in range(pred.shape[0]):
                for j in range(pred.shape[1]):
                    if pred[i,j] > 0.7:
                        pred_c[i,j] = 1
                    else:
                        pred_c[i,j] = 0

            correct += (pred_c == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def train_and_test(csv_train, csv_test, dir_train, dir_test, model_path):

    # loading datasets
    training_data = BoardDataset(csv_train, dir_train)
    test_data = BoardDataset(csv_test, dir_test)

    # creation of dataloader
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    num_channels = 16
    model = DiscriminatorNet(num_channels)

    loss = BCELoss()
    optimizer = Adam(model.parameters(), lr=0.07)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, optimizer, loss)
        test_loop(test_dataloader, model, loss)
    print("Done!")

    # saving the model
    torch.save(model.state_dict(), model_path)