from pommerman import constants
import pandas as pd
import os

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.io import read_image


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
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def train_and_test(csv_train, csv_test, dir_train, dir_test, model_path):

    # loading datasets
    training_data = BoardDataset(csv_train, dir_train, transform=ToTensor())
    test_data = BoardDataset(csv_test, dir_test, transform=ToTensor())

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