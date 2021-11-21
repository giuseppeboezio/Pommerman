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


