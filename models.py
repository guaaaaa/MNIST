import torch
from torch import nn

class MNIST_CNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv_block1 = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.conv_block2 = nn.Sequential(
        nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.fully_connected = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=16 * 7 * 7, out_features=10),
    )

  def forward(self, x):
    x = self.conv_block1(x)
    x = self.conv_block2(x)
    x = self.fully_connected(x)
    return x 