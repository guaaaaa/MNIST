import torch
from torch import nn

def loss_optimizer(model):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    return loss_fn, optimizer