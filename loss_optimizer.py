import torch
from torch import nn

def loss_optimizer(model, lr):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    return loss_fn, optimizer