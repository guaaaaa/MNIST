import os
import torch
import torchvision
from torch.utils.data import DataLoader # Import DataLoader here
from torchvision.transforms import ToTensor, Compose, Normalize

def mean(dataloader):
  train_mean = 0
  for X, y in dataloader:
    batch_mean = X.mean(dim=[0, 2, 3])
    train_mean += batch_mean
  train_mean /= len(dataloader)
  return train_mean

def std(dataloader):
  mu = mean(dataloader)
  train_std = 0
  for X, y in dataloader:
    batch_var = torch.sum((X - mu)**2) / (X.shape[0] * X.shape[2] * X.shape[3])
    train_std += batch_var
  train_std = torch.sqrt(train_std / len(dataloader))
  return train_std

def normalize(train_data, test_data, bs):
  BATCH_SIZE = bs 
  train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
  test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
  # print(f'Train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}')
  # print(f'Test dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}')
  mu = mean(dataloader=train_dataloader)
  sd = std(dataloader=train_dataloader)

  # Download normalized input
  train_data_normal = torchvision.datasets.MNIST(
    root = 'data',
    train = True,
    download = True,
    transform = Compose([ToTensor(), Normalize(mu, sd)]), # first ToTensor then Normalize
  )

  test_data_normal = torchvision.datasets.MNIST(
      root = 'data',
      train = False,
      download = True,
      transform = Compose([ToTensor(), Normalize(mu, sd)]),
  )

  # Set up normalized dataloader
  train_dataloader_normal = DataLoader(train_data_normal, batch_size=BATCH_SIZE, shuffle=True)
  test_dataloader_normal = DataLoader(test_data_normal, batch_size=BATCH_SIZE, shuffle=False)
  print(f'Train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}')

  return train_dataloader_normal, test_dataloader_normal