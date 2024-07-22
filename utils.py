import torch

def device():
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print(f'Device: {device}')
  return device