import torch
from pathlib import Path
from models import MNIST_CNN

def device():
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print(f'Device: {device}')
  return device

def save(params):
    MODEL_PATH = Path("cnn_model")
    MODEL_NAME = "custom.pth"

    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    if MODEL_SAVE_PATH.is_file():
        print("A customized model will be override")
    else:
        print("No customized model, creating one")
    torch.save(obj=params, f=MODEL_SAVE_PATH)
    print(f'Model is successfully saved to {MODEL_SAVE_PATH}')

def load():
    MODEL_PATH = Path("cnn_model")
    MODEL_NAME = "custom.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    model=MNIST_CNN().to(device)
    model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))