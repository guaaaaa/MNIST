import matplotlib.pyplot as plt
import torch
import argparse
from pathlib import Path
from models import MNIST_CNN
from dataset import dataset
from utils import device, load
from loss_optimizer import loss_optimizer
from engine import accuracy_fn
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description="Load")

# Add arguments to parser
parser.add_argument("--load",
                    default=False,
                    type=bool,
                    )
args = parser.parse_args()
load = args.load

# Model setup
device = device()
model=MNIST_CNN().to(device)

# Load state_dic
MODEL_PATH = Path("cnn_model")
MODEL_NAME = "custom.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

if load and MODEL_SAVE_PATH.is_file():
    model.load_state_dict(torch.load(f='cnn_model/custom.pth'))
    print("Successfully load the customized model")
else:
    model.load_state_dict(torch.load(f='cnn_model/cnn_model.pth'))
    print("Successfully load the default model")

# Get data set
BATCH_SIZE = 32
train_data, test_data = dataset()

loss_fn, optimizer = loss_optimizer(model, lr=0.01)

def make_pred(model, test_data, device):
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    acc = 0
    num = 0
    model.eval()
    with torch.inference_mode():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            acc += accuracy_fn(y, y_pred.argmax(dim=1))
            num += 1
    print(f'Accuracy on test set: {acc/num}')

make_pred(model, test_data, device)