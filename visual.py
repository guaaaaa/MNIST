import matplotlib.pyplot as plt
import torch
import argparse
from pathlib import Path
from models import MNIST_CNN
from dataset import dataset
from utils import device, load
from loss_optimizer import loss_optimizer
from engine import accuracy_fn

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
train_data, test_data = dataset()

loss_fn, optimizer = loss_optimizer(model, lr=0.01)

def make_pred(model, test_data, device, n=9):
    model.eval()
    with torch.inference_mode():
        fig = plt.figure(figsize=(10, 10))
        for i in range(n):
            rand_int = torch.randint(0, len(test_data) - n, size=[1]).item()
            image, label = test_data[rand_int]
            image = image.unsqueeze(dim=0).to(device)

          # Show image first
            fig.add_subplot(3, 3, i+1)
            plt.imshow(image.squeeze().cpu(),cmap='gray')

          # Then normalize for prediction
            mu = image.mean()
            std = image.std()
            image = (image - mu)/ image.std()
            pred = model(image)

            if pred.argmax(dim=1).item() == label:
                plt.title(f'Label: {label} | Pred: {pred.argmax(dim=1).item()}', c='g')
            else:
                plt.title(f'Label: {label} | Pred: {pred.argmax(dim=1).item()}', c='r')
            plt.axis('off')

make_pred(model, test_data, device)