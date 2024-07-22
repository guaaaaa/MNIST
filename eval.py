import matplotlib.pyplot as plt
import torch
from models import MNIST_CNN
from dataset import dataset
from utils import device

train_data, test_data = dataset()

device = device()

model=MNIST_CNN().to(device)
model.load_state_dict(torch.load(f='cnn_model/cnn_model.pth'))

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