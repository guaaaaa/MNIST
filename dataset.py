import os
from pathlib import Path

import torchvision
from torchvision.transforms import ToTensor

def dataset():
    if not os.path.exists('data/MNIST'):
        print(f"Downloading data")

    else:
        print("Dataset already downloaded")
    train_data = torchvision.datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
    )

    test_data= torchvision.datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
    )
    
    return train_data, test_data