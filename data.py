from torchvision.datasets import MNIST
import torch.utils.data as data
from torchvision import transforms

import numpy as np


def get_loaders(batch_size, classes):
    DATASET_PATH = "./data"

    train_transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,)),#-1,1
                                        ])
    test_transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,)),#-1,1
                                        ])

    train_dataset = MNIST(root=DATASET_PATH, train=True, transform=train_transform, download=True)
    # Filter out labels (originally 0-9), leaving only labels for the class selected
    idx = []
    for l in classes:
        idx = idx + list(np.where(train_dataset.targets == l)[0])

    train_dataset.data = train_dataset.data[idx]
    train_dataset.targets = train_dataset.targets[idx]

    # Loading the test set
    test_set = MNIST(root=DATASET_PATH, train=False, transform=test_transform, download=True)

    # Filter out labels (originally 0-9), leaving only labels for the class selected
    idx = []
    for l in classes:
        idx = idx + list(np.where(test_set.targets == l)[0])
    test_set.data = test_set.data[idx]
    test_set.targets = test_set.targets[idx]

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)

    return train_loader, test_loader