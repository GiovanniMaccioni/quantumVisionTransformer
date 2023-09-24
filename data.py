from torchvision.datasets import MNIST


from torchvision.datasets import MNIST
import torch.utils.data as data
from torchvision import datasets, transforms



def get_loaders():
    DATASET_PATH = "./data"
    batch_size = 256

    # For training, we add some augmentation. Networks are too powerful and would overfit.
    train_transform = transforms.Compose([transforms.Resize(28),
                                        #transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,)),#-1,1
                                        ])
    test_transform = transforms.Compose([transforms.Resize(28),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,)),#-1,1
                                        ])
    # Loading the training dataset. We need to split it into a training and validation part
    # We need to do a little trick because the validation set should not use the augmentation.
    train_dataset = MNIST(root=DATASET_PATH, train=True, transform=train_transform, download=True)
    #val_dataset = MNIST(root=DATASET_PATH, train=True, transform=test_transform, download=True)

    #pl.seed_everything(42)
    #train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000])
    #pl.seed_everything(42)
    #_, val_set = torch.utils.data.random_split(val_dataset, [45000, 5000])

    # Loading the test set
    test_set = MNIST(root=DATASET_PATH, train=False, transform=test_transform, download=True)

    # We define a set of data loaders that we can use for various purposes later.
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    #val_loader = data.DataLoader(val_set, batch_size=128, shuffle=False, drop_last=False, num_workers=4)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)

    return train_loader, test_loader