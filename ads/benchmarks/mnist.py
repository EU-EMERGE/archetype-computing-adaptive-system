import torchvision
import torch
from torchvision import transforms


def get_mnist_data(root, bs_train, bs_test, valid_perc=10):
    train_dataset = torchvision.datasets.MNIST(root=root,
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root=root,
                                              train=False,
                                              transform=transforms.ToTensor())

    valid_size = int(len(train_dataset) * (valid_perc / 100.))
    train_size = len(train_dataset) - valid_size
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size])

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=bs_train,
                                               shuffle=True)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=bs_test,
                                               shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=bs_test,
                                              shuffle=False)

    return train_loader, valid_loader, test_loader
