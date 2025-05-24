import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from image_show import ImageShow
from models import Net
import torch.optim as optim
import torch.nn as nn
from train_test import train, test_combinded, test_by_class


def import_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4
    num_workers = 0

    train_data = datasets.CIFAR10('data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10('data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, classes, batch_size


def main():
    trainloader, testloader, classes, batch_size = import_data()
    # ImageShow_instance = ImageShow(trainloader, classes, batch_size)
    # ImageShow_instance.show_batch()
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # lr is the how much the weights are updated, momentum is the speed of the update, SGF is the stochastic gradient descent
    Path = './cifar_net.pth'
    train(trainloader, criterion, optimizer, net, Path)
    test_combinded(testloader, classes, net, Path)
    test_by_class(testloader, classes, net, Path)

if __name__ == "__main__":
    main()