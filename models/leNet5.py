from torch import nn, optim
import torch.nn.functional as F
from Model_StatsLogger import Model_StatsLogger
import Config as cfg
from customConv2d import customConv2d


class LeNet(nn.Module):
    def __init__(self, compute_flavour, device, verbose, num_classes):
        super(LeNet, self).__init__()
        self.compute_flavour = compute_flavour
        self.device = device
        self.verbose = verbose
        self.num_classes = num_classes

        self.print_verbose('LeNet __init__() compute_flavour: {} device: {} num_classes: {}'.format(compute_flavour, device, num_classes), 1)


        self.conv1 = customConv2d(3, 6, 5, compute_flavour=compute_flavour, device=device, verbose=verbose)
        self.conv2 = customConv2d(6, 16, 5, compute_flavour=compute_flavour, device=device, verbose=verbose)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


    def print_verbose(self, msg, v):
        if self.verbose >= v:
            cfg.LOG.write(msg)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.reshape(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out




def leNet(compute_flavour, device, verbose, num_classes=10):
    return LeNet(compute_flavour, device, verbose, num_classes=num_classes)

def leNet_cifar10(compute_flavour, device, verbose):
    return LeNet(compute_flavour, device, verbose, num_classes=10)

def leNet_cifar100(compute_flavour, device, verbose):
    return LeNet(compute_flavour, device, verbose, num_classes=100)