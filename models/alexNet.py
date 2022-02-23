from torch import nn
import Config as cfg
from customConv2d import customConv2d


class AlexNet(nn.Module):
    def __init__(self, compute_flavour, device, verbose, num_classes):
        super(AlexNet, self).__init__()
        self.compute_flavour = compute_flavour
        self.device = device
        self.verbose = verbose
        self.num_classes = num_classes
        self.print_verbose('AlexNet __init__() compute_flavour: {} device: {} num_classes: {}'.format(compute_flavour, device, num_classes), 1)

        self.features = nn.Sequential(
            customConv2d(3, 64, kernel_size=3, stride=1, padding=2, compute_flavour=compute_flavour,
                         device=device, verbose=verbose),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            customConv2d(64, 192, kernel_size=3, padding=2, compute_flavour=compute_flavour,
                         device=device, verbose=verbose),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            customConv2d(192, 384, kernel_size=3, padding=1, compute_flavour=compute_flavour,
                         device=device, verbose=verbose),
            nn.ReLU(inplace=True),
            customConv2d(384, 256, kernel_size=3, padding=1, compute_flavour=compute_flavour,
                         device=device, verbose=verbose),
            nn.ReLU(inplace=True),
            customConv2d(256, 256, kernel_size=3, padding=1, compute_flavour=compute_flavour,
                         device=device, verbose=verbose),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )


    def print_verbose(self, msg, v):
        if self.verbose >= v:
            cfg.LOG.write(msg)

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), 4096)
        x = self.classifier(x)
        return x

def alexNet(compute_flavour, device, verbose, num_classes=10):
    return AlexNet(compute_flavour, device, verbose, num_classes=num_classes)

def alexNet_cifar10(compute_flavour, device, verbose):
    return AlexNet(compute_flavour, device, verbose, num_classes=10)

def alexNet_cifar100(compute_flavour, device, verbose):
    return AlexNet(compute_flavour, device, verbose, num_classes=100)

def alexNet_svhn(compute_flavour, device, verbose):
    return AlexNet(compute_flavour, device, verbose, num_classes=10)

# This alexnet implementation supports using FP32 in the first layer (which is assumed to be the most sensitive)
    # while the other conv layer may use converted data-type for matmul
class AlexNetAsym(nn.Module):
    def __init__(self, compute_flavour, device, verbose, num_classes):
        super(AlexNetAsym, self).__init__()
        self.compute_flavour = compute_flavour
        self.device = device
        self.verbose = verbose
        self.num_classes = num_classes
        self.print_verbose('AlexNet Asym __init__() compute_flavour: {} device: {} num_classes: {}'.format(compute_flavour, device, num_classes), 1)

        self.features = nn.Sequential(
            # first conv layer - run legacy Conv2D (FP32), compute_flavor=0
            customConv2d(3, 64, kernel_size=3, stride=1, padding=2, compute_flavour=0,
                         device=device, verbose=verbose),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            customConv2d(64, 192, kernel_size=3, padding=2, compute_flavour=compute_flavour,
                         device=device, verbose=verbose),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            customConv2d(192, 384, kernel_size=3, padding=1, compute_flavour=compute_flavour,
                         device=device, verbose=verbose),
            nn.ReLU(inplace=True),
            customConv2d(384, 256, kernel_size=3, padding=1, compute_flavour=compute_flavour,
                         device=device, verbose=verbose),
            nn.ReLU(inplace=True),
            customConv2d(256, 256, kernel_size=3, padding=1, compute_flavour=compute_flavour,
                         device=device, verbose=verbose),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )


    def print_verbose(self, msg, v):
        if self.verbose >= v:
            cfg.LOG.write(msg)

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), 4096)
        x = self.classifier(x)
        return x

def alexnetAsym_cifar100(compute_flavour, device, verbose):
    return AlexNetAsym(compute_flavour, device, verbose, num_classes=100)