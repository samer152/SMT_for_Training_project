from torch import nn
import Config as cfg
from customConv2d import customConv2d


class AlexNet(nn.Module):
    def __init__(self, threads, device, verbose, num_classes, muxing=0):
        super(AlexNet, self).__init__()
        self.threads = threads
        self.muxing = muxing
        self.device = device
        self.verbose = verbose
        self.num_classes = num_classes
        self.print_verbose('AlexNet __init__() threads: {} muxing: {} device: {} num_classes: {}'.format(threads, muxing, device, num_classes), 1)

        if threads == 0:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(64, 192, kernel_size=3, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        else:
            self.features = nn.Sequential(
                customConv2d(3, 64, kernel_size=3, stride=1, padding=2, threads_num=threads,
                                     device=device, macMuxType=muxing, verbose=verbose),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                customConv2d(64, 192, kernel_size=3, padding=2, threads_num=threads,
                                          device=device, macMuxType=muxing, verbose=verbose),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                customConv2d(192, 384, kernel_size=3, padding=1, threads_num=threads,
                                     device=device, macMuxType=muxing, verbose=verbose),
                nn.ReLU(inplace=True),
                customConv2d(384, 256, kernel_size=3, padding=1, threads_num=threads,
                                     device=device, macMuxType=muxing, verbose=verbose),
                nn.ReLU(inplace=True),
                customConv2d(256, 256, kernel_size=3, padding=1, threads_num=threads,
                                     device=device, macMuxType=muxing, verbose=verbose),
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




def alexNet(threads, device, verbose, num_classes=10, muxing=0):
    return AlexNet(threads, device, verbose, num_classes=num_classes, muxing=muxing)

def alexNet_cifar10(threads, device, verbose, muxing=0):
    return AlexNet(threads, device, verbose, num_classes=10, muxing=muxing)

def alexNet_cifar100(threads, device, verbose, muxing=0):
    return AlexNet(threads, device, verbose, num_classes=100, muxing=muxing)

def alexNet_svhn(threads, device, verbose, muxing=0):
    return AlexNet(threads, device, verbose, num_classes=10, muxing=muxing)