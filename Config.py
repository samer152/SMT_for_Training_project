import os
from Logger import Logger
from Datasets import Datasets
from models.leNet5 import leNet_cifar10, leNet_cifar100
from models.alexNet import alexNet_cifar10, alexNet_cifar100
from models.resnet_cifar import resNet18_cifar100
from models.resnet_imagenet import resNet18_imagenet

basedir, _ = os.path.split(os.path.abspath(__file__))
basedir = os.path.join(basedir, 'data')

WINDOWS = True
EXPERIMENT = ''
DIR = ''

MODELS = {'lenet5_cifar10': leNet_cifar10,
          'alexnet_cifar10': alexNet_cifar10,
          'alexnet_cifar100': alexNet_cifar100,
          'resnet18_cifar100': resNet18_cifar100,
          'resnet18_imagenet': resNet18_imagenet,
          }

BATCH_SIZE = 128

# ------------------------------------------------
#                   Directories
# ------------------------------------------------
RESULTS_DIR = os.path.join(basedir, f'{DIR}_results')
DATASET_DIR = os.path.join(basedir, 'datasets')
DATASET_DIR_IMAGENET = '/mnt/ilsvrc2012'
FINAL_RESULTS_DIR = os.path.join(basedir, f'backward_results')


# ------------------------------------------------
#                Init and Defines
# ------------------------------------------------
USER_CMD = None
# ------------------------------------------------
#                Init and Defines
# ------------------------------------------------
USER_CMD = None
LOG = Logger()


def get_dataset(dataset):
    if dataset == 'cifar10':
        return Datasets.get('CIFAR10', DATASET_DIR)
    elif dataset == 'cifar100':
        return Datasets.get('CIFAR100', DATASET_DIR)
    elif dataset == 'imagenet':
        return Datasets.get('ImageNet', DATASET_DIR_IMAGENET)
    else:
        raise NotImplementedError