import argparse
import math
import os
import sys
import random

import matplotlib
import torch.multiprocessing as mp
import torch.distributed as dist

matplotlib.use('Agg')
import torch

import Config as cfg
from NeuralNet import NeuralNet

parser = argparse.ArgumentParser(description='Samer Kurzum, samer152@gmail.com',
                                 formatter_class=argparse.RawTextHelpFormatter)
model_names = ['lenet5-cifar10',
               'alexnet-cifar10',
               'alexnet-cifar100',
               'resnet18-cifar100',
               'resnet18-imagenet'
               ]

parser.add_argument('-a', '--arch', metavar='ARCH', choices=model_names, required=False,
                    help='model architectures and datasets:\n' + ' | '.join(model_names))
parser.add_argument('--action', choices=['TRAINING'], required=True,
                    help='TRAINING: Run given model on a given dataset')
parser.add_argument('--desc')
parser.add_argument('--batch_size', default=128, type=int, metavar='N',
                    help='batch size (default: 128)')
parser.add_argument('--epochs', default=10, type=int,
                    help='number of epochs in training mode')
parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda',
                    help='device to run on')
parser.add_argument('--cuda_conv', default=0, type=int,
                    help='compute cuda convolution for each network')
parser.add_argument('--LR', default=0.1, type=float,
                    help='starting learning rate')
parser.add_argument('--LRD', default=0, type=int,
                    help='learning rate decay - if enabled LR is decreased')
parser.add_argument('--WD', default=0, type=float,
                    help='weight decay')
parser.add_argument('--MOMENTUM', default=0, type=float,
                    help='momentum')
parser.add_argument('--GAMMA', default=0.1, type=float,
                    help='gamma')
parser.add_argument('--MILESTONES', nargs='+', default=[60, 120, 160], type=int,
                    help='milestones')
parser.add_argument('--seed', default=42, type=int,
                    help='seed number')
parser.add_argument('--threads', nargs='+', default=None, type=int,
                    help='List of threads that custom convolution will run with')
parser.add_argument('--muxing', nargs='+', default=None, type=int,
                    help='Muxing type to enable - matched with threads command option')
parser.add_argument('--gpu', nargs='+', default=None,
                    help='GPU to run on (default: 0)')
parser.add_argument('--distributed', default=0, type=int,
                    help='DistributedDataParallel')
parser.add_argument('--model_path', default=None, help='model path to load')
parser.add_argument('--v', default=0, type=int, help='verbosity level (0,1,2) (default:0)')






if __name__ == '__main__':
    main()
