#!/bin/bash

echo "Test Script"
python3 main.py --action TRAINING --arch resnet18-cifar100 --epoch 2 --batch_size 128 --seed 44 --gpu 0 --compute_flavour 0 --v 1