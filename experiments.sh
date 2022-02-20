#!/bin/bash

#echo "Test Script"
#python3 main.py --action TRAINING --arch resnet18-cifar100 --epoch 2 --batch_size 128 --seed 44 --gpu 0 --compute_flavour 0 --v 1

echo "Baseline - Mantissa 23"
python3 main.py --action TRAINING --arch resnet18-cifar100 --epoch 200 --batch_size 128 --seed 345 --compute_flavour 0 --v 1 --save_all_states 1  --WD 5e-4 --GAMMA 0.2 --MOMENTUM 0.9 --distributed 0 --gpu 1

echo "Mantissa 7"
python3 main.py --action TRAINING --arch resnet18-cifar100 --epoch 200 --batch_size 128 --seed 345 --compute_flavour 4 --v 1 --save_all_states 1  --WD 5e-4 --GAMMA 0.2 --MOMENTUM 0.9 --distributed 0 --gpu 1

echo "Mantissa 4"
python3 main.py --action TRAINING --arch resnet18-cifar100 --epoch 200 --batch_size 128 --seed 345 --compute_flavour 3 --v 1 --save_all_states 1  --WD 5e-4 --GAMMA 0.2 --MOMENTUM 0.9 --distributed 0 --gpu 1

echo "Mantissa 2"
python3 main.py --action TRAINING --arch resnet18-cifar100 --epoch 200 --batch_size 128 --seed 345 --compute_flavour 2 --v 1 --save_all_states 1  --WD 5e-4 --GAMMA 0.2 --MOMENTUM 0.9 --distributed 0 --gpu 1

echo "Mantissa 0"
python3 main.py --action TRAINING --arch resnet18-cifar100 --epoch 200 --batch_size 128 --seed 345 --compute_flavour 1 --v 1 --save_all_states 1  --WD 5e-4 --GAMMA 0.2 --MOMENTUM 0.9 --distributed 0 --gpu 1

