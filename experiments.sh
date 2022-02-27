#!/bin/bash

#echo "Test Script"
#python3 main.py --action TRAINING --arch resnet18-cifar100 --epoch 2 --batch_size 128 --seed 44 --gpu 0 --compute_flavour 0 --v 1
#
#echo "Baseline - Mantissa 23"
#python3 main.py --action TRAINING --arch resnet18-cifar100 --epoch 200 --batch_size 128 --seed 345 --compute_flavour 0 --v 1 --save_all_states 1  --WD 5e-6 --GAMMA 0.1 --MOMENTUM 0.5 --distributed 0 --gpu 1
#
#echo "Mantissa 7"
#python3 main.py --action TRAINING --arch resnet18-cifar100 --epoch 200 --batch_size 128 --seed 345 --compute_flavour 4 --v 1 --save_all_states 1  --WD 5e-6 --GAMMA 0.1 --MOMENTUM 0.5 --distributed 0 --gpu 1
#
#echo "Mantissa 4"
#python3 main.py --action TRAINING --arch resnet18-cifar100 --epoch 200 --batch_size 128 --seed 345 --compute_flavour 3 --v 1 --save_all_states 1  --WD 5e-6 --GAMMA 0.1 --MOMENTUM 0.5 --distributed 0 --gpu 1
#
#echo "Mantissa 2"
#python3 main.py --action TRAINING --arch resnet18-cifar100 --epoch 200 --batch_size 128 --seed 345 --compute_flavour 2 --v 1 --save_all_states 1  --WD 5e-6 --GAMMA 0.1 --MOMENTUM 0.5 --distributed 0 --gpu 1
#
#echo "Mantissa 0"
#python3 main.py --action TRAINING --arch resnet18-cifar100 --epoch 200 --batch_size 128 --seed 345 --compute_flavour 1 --v 1 --save_all_states 1  --WD 5e-6 --GAMMA 0.1 --MOMENTUM 0.5 --distributed 0 --gpu 1

echo "Custom Layer Training"
echo "Mantissa 0"
echo "Default - Layer 0. all layers custom"
python3 main.py --action TRAINING --arch resnet18-cifar100 --epoch 200 --batch_size 128 --seed 345 --compute_flavour 1 --v 1  --WD 5e-6 --GAMMA 0.1 --MOMENTUM 0.5 --distributed 0 --gpu 1 --layer 0

echo "Layer 1"
python3 main.py --action TRAINING --arch resnet18-cifar100 --epoch 200 --batch_size 128 --seed 345 --compute_flavour 1 --v 1  --WD 5e-6 --GAMMA 0.1 --MOMENTUM 0.5 --distributed 0 --gpu 1 --layer 1

echo "Layer 2"
python3 main.py --action TRAINING --arch resnet18-cifar100 --epoch 200 --batch_size 128 --seed 345 --compute_flavour 1 --v 1  --WD 5e-6 --GAMMA 0.1 --MOMENTUM 0.5 --distributed 0 --gpu 1 --layer 2

echo "Layer 3"
python3 main.py --action TRAINING --arch resnet18-cifar100 --epoch 200 --batch_size 128 --seed 345 --compute_flavour 1 --v 1  --WD 5e-6 --GAMMA 0.1 --MOMENTUM 0.5 --distributed 0 --gpu 1 --layer 3

echo "Layer 4"
python3 main.py --action TRAINING --arch resnet18-cifar100 --epoch 200 --batch_size 128 --seed 345 --compute_flavour 1 --v 1  --WD 5e-6 --GAMMA 0.1 --MOMENTUM 0.5 --distributed 0 --gpu 1 --layer 4

echo "Layer 5"
python3 main.py --action TRAINING --arch resnet18-cifar100 --epoch 200 --batch_size 128 --seed 345 --compute_flavour 1 --v 1  --WD 5e-6 --GAMMA 0.1 --MOMENTUM 0.5 --distributed 0 --gpu 1 --layer 5


echo "Custom Layer Training"
echo "Mantissa 2"
echo "Default - Layer 0. all layers custom"
python3 main.py --action TRAINING --arch resnet18-cifar100 --epoch 200 --batch_size 128 --seed 345 --compute_flavour 2 --v 1  --WD 5e-6 --GAMMA 0.1 --MOMENTUM 0.5 --distributed 0 --gpu 1 --layer 0

echo "Layer 1"
python3 main.py --action TRAINING --arch resnet18-cifar100 --epoch 200 --batch_size 128 --seed 345 --compute_flavour 2 --v 1  --WD 5e-6 --GAMMA 0.1 --MOMENTUM 0.5 --distributed 0 --gpu 1 --layer 1

echo "Layer 2"
python3 main.py --action TRAINING --arch resnet18-cifar100 --epoch 200 --batch_size 128 --seed 345 --compute_flavour 2 --v 1  --WD 5e-6 --GAMMA 0.1 --MOMENTUM 0.5 --distributed 0 --gpu 1 --layer 2

echo "Layer 3"
python3 main.py --action TRAINING --arch resnet18-cifar100 --epoch 200 --batch_size 128 --seed 345 --compute_flavour 2 --v 1  --WD 5e-6 --GAMMA 0.1 --MOMENTUM 0.5 --distributed 0 --gpu 1 --layer 3

echo "Layer 4"
python3 main.py --action TRAINING --arch resnet18-cifar100 --epoch 200 --batch_size 128 --seed 345 --compute_flavour 2 --v 1  --WD 5e-6 --GAMMA 0.1 --MOMENTUM 0.5 --distributed 0 --gpu 1 --layer 4

echo "Layer 5"
python3 main.py --action TRAINING --arch resnet18-cifar100 --epoch 200 --batch_size 128 --seed 345 --compute_flavour 2 --v 1  --WD 5e-6 --GAMMA 0.1 --MOMENTUM 0.5 --distributed 0 --gpu 1 --layer 5

