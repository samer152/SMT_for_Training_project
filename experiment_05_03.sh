#!/bin/bash


# ==============================================================================
echo "Custom Layer Training"
echo "Mantissa 0"
echo "Layer 6. first layer FP32"
python3 main.py --action TRAINING --arch resnet18-cifar100 --epoch 150 --batch_size 128 --seed 345 --compute_flavour 1 --v 1  --WD 5e-6 --GAMMA 0.1 --MOMENTUM 0.5 --distributed 0 --gpu 1 --exp normal --dir layers --layer 6

echo "Mantissa 2"
echo "Layer 6. first layer FP32"
python3 main.py --action TRAINING --arch resnet18-cifar100 --epoch 150 --batch_size 128 --seed 345 --compute_flavour 2 --v 1  --WD 5e-6 --GAMMA 0.1 --MOMENTUM 0.5 --distributed 0 --gpu 1 --exp normal --dir layers --layer 6
# ==============================================================================