#!/bin/bash


# ==============================================================================
#echo "Custom Layer Training"
#echo "Mantissa 0"
#python3 main.py --action TRAINING --arch resnet18-cifar100 --epoch 1 --batch_size 128 --seed 345 --compute_flavour 1 --v 1  --WD 5e-6 --GAMMA 0.1 --MOMENTUM 0.5 --distributed 0 --gpu 0 --exp normal --dir layers --layer 6
#
#echo "Mantissa 2"
#echo "Default - Layer 6. last layer FP32"
#python3 main.py --action TRAINING --arch resnet18-cifar100 --epoch 150 --batch_size 128 --seed 345 --compute_flavour 2 --v 1  --WD 5e-6 --GAMMA 0.1 --MOMENTUM 0.5 --distributed 0 --gpu 0 --exp normal --dir layers --layer 6
#
#echo "Layer 1"
#python3 main.py --action TRAINING --arch resnet18-cifar100 --epoch 150 --batch_size 128 --seed 345 --compute_flavour 2 --v 1  --WD 5e-6 --GAMMA 0.1 --MOMENTUM 0.5 --distributed 0 --gpu 0 --exp normal --dir layers --layer 1
#
#echo "Layer 2"
#python3 main.py --action TRAINING --arch resnet18-cifar100 --epoch 150 --batch_size 128 --seed 345 --compute_flavour 2 --v 1  --WD 5e-6 --GAMMA 0.1 --MOMENTUM 0.5 --distributed 0 --gpu 0 --exp normal --dir layers --layer 2
#
#echo "Layer 3"
#python3 main.py --action TRAINING --arch resnet18-cifar100 --epoch 150 --batch_size 128 --seed 345 --compute_flavour 2 --v 1  --WD 5e-6 --GAMMA 0.1 --MOMENTUM 0.5 --distributed 0 --gpu 0 --exp normal --dir layers --layer 3
#
#echo "Layer 4"
#python3 main.py --action TRAINING --arch resnet18-cifar100 --epoch 150 --batch_size 128 --seed 345 --compute_flavour 2 --v 1  --WD 5e-6 --GAMMA 0.1 --MOMENTUM 0.5 --distributed 0 --gpu 0 --exp normal --dir layers --layer 4
#
#echo "Layer 5"
#python3 main.py --action TRAINING --arch resnet18-cifar100 --epoch 150 --batch_size 128 --seed 345 --compute_flavour 2 --v 1  --WD 5e-6 --GAMMA 0.1 --MOMENTUM 0.5 --distributed 0 --gpu 0 --exp normal --dir layers --layer 5
## ==============================================================================
#
## ==============================================================================
#echo "Backward"
#
#echo "Mantissa 2"
#python3 main.py --action TRAINING --arch resnet18-cifar100 --epoch 150 --batch_size 128 --seed 345 --compute_flavour 2 --v 1  --WD 5e-6 --GAMMA 0.1 --MOMENTUM 0.5 --distributed 0 --gpu 0 --exp backward --dir backward
#
#echo "Mantissa 0"
#python3 main.py --action TRAINING --arch resnet18-cifar100 --epoch 150 --batch_size 128 --seed 345 --compute_flavour 1 --v 1  --WD 5e-6 --GAMMA 0.1 --MOMENTUM 0.5 --distributed 0 --gpu 0 --exp backward --dir backward

# ==============================================================================

# ==============================================================================
echo "forward"

echo "Mantissa 2"
python3 main.py --action TRAINING --arch resnet18-cifar100 --epoch 150 --batch_size 128 --seed 345 --compute_flavour 2 --v 1  --WD 5e-6 --GAMMA 0.1 --MOMENTUM 0.5 --distributed 0 --gpu 0 --exp forward --dir forward

echo "Mantissa 0"
python3 main.py --action TRAINING --arch resnet18-cifar100 --epoch 150 --batch_size 128 --seed 345 --compute_flavour 1 --v 1  --WD 5e-6 --GAMMA 0.1 --MOMENTUM 0.5 --distributed 0 --gpu 0 --exp forward --dir forward

# ==============================================================================
