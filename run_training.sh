#!/bin/bash

python train.py --outdir=./training-runs/MNIST --cfg=stylegan3-t --data=data/mnist.zip --cond=True \
  --gpus=2 --batch=64 --gamma=0.125 --batch-gpu=32 \
  --kimg=1500 --tick=10 --snap=5 --metrics=none \
  --mirror=False --aug=noaug 