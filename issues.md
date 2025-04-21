# Issues running the StyleGAN3

## Issue 1: AttributeError: module 'distutils' has no attribute 'version'

When running the StyleGAN3 code, I encountered the following error:

```shell
File "/home/anton.gomez.lopez/miniconda3/envs/stylegan3/lib/python3.9/site-packages/torch/utils/tensorboard/__init__.py", line 4, in <module>
    LooseVersion = distutils.version.LooseVersion
AttributeError: module 'distutils' has no attribute 'version'
```

The main reason of the problem could be that the installed version of `tensorboard` (2.17.0) doesn't match the version of `torch` (1.9). I'm not sure, but the minimum version of `torch` required by `tensorboard` is 1.12. solving this, also could solve the problem.

### Solution

Open the file `/home/anton.gomez.lopez/miniconda3/envs/stylegan3/lib/python3.9/site-packages/torch/utils/tensorboard/__init__.py` and replace the line:

```python
LooseVersion = distutils.version.LooseVersion
```

with:

```python
from packaging.version import Version as LooseVersion
```

### Note

The main reason of the problem could be that the installed version of `tensorboard` (2.17.0) doesn't match the version of `torch` (1.9). I'm not sure, but the minimum version of `torch` required by `tensorboard` is 1.12. solving this, also could solve the problem.

## Issue 2: Training with labels

To use labels in StyleGAN3, you need to specify the flag `--cond=True`. By default is False. Once you launch the training script, with the MNIST dataset you should see:

```shell
Num images:  120000
Image shape: [1, 32, 32]
Label shape: [10]
```

## Issue 3: Evaluating metrics slowing down the training

When you are training the model, the evaluation of the metrics is slowing down the training. You can disable the evaluation of the metrics by setting `--metrics=none`. This will speed up the training process.

## Issue 4: Pin memory thread exited unexpectedly

When I was training the model, I got the following error:

```shell
RuntimeError: Pin memory thread exited unexpectedly
```

This error is related to the `pin_memory` option in PyTorch. When this option is set to `True`, PyTorch will use a separate thread to copy data from the CPU to the GPU. If this thread exits unexpectedly, you will see this error.

## Issue 5: Tensorboard

I don't know why have I changed the code to start and finish tensorboar. I have to try to run it again as it was in a first moment. I suppose it has to work (I have to check it). Now, it seems that I am seeing the tensorboard associated to the previous training (thta doesn't finish properly).

# Notes

## Note 1: Fisrt training output

I run a first training and the output was:

```shell
Training for 25000 kimg...
tick 0     kimg 0.0      time 1m 31s       sec/tick 11.6    sec/kimg 362.35  maintenance 79.3   cpumem 3.31   gpumem 9.65   reserved 10.41  augment 0.000
Evaluating metrics...
{"results": {"fid50k_full": 386.7726983706855}, "metric": "fid50k_full", "total_time": 3876.3731479644775, "total_time_str": "1h 04m 36s", "num_gpus": 1, "snapshot_pkl": "network-snapshot-000000.pkl", "timestamp": 1743427398.6298556}
tick 1     kimg 4.0      time 1h 23m 02s   sec/tick 955.4   sec/kimg 238.85  maintenance 3935.3 cpumem 3.87   gpumem 9.79   reserved 10.24  augment 0.003
tick 2     kimg 8.0      time 1h 38m 58s   sec/tick 956.0   sec/kimg 239.00  maintenance 0.2    cpumem 3.87   gpumem 9.79   reserved 10.24  augment 0.007
tick 3     kimg 12.0     time 1h 54m 54s   sec/tick 956.1   sec/kimg 239.02  maintenance 0.2    cpumem 3.87   gpumem 9.79   reserved 10.24  augment 0.009
tick 4     kimg 16.0     time 2h 10m 50s   sec/tick 955.8   sec/kimg 238.96  maintenance 0.0    cpumem 3.87   gpumem 9.79   reserved 10.24  augment 0.010
tick 5     kimg 20.0     time 2h 26m 46s   sec/tick 956.1   sec/kimg 239.03  maintenance 0.2    cpumem 3.87   gpumem 9.79   reserved 10.24  augment 0.006
```

Before ending the training, an error appeared:

```shell
RuntimeError: Pin memory thread exited unexpectedly
```

I'm not sure what this error means, and why it appeared. However, the training was programmed to run for 25,000 kimg (I think it is excesive for MNIST).

### Conclusions

- The part of the training thta took the most time was the evaluation of the metrics `fid50k_full`.
- Each tick took ~15 minutes (~4 minutes each kimg).
- It seems that the GPU memory is stable (9.79 GB). Therefore, the memory of the server GPUs should not be a problem (we have 24 GB per GPU).
- The CPU memory does not seem to be a problem, since most of the time is occupied in `cpumem 3.87` and we have 128 GB of RAM.

I will change some parameters to build a basemodel for MNIST. First I will increase the batch size because I have seen that I have more space in the GPU. I will also reduce the number of kimg to 200, since I think that 25,000 kimg is too much for MNIST. I will also disable the evaluation of the metrics, since it is slowing down the training process.

- COND = True
- GPUS = 2
- BATCH = 64
- BATCH_GPU = 32 (The batch size should be a multiple of the number of GPUs by the batch siz eof the GPU)
- GAMMA = 0.5 (Random)
- KIMG = 200 (I will reduce the number of kimg to 200)
- TICK = 5 (A tick every 5 kimg)
- SNAP = 2 (Save the model every 2 ticks - 10 kimg)
- METRICS = none (Disable the evaluation of the metrics)
- MIRROR = False (I will not use the mirror augmentation)
- AUG = noaug (I will not use any augmentation)

```python
python train.py --outdir=./training-runs/MNIST --cfg=stylegan3-t --data=data/mnist.zip --cond=True \
  --gpus=2 --batch=64 --gamma=0.5 --batch-gpu=32 \
  --kimg=200 --tick=5 --snap=2 --metrics=none \
  --mirror=False --aug=noaug
```

Note that each time we create a snapshot, **we also generate a set of fake images**. If in any moment I want to change this, I can modify the values of the parameters `image_snapshot_ticks` and `network_snapshot_ticks` in `training_loop` function in the `training_loop.py` file. The default values are:

```python
image_snapshot_ticks=50,    # How often to save image snapshots? None = disable.
network_snapshot_ticks=50,  # How often to save network snapshots? None = disable.
```

## Note 2: Second training output

the first observation is that if we increase the batch size, the time per kimg is reduced significantly from 239.02 sec/kimg to 53.82 sec/kimg. Also, it is important to note that the gpu memory reserved is bigger, 17.74GB (near to our limit of 24 GB). I'm not sure if we can increase the batch size to 128, so I will keep it at 64. The output is in the first folder of `training-runs/MNIST`.

## Note 3: Developed util code

I have added some functionality to `gen_images.py` to generate a grid of images. However I tried to keep the original functionality. Therefore, if you call the script with the same arguments as before, it will work as before. The only difference is that now you can also generate a grid of images using the `--classes` flag that works very similar to `--seeds` flag. In this case, we will use a seed for each class and we will generate the same number of images per class as the number of classes (to ensure the grid is square).

Also, I'm trying to develop a util notebook to analyze the metrics of the training.

## Note 4: Launching another training

To see if I can replicate the results of [Oliver](https://huggingface.co/oweissl/stylegan3_conditional_MNIST), I will launch another training with the same parameters as him. I'm not sure if he uses data augmentation or not. I'm not using it. I think the `kimg` parameter is too high, but I will keep it for now. The output is in the second folder of `training-runs/MNIST`. I have stopped the training after 3100 kimg, since I think it is enough for MNIST.

```python
python train.py --outdir=./training-runs/MNIST --cfg=stylegan3-t --data=data/mnist.zip --cond=True \
  --gpus=2 --batch=64 --gamma=0.125 --batch-gpu=32 \
  --kimg=3100 --tick=10 --snap=5 --metrics=none \
  --mirror=False --aug=noaug
```
