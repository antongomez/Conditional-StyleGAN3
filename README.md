# StyleGAN3 for Multispectral Image Classification

This project is an adaptation of the official StyleGAN3 implementation in PyTorch, available at the [original StyleGAN3 repository](https://github.com/NVlabs/stylegan3). The main goal is to convert the discriminator into a classifier and combine the adversarial training of StyleGAN3 with a supervised classification task. The objective is to classify multispectral drone images to detect the presence of different tree species, human constructions, water, and different types of terrain on the banks of some Galician rivers.

## Environment Setup 🐳

To set up the environment using Docker, you can build an image from the `Dockerfile` provided in the root of the repository.

### Build the Docker Image

```bash
docker build -t stylegan3-multispectral .
```

### Run the Docker Container

Once the image is built, you can run a container. Make sure to mount the necessary volumes to access your data and save the results.

```bash
docker run --gpus all -it --rm -v /path/to/your/data:/workspace/data -v /path/to/your/output:/workspace/out stylegan3-multispectral
```

Inside the container, you can then run the training or image generation scripts.

## Training ⚙️

To train the network, you can use the `train.py` script.

### Usage

```bash
python train.py [OPTIONS]
```

### Example

```bash
python train.py \
    --outdir=~/training-runs \
    --data=~/datasets/oitaven.zip \
    --cfg=stylegan3-t \
    --gpus=2 \
    --batch=64 \
    --gamma=0.125 \
    --cond=True \
    --cls-weight=0.1
```

### Arguments

The following are the most common arguments for the `train.py` script:

| Argument                    | Description                                                                    | Default      |
| --------------------------- | ------------------------------------------------------------------------------ | ------------ |
| `--outdir=<dir>`            | Where to save the results.                                                     | **Required** |
| `--cfg=<config>`            | Base configuration. Allowed values: `stylegan3-t`, `stylegan3-r`, `stylegan2`. | **Required** |
| `--data=<path>`             | Path to the training data (ZIP or directory).                                  | **Required** |
| `--kimg=<int>`              | Total training duration in thousands of images.                                | `25000`      |
| `--gpus=<num>`              | Number of GPUs to use.                                                         | **Required** |
| `--batch=<size>`            | Total batch size.                                                              | **Required** |
| `--gamma=<float>`           | R1 regularization weight.                                                      | **Required** |
| `--cond=<bool>`             | Train conditional model.                                                       | `False`      |
| `--aug=<mode>`              | Augmentation mode (`noaug`, `ada`, `fixed`).                                   | `ada`        |
| `--data-val=<path>`         | Path to the validation data.                                                   | `""`         |
| `--cls-weight=<float>`      | Weight of the classification loss.                                             | `0.0`        |
| `--uniform-class=<bool>`    | Use uniform class labels.                                                      | `False`      |
| `--save-all-snaps=<bool>`   | Save all snapshots during training.                                            | `False`      |
| `--save-all-fakes=<bool>`   | Save all fake image grids during training.                                     | `False`      |
| `--tick=<kimg>`             | How often to print progress in kimg.                                           | `4`          |
| `--snap=<ticks>`            | How often to save snapshots.                                                   | `50`         |
| `--workers=<int>`           | Number of DataLoader worker processes.                                         | `3`          |
| `--judge-model-path=<path>` | Path to the judge model for manifold metrics.                                  | `None`       |

## Generate Images 🖼️

To generate images using a trained model, you can use the `gen_images.py` script.

### Arguments

| Argument           | Description                             | Default      |
| ------------------ | --------------------------------------- | ------------ |
| `--network=<path>` | Path to the network pickle file.        | **Required** |
| `--seeds=<list>`   | List of random seeds (e.g., '0,1,4-6'). | **Required** |
| `--class=<int>`    | Class label to generate.                | `None`       |
| `--outdir=<dir>`   | Directory to save the output images.    | **Required** |
| `--save-images`    | Whether to save the generated images.   | `False`      |

### Example

The following command generates 6 images of class 0, using seeds from 0 to 5, with the selected network.

```bash
python gen_images.py \
    --outdir=out/images \
    --seeds=0-5 \
    --class=0 \
    --network=path/to/your/network.pkl \
    --save-images
```

## License 📄

This project is licensed under the NVIDIA Source Code License for StyleGAN3. See the `LICENSE.txt` file for details.
