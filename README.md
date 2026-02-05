# StyleGAN3 for Multispectral Image Classification

# StyleGAN3 for Multispectral Image Classification

This project is an adaptation of the official StyleGAN3 implementation in PyTorch. The main goal is to convert the discriminator into a classifier and combine the adversarial training of StyleGAN3 with a supervised classification task. The objective is to classify multispectral drone images to detect the presence of different tree species, human constructions, water, and different types of terrain on the banks of some Galician rivers.

## Training

To train the network, you can use the `run_training.sh` script located in the `scripts` directory.

### Usage

```bash
bash scripts/run_training.sh [OPTIONS]
```

### Example

```bash
bash scripts/run_training.sh \
    --epochs=100 \
    --filename=oitaven \
    --gpus=2 \
    --batch=64 \
    --gamma=0.125 \
    --cfg=stylegan3-t \
    --tick=10 \
    --snap=5
```

### Arguments

The following are the most common arguments for the `run_training.sh` script:

| Argument | Description | Default |
|---|---|---|
| `--epochs=<num>` | Total number of training epochs. | `0` |
| `--filename=<name>` | The name of the dataset to use (e.g., `oitaven`). | `oitaven` |
| `--gpus=<num>` | Number of GPUs to use for training. | `2` |
| `--batch=<size>` | Total batch size across all GPUs. Must be a power of two. | `128` |
| `--gamma=<float>` | R1 regularization weight. | `0.125` |
| `--cfg=<config>` | Model configuration. Allowed values: `stylegan3-t`, `stylegan3-r`, `stylegan2`. | `stylegan3-t` |
| `--tick=<int>` | How often to print statistics in kimg. | `10` |
| `--snap=<int>` | How often to save network snapshots in kimg. | `1` |
| `--input-path=<path>` | Path to the directory containing the datasets. | `data` |
| `--seed=<int>` | Random seed for dataset splitting. | `""` |
| `--cls-weight=<float>` | Weight for the classification loss. | `0.1` |
| `--no-eval` | If set, disables evaluation at the end of the training. | `False` |

```