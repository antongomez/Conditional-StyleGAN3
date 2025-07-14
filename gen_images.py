# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt

import legacy

# ----------------------------------------------------------------------------


def parse_range(s: Union[str, List]) -> List[int]:
    """Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    """
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r"^(\d+)-(\d+)$")
    for p in s.split(","):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


# ----------------------------------------------------------------------------


def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    """Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    """
    if isinstance(s, tuple):
        return s
    parts = s.split(",")
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f"cannot parse 2-vector {s}")


# ----------------------------------------------------------------------------


def make_transform(translate: Tuple[float, float], angle: float):
    m = np.eye(3)
    s = np.sin(angle / 360.0 * np.pi * 2)
    c = np.cos(angle / 360.0 * np.pi * 2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m


# ----------------------------------------------------------------------------


def noise_to_images(
    G,
    seed: int,
    truncation_psi: float,
    noise_mode: str,
    translate: Tuple[float, float],
    rotate: float,
    class_idx: int,
    device: torch.device,
    num_images: int = 1,
):
    """Generate images from the generator using a random seed.
    Args:
        G: The generator model.
        seed: Random seed for generating the latent vector.
        truncation_psi: Truncation psi value.
        noise_mode: Noise mode for the generator.
        translate: Translation vector (x, y).
        rotate: Rotation angle in degrees.
        class_idx: Class label index.
        device: Device to run the computation on (CPU or GPU).
        num_images: Number of images to generate.
    Returns:
        img: Generated image tensor.
    """
    z = torch.from_numpy(np.random.RandomState(seed).randn(num_images, G.z_dim)).to(device)

    # Construct an inverse rotation/translation matrix and pass to the generator.  The
    # generator expects this matrix as an inverse to avoid potentially failing numerical
    # operations in the network.
    if hasattr(G.synthesis, "input"):
        m = make_transform(translate, rotate)
        m = np.linalg.inv(m)
        G.synthesis.input.transform.copy_(torch.from_numpy(m))

    # Construct the label tensor
    labels = torch.zeros([num_images, G.c_dim], device=device)
    if class_idx is not None:
        labels[:, class_idx] = 1
    else:
        labels = None

    imgs = G(z, labels, truncation_psi=truncation_psi, noise_mode=noise_mode)
    imgs = (imgs.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

    return imgs.cpu()


# ----------------------------------------------------------------------------


def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, H, W, C = img.shape
    img = img.reshape([gh, gw, H, W, C])
    img = img.transpose(0, 2, 1, 3, 4)
    img = img.reshape([gh * H, gw * W, C])

    if C == 5:
        img = img[:, :, [2, 1, 0]]  # Select rgb channels
        C = 3

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], "L").save(fname)
    if C == 3:
        PIL.Image.fromarray(img, "RGB").save(fname)


# ----------------------------------------------------------------------------


@click.command()
@click.option("--network", "network_pkl", help="Network pickle filename", required=True)
@click.option("--seeds", type=parse_range, help="List of random seeds (e.g., '0,1,4-6')", required=True)
@click.option("--trunc", "truncation_psi", type=float, help="Truncation psi", default=1, show_default=True)
@click.option("--class", "class_idx", type=int, help="Class label (unconditional if not specified)")
@click.option("--classes", type=parse_range, help="List of class labels (e.g., '0,1,4-6')", default=None)
@click.option(
    "--noise-mode",
    help="Noise mode",
    type=click.Choice(["const", "random", "none"]),
    default="const",
    show_default=True,
)
@click.option(
    "--translate",
    help="Translate XY-coordinate (e.g. '0.3,1')",
    type=parse_vec2,
    default="0,0",
    show_default=True,
    metavar="VEC2",
)
@click.option("--rotate", help="Rotation angle in degrees", type=float, default=0, show_default=True, metavar="ANGLE")
@click.option("--outdir", help="Where to save the output images", type=str, required=True, metavar="DIR")
def generate_images(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    translate: Tuple[float, float],
    rotate: float,
    class_idx: Optional[int],
    classes: Optional[List[int]],
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained AFHQv2 model ("Ours" in Figure 1, left).
    python gen_images.py --outdir=out --trunc=1 --seeds=2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl

    \b
    # Generate uncurated images with truncation using the MetFaces-U dataset
    python gen_images.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-metfacesu-1024x1024.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device("cuda")
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)  # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # Labels.
    if G.c_dim != 0:
        if class_idx is None and classes is None:
            raise click.ClickException("Must specify class label with --class when using a conditional network")
    else:
        if class_idx is not None:
            print("warn: --class=lbl ignored when running on an unconditional network")

    if classes is not None:
        if len(classes) != len(seeds):
            print(
                "warn: --seeds and --classes should have the same length. Ensuring they are equal by using classes as seeds."
            )
            seeds = classes

    if classes is None:
        # Generate images in the same way as it was done previously in the repo.
        for seed_idx, seed in enumerate(seeds):
            print("Generating image for seed %d (%d/%d) ..." % (seed, seed_idx, len(seeds)))
            class_imgs = noise_to_images(
                G,
                seed,
                truncation_psi=truncation_psi,
                noise_mode=noise_mode,
                translate=translate,
                rotate=rotate,
                class_idx=class_idx,
                device=device,
            )
            img = class_imgs[0]
            if img.shape[-1] == 1:
                PIL.Image.fromarray(img[:, :, 0].numpy(), "L").save(f"{outdir}/seed{seed:04d}.png")
            else:
                PIL.Image.fromarray(img.numpy(), "RGB").save(f"{outdir}/seed{seed:04d}.png")
    else:
        # Generate as many images as there are classes. Use a different seed for each class.
        all_images = []
        for class_idx, seed in tqdm(
            zip(classes, seeds), desc="Generating images for each class and seed", total=len(classes)
        ):
            class_imgs = noise_to_images(
                G,
                seed,
                truncation_psi=truncation_psi,
                noise_mode=noise_mode,
                translate=translate,
                rotate=rotate,
                class_idx=class_idx,
                device=device,
                num_images=len(classes),  # ensure the grid is square
            )
            all_images.append(class_imgs)

        save_image_grid(
            torch.cat(all_images, dim=0),
            f"{outdir}/fakes_grid.png",
            drange=(0, 255),
            grid_size=(len(classes), len(classes)),
        )


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
