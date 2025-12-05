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
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm

import dnnlib
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
    to_int8: bool = True,
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
        to_int8: Whether to convert the output images to uint8 format.
    Returns:
        imgs: Generated image tensor.
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
    imgs = imgs.permute(0, 2, 3, 1)  # NCHW -> NHWC
    if to_int8:
        imgs = (imgs * 127.5 + 128).clamp(0, 255).to(torch.uint8)

    return imgs.cpu()


def noise_to_images_multiclass(
    G,
    seed: int,
    truncation_psi: float,
    noise_mode: str,
    translate: Tuple[float, float],
    rotate: float,
    classes_idx: int,
    device: torch.device,
    num_images_per_class: int = 1,
    batch_size: int = 64,
    to_int8: bool = True,
):
    """Generate images from the generator using a random seed.
    Args:
        G: The generator model.
        seed: Random seed for generating the latent vector.
        truncation_psi: Truncation psi value.
        noise_mode: Noise mode for the generator.
        translate: Translation vector (x, y).
        rotate: Rotation angle in degrees.
        classes_idx: Class label index.
        device: Device to run the computation on (CPU or GPU).
        num_images_per_class: Number of images to generate per class.
        batch_size: Number of images to generate per batch (default: 64).
        to_int8: Whether to convert the output images to uint8 format.
    Returns:
        imgs: Generated image tensor.
    """
    total_images = num_images_per_class * len(classes_idx)

    # Generate all latent vectors at once
    z_all = torch.from_numpy(np.random.RandomState(seed).randn(total_images, G.z_dim)).to(device)

    # Construct an inverse rotation/translation matrix and pass to the generator.
    if hasattr(G.synthesis, "input"):
        m = make_transform(translate, rotate)
        m = np.linalg.inv(m)
        G.synthesis.input.transform.copy_(torch.from_numpy(m))

    # Construct the label tensor
    labels_all = torch.zeros([total_images, G.c_dim], device=device)
    for i in range(len(classes_idx)):
        labels_all[i * num_images_per_class : (i + 1) * num_images_per_class, classes_idx[i]] = 1

    # Generate images in batches
    imgs_list = []
    for i in tqdm(range(0, total_images, batch_size), desc="Generating images"):
        end_idx = min(i + batch_size, total_images)
        z_batch = z_all[i:end_idx]
        labels_batch = labels_all[i:end_idx]

        imgs_batch = G(z_batch, labels_batch, truncation_psi=truncation_psi, noise_mode=noise_mode)
        imgs_batch = imgs_batch.permute(0, 2, 3, 1)  # NCHW -> NHWC
        if to_int8:
            imgs_batch = (imgs_batch * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        imgs_list.append(imgs_batch.cpu())

    # Concatenar todos os batches
    imgs = torch.cat(imgs_list, dim=0)

    return imgs


# ----------------------------------------------------------------------------


def save_image_grid(img, fname, drange, grid_size, save_rgb=True):
    """Save a grid of images into a single image file."""
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)

    gw, gh = grid_size
    _N, H, W, C = img.shape
    img = img.reshape([gh, gw, H, W, C])
    img = img.transpose(0, 2, 1, 3, 4)
    img = img.reshape([gh * H, gw * W, C])

    # Save a copy keeping the original range
    original_img = img.copy()

    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    assert C in [1, 3, 5], f"Invalid value for C: {C}. Must be one of [1, 3, 5]."
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], "L").save(fname)
    if C == 3:
        PIL.Image.fromarray(img, "RGB").save(fname)
    if C == 5:
        if save_rgb:
            PIL.Image.fromarray(img[:, :, [2, 1, 0]], "RGB").save(fname)  # Select rgb channels
        # Save as raw image with all channels
        save_raw_image(fname.replace(".png", ".raw"), original_img, drange=drange)


# ----------------------------------------------------------------------------


def save_raw_image(filename: str, image: np.ndarray, drange: Tuple[float, float] = (-1, 1)):
    """
    Save a multi-channel image to a .raw file in the format:
    [num_channels, height, width] (uint32) + image data (uint32).

    The image is scaled to the range [0, 65535] (uint16) based on its current range,
    which can be [-1, 1], [0, 1], or [0, 255].

    Args:
        filename (str): path to the output .raw file
        image (np.ndarray): array with shape (height, width, num_channels)
        drange (Tuple[float, float]): range of the input image values
    """
    if image.ndim != 3:
        raise ValueError("Image must have 3 dimensions: (height, width, num_channels)")

    height, width, num_channels = image.shape
    lo, hi = drange

    # Normalize image and convert to uint32
    image = image.astype(np.float32)
    image = (image - lo) * (65535 / (hi - lo))
    image = np.rint(image).clip(0, 65535).astype(np.uint32)

    header = np.array([num_channels, height, width], dtype=np.uint32)

    with open(filename, "wb") as f:
        header.tofile(f)
        image.tofile(f)


# ----------------------------------------------------------------------------


# fmt: off
@click.command()
@click.option("--network", "network_pkl",                       help="Network pickle filename",                                             required=True)
@click.option("--seeds",                                        help="List of random seeds (e.g., '0,1,4-6')",          type=parse_range,   required=True)
@click.option("--trunc", "truncation_psi",                      help="Truncation psi",                                  type=float,         default=1, show_default=True)
@click.option("--class", "class_idx",                           help="Class label (unconditional if not specified)",    type=int,           default=None)
@click.option("--classes",                                      help="List of class labels (e.g., '0,1,4-6')",          type=parse_range,   default=None)
@click.option("--num-images-per-class", "num_images_per_class", help="Number of images per class (only for --classes)", type=int,           default=None)
@click.option("--noise-mode",                                   help="Noise mode",                                      type=click.Choice(["const", "random", "none"]), default="const", show_default=True)
@click.option("--translate",                                    help="Translate XY-coordinate (e.g. '0.3,1')",          type=parse_vec2,    default="0,0", show_default=True, metavar="VEC2")
@click.option("--rotate",                                       help="Rotation angle in degrees",                       type=float,         default=0, show_default=True, metavar="ANGLE")
@click.option("--outdir",                                       help="Where to save the output images",                 type=str,           required=True, metavar="DIR")
@click.option("--save-images", "save_images",                   help="Wheter to save or not the images",                is_flag=True,       default=False, show_default=True)     
@click.option("--no-rgb", "no_rgb",                             help="Avoid saving images in RGB format",               is_flag=True,       default=False, show_default=True) 
@click.option("--no-int8", "no_int8",                           help="Avoid converting images to int8 format",          is_flag=True,       default=False, show_default=True) # fmt: on
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
    num_images_per_class: Optional[int],
    save_images: bool,
    no_rgb: bool,
    no_int8:  bool,
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
        if len(seeds) > 1:
            print(
                f"warn: when classes is specified, the number of seeds is forced to be 1 (got {len(seeds)}). Using the first seed."
            )
            seeds = [seeds[0]]

    if classes is None:
        # Generate images in the same way as it was done previously in the repo.
        images = []
        for seed_idx, seed in enumerate(seeds):
            print("Generating image for seed %d (%d/%d) ..." % (seed, seed_idx + 1, len(seeds)))
            class_imgs = noise_to_images(
                G,
                seed,
                truncation_psi=truncation_psi,
                noise_mode=noise_mode,
                translate=translate,
                rotate=rotate,
                class_idx=class_idx,
                device=device,
                to_int8=not no_int8,
            )
            img = class_imgs[0]
            images.append(img.clone()) # return a copy to avoid issues later on

            if save_images:
                # To save the image, we need to convert it to uint8
                if no_int8:
                    img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)

                if img.shape[-1] == 1:
                    PIL.Image.fromarray(img[:, :, 0].numpy(), "L").save(f"{outdir}/seed{seed:04d}.png")
                elif img.shape[-1] == 3:
                    PIL.Image.fromarray(img.numpy(), "RGB").save(f"{outdir}/seed{seed:04d}.png")
                elif img.shape[-1] == 5:
                    # RGB + NIR + IR -> Save RGB
                    if not no_rgb:
                        PIL.Image.fromarray(img[:, :, [2, 1, 0]].numpy(), "RGB").save(f"{outdir}/seed{seed:04d}.png")
                    # Save as raw image with all channels
                    save_raw_image(f"{outdir}/seed{seed:04d}.raw", img.numpy(), drange=(-1, 1) if not no_int8 else (0, 255))
                else:
                    raise ValueError(f"Unexpected number of channels: {img.shape[-1]}")

        return images

    else:
        if num_images_per_class is None:
            num_images_per_class = len(classes)
        images = noise_to_images_multiclass(
            G,
            seeds[0],
            truncation_psi=truncation_psi,
            noise_mode=noise_mode,
            translate=translate,
            rotate=rotate,
            classes_idx=classes,
            device=device,
            num_images_per_class=num_images_per_class, 
            to_int8=not no_int8,
        )

        if save_images:
            save_image_grid(
                images,
                f"{outdir}/fakes_grid.png",
                drange=(-1, 1) if no_int8 else (0, 255),
                grid_size=(num_images_per_class, len(classes)),
                save_rgb=not no_rgb,
            )

        return images


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
