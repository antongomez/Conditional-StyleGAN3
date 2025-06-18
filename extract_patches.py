import os, math, random, argparse, json
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn import preprocessing
from PIL import Image

# -----------------------------------------------------------------
# FUNCTIONS TO READ DATASETS AND SELECT SAMPLES
# -----------------------------------------------------------------


def read_raw(fichero):
    (B, H, V) = np.fromfile(fichero, count=3, dtype=np.uint32)
    datos = np.fromfile(fichero, count=B * H * V, offset=3 * 4, dtype=np.int32)
    print("* Read dataset:", fichero)
    print("  B:", B, "H:", H, "V:", V)
    print("  Read:", len(datos))

    # normalize data to [-1, 1]
    datos = datos.astype(np.float64)
    preprocessing.minmax_scale(datos, feature_range=(-1, 1), copy=False)

    datos = datos.reshape(V, H, B)
    datos = torch.FloatTensor(datos)
    return datos, H, V, B


def read_pgm(file):
    try:
        pgmf = open(file, "rb")
    except IOError:
        print("Cannot open", file)
    else:
        assert pgmf.readline().decode() == "P5\n"
        line = pgmf.readline().decode()
        while line[0] == "#":
            line = pgmf.readline().decode()
        (H, V) = line.split()
        H = int(H)
        V = int(V)
        depth = int(pgmf.readline().decode())
        assert depth <= 255
        raster = []
        for i in range(H * V):
            raster.append(ord(pgmf.read(1)))
        print("* Read GT:", file)
        print("  H:", H, "V:", V, "depth:", depth)
        print("  Read:", len(raster))
        return raster, H, V


def read_seg_centers(file):
    (H, V, nseg) = np.fromfile(file, count=3, dtype=np.uint32)
    data = np.fromfile(file, count=H * V, offset=3 * 4, dtype=np.uint32)
    print("* Read centers:", file)
    print("  H:", H, "V:", V, "nseg:", nseg)
    print("  Read:", len(data))
    return data, H, V, nseg


def select_patch(data, patch_size_x, patch_size_y, x, y):
    x1 = x - int(patch_size_x / 2)
    x2 = x + int(math.ceil(patch_size_x / 2))
    y1 = y - int(patch_size_y / 2)
    y2 = y + int(math.ceil(patch_size_y / 2))
    patch = data[:, y1:y2, x1:x2]
    return patch


# -----------------------------------------------------------------
# FUNCTIONS TO SELECT TRAINING SAMPLES
# -----------------------------------------------------------------


def _filter_valid_patches(truth, center, H, V, sizex, sizey):
    """Filter valid patches grouped by class."""
    patches_by_class = dict()

    # Calculate valid boundaries
    xmin = sizex // 2
    xmax = H - math.ceil(sizex / 2)
    ymin = sizey // 2
    ymax = V - math.ceil(sizey / 2)

    # Filter valid patches
    for patch_idx in center:
        i = patch_idx // H
        j = patch_idx % H

        # Check if within boundaries
        if ymin <= i <= ymax and xmin <= j <= xmax and truth[patch_idx] > 0:
            class_idx = truth[patch_idx] - 1
            if class_idx not in patches_by_class:
                patches_by_class[class_idx] = []
            patches_by_class[class_idx].append(patch_idx)

    return patches_by_class


def _shuffle_patches_by_class(patches_by_class, seed):
    """Shuffle patches for each class reproducibly."""
    base_seed = seed if seed is not None else 0

    for class_idx in patches_by_class.keys():
        random.Random(base_seed + class_idx).shuffle(patches_by_class[class_idx])


def _create_label_map(patches_by_class):
    """Create label mapping for classes with samples."""
    label_map = {}
    label_counter = 1

    for class_idx in sorted(patches_by_class.keys()):
        label_map[class_idx + 1] = label_counter
        label_counter += 1

    return label_map


def _split_patches(patches_by_class, train_size, val_size, batch_size=None):
    """Split patches into training, validation and test sets."""
    train = []
    validation = []
    test = []

    total_samples_per_class = {k: len(v) for k, v in patches_by_class.items()}
    total_samples = sum(total_samples_per_class.values())

    # Calculate number of samples per class
    train_samples, val_samples, total_train, total_val = _calculate_sample_sizes(
        total_samples_per_class, train_size, val_size, batch_size
    )

    print(f"  Total train samples: {total_train} ({(total_train / total_samples) * 100:.2f}%)")
    print(
        f"  Total validation samples: {total_val} ({(total_val / total_samples) * 100:.2f}%) ({'batch_aligned' if batch_size else 'not batch_aligned'})"
    )
    print("  Class    : seg.tot | train | validation | train % |   val %")

    for class_idx in sorted(patches_by_class.keys()):
        class_patches = patches_by_class[class_idx]

        if len(class_patches) == 0:
            continue

        # Assign samples
        for j, patch_idx in enumerate(class_patches):
            test.append(patch_idx)  # All samples go to test

            if j < train_samples[class_idx]:
                train.append(patch_idx)
            elif j < train_samples[class_idx] + val_samples[class_idx]:
                validation.append(patch_idx)

        # Show statistics
        print(
            f"  Class {class_idx+1:2d} : {len(class_patches):7d} | {train_samples[class_idx]:5d} | {val_samples[class_idx]:10d} | {train_samples[class_idx] / len(class_patches) * 100:6.2f}% | {val_samples[class_idx] / len(class_patches) * 100:6.2f}%"
        )

    return train, validation, test


import math


def _calculate_sample_sizes(total_samples_per_class, train_ratio, val_ratio, batch_size=None):
    """
    Calculate the number of training and validation samples per class, given the total samples per class
    and the desired train/validation split ratios.

    If a batch size is provided, the total number of validation samples will be adjusted to be a multiple
    of the batch size. This is done by:
      1. Computing the initial number of validation samples per class (using floor).
      2. Calculating how many more samples are needed to reach a multiple of the batch size.
      3. Distributing those additional samples to classes with the highest remainder
         from val_size * class_total to floor(val_size * class_total).

    Args:
        total_samples_per_class (dict[int, int]): Dictionary with total samples per class.
        train_ratio (float): Proportion of samples to use for training (e.g., 0.15).
        val_ratio (float): Proportion of samples to use for validation (e.g., 0.05).
        batch_size (int, optional): Batch size to align the total validation sample count with.

    Returns:
        tuple:
            - train_samples_per_class (dict[int, int]): Training samples per class.
            - val_samples_per_class (dict[int, int]): Validation samples per class.
            - total_train (int): Total number of training samples.
            - total_val (int): Total number of validation samples (adjusted if needed).
    """
    train_samples = {}
    val_samples = {}
    val_raw = {}

    for cls, total in total_samples_per_class.items():
        train_samples[cls] = math.floor(train_ratio * total)
        val_raw[cls] = val_ratio * total
        val_samples[cls] = math.floor(val_raw[cls])

    total_val = sum(val_samples.values())

    if batch_size is not None:
        remainder = total_val % batch_size
        if remainder != 0:
            to_add = batch_size - remainder
            print(f"  Adjusting validation samples to be a multiple of {batch_size}. Adding {to_add} samples.")

            for _ in range(to_add):
                # Comute the remainders for each class and get the maximum. In each step, the class with the highest remainder will receive the extra sample.
                best_cls = max(val_raw.keys(), key=lambda k: val_raw[k] - val_samples[k])
                val_samples[best_cls] += 1

            total_val = sum(val_samples.values())

    total_train = sum(train_samples.values())
    return train_samples, val_samples, total_train, total_val


def select_training_samples_seg(truth, center, H, V, sizex, sizey, train_size, val_size, seed=None, batch_size=None):
    """
    Split patches into training, validation and test sets.

    Args:
        truth: Array with ground truth labels
        center: List of patch center indices
        H, V: Image dimensions
        sizex, sizey: Patch sizes
        train_size: Training percentage (0-1)
        val_size: Validation percentage (0-1)
        seed: Seed for reproducibility

    Returns:
        tuple: (train_indices, validation_indices, test_indices, label_map)
    """
    print("* Selecting training samples")

    # Initial configuration
    valid_patches = _filter_valid_patches(truth, center, H, V, sizex, sizey)
    nclases = len(valid_patches)
    print(f"  Total classes with valid patches: {nclases}")

    # Shuffle patches by class
    _shuffle_patches_by_class(valid_patches, seed)

    # Create label mapping
    label_map = _create_label_map(valid_patches)

    # Split into sets
    train, validation, test = _split_patches(valid_patches, train_size, val_size, batch_size)

    return train, validation, test, label_map


# -----------------------------------------------------------------
#
# -----------------------------------------------------------------


def _read_and_save_patch(idx, patch, class_label, rgb, output_dir):
    if rgb:
        rgb_patch = torch.index_select(patch, dim=0, index=torch.tensor([2, 1, 0]))
        # Normalize the patch to [0, 1]
        rgb_patch = (rgb_patch - rgb_patch.min()) / (rgb_patch.max() - rgb_patch.min())
        patch_img = Image.fromarray((rgb_patch.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    else:
        # Normalize the patch to [0, 1]
        patch = (patch - patch.min()) / (patch.max() - patch.min())
        patch_img = patch.permute(1, 2, 0).numpy()
        patch_img = (patch_img * 255).astype(np.uint8)

    # Save patch in the corresponding class folder
    class_dir = os.path.join(output_dir, f"{class_label:05d}")
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    patch_filename = f"img{idx:08d}.png" if rgb else f"img{idx:08d}.npy"
    patch_path = os.path.join(class_dir, patch_filename)
    if rgb:
        patch_img.save(patch_path)
    else:
        np.save(patch_path, patch_img)

    return patch_path


def process_dataset(data, truth, centers, H, V, PATCH_SIZE, output_dir, rgb, desc="* Processing patches"):
    labels = []
    for idx, center in enumerate(tqdm(centers, desc=desc)):

        x = center % H
        y = center // H

        # Check if the center is within valid bounds
        xmin = int(PATCH_SIZE / 2)
        xmax = H - int(math.ceil(PATCH_SIZE / 2))
        ymin = int(PATCH_SIZE / 2)
        ymax = V - int(math.ceil(PATCH_SIZE / 2))

        if y < ymin or y > ymax or x < xmin or x > xmax:
            continue
        if truth[center] <= 0:  # Skip centers with invalid class labels
            continue

        # Extract patch
        patch = select_patch(data, PATCH_SIZE, PATCH_SIZE, x, y)
        class_label = truth[center] - 1  # Convert to zero-based index
        patch_path = _read_and_save_patch(idx, patch, class_label, rgb, output_dir)

        # Add the patch path and label to the labels list
        relative_path = os.path.relpath(patch_path, output_dir)
        labels.append([relative_path, class_label])

    # Save the labels to a dataset.json file
    dataset_json = {"labels": labels}
    json_path = os.path.join(output_dir, "dataset.json")
    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=4)

    print(f"* Labels saved in: {json_path}")


# -----------------------------------------------------------------
# MAIN SCRIPT
# -----------------------------------------------------------------


def main(input_dir, output_dir, filename, rgb):
    # Construct input file paths
    raw_file = os.path.join(input_dir, f"{filename}.raw")
    gt_file = os.path.join(input_dir, f"{filename}.pgm")
    centers_file = os.path.join(input_dir, f"seg_{filename}_wp_centers.raw")

    # Patch size
    PATCH_SIZE = 32  # Square patch size

    # Read data, ground truth, and segment centers
    data, H, V, _B = read_raw(raw_file)
    truth, H_gt, V_gt = read_pgm(gt_file)
    centers, _H_centers, _V_centers, _nseg = read_seg_centers(centers_file)

    # Validate dimensions
    assert H == H_gt and V == V_gt, "Data and GT dimensions do not match"

    # Transpose data to band-vector format for convolutions
    data = np.transpose(data, (2, 0, 1))

    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Split centers into training, validation, and test sets
    train_size = args.train_size
    validation_size = args.validation_size

    train_centers, validation_centers, test_centers, label_map = select_training_samples_seg(
        truth,
        centers,
        H,
        V,
        PATCH_SIZE,
        PATCH_SIZE,
        train_size,
        validation_size,
        seed=args.seed,
        batch_size=args.batch_size,
    )

    print("* Label map:")
    for class_idx, label in label_map.items():
        print(f"  Class {class_idx:2d} => Label {label:2d}")

    process_dataset(
        data,
        truth,
        train_centers,
        H,
        V,
        PATCH_SIZE,
        os.path.join(output_dir, "train"),
        rgb,
        desc="* Processing training patches",
    )
    process_dataset(
        data,
        truth,
        validation_centers,
        H,
        V,
        PATCH_SIZE,
        os.path.join(output_dir, "validation"),
        rgb,
        desc="* Processing validation patches",
    )
    process_dataset(
        data,
        truth,
        test_centers,
        H,
        V,
        PATCH_SIZE,
        os.path.join(output_dir, "test"),
        rgb,
        desc="* Processing test patches",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract patches from hyperspectral images using segment centers")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/oitaven/",
        help="Input directory containing the raw, pgm, and segment center files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/oitaven/patches/",
        help="Output directory to save the patches",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="oitaven",
        help="Base filename (without extension) of the raw, pgm, and segment center files",
    )
    parser.add_argument(
        "--rgb",
        action="store_true",
        default=False,
        help="Whether to save the patches as RGB images",
    )
    parser.add_argument(
        "--train_size",
        type=float,
        default=0.15,
        help="Size of the train set (default: 0.15)",
    )
    parser.add_argument(
        "--validation_size",
        type=float,
        default=0.05,
        help="Size of the validation set (default: 0.05)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Size of the batch for processing patches (default: None)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (default: None)")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.filename, args.rgb)
