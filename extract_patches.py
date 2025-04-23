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

    labels = []
    for idx, center in enumerate(tqdm(centers, desc="* Processing patches")):

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
        class_label = truth[center] - 1  # Adjust class label to be zero-indexed
        class_dir = os.path.join(output_dir, f"{class_label:05d}")
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        patch_filename = f"img{idx:08d}.png" if rgb else f"img{idx:08d}.npy"
        patch_path = os.path.join(class_dir, patch_filename)
        if rgb:
            patch_img.save(patch_path)
        else:
            np.save(patch_path, patch_img)

        # Add the patch path and label to the labels list
        relative_path = os.path.relpath(patch_path, output_dir)
        labels.append([relative_path, class_label])

    # Save the labels to a dataset.json file
    dataset_json = {"labels": labels}
    json_path = os.path.join(output_dir, "dataset.json")
    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=4)

    print(f"* Labels saved in: {json_path}")


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
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.filename, args.rgb)
