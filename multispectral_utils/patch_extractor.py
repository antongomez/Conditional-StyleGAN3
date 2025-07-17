"""
Patch extraction utilities for multispectral images.
"""

import os
import math
import json
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def select_patch(data, patch_size_x, patch_size_y, x, y):
    """
    Extract a patch from the data at the specified coordinates.
    
    Args:
        data: Input data tensor
        patch_size_x: Patch width
        patch_size_y: Patch height
        x: X coordinate of patch center
        y: Y coordinate of patch center
        
    Returns:
        torch.Tensor: Extracted patch
    """
    x1 = x - int(patch_size_x / 2)
    x2 = x + int(math.ceil(patch_size_x / 2))
    y1 = y - int(patch_size_y / 2)
    y2 = y + int(math.ceil(patch_size_y / 2))
    patch = data[:, y1:y2, x1:x2]
    return patch


def is_valid_patch_center(center, image_height, image_width, patch_size, truth=None):
    """
    Check if a patch center is valid for extraction.
    
    Args:
        center: Patch center index
        image_height: Image height
        image_width: Image width
        patch_size: Patch size (assuming square patches)
        truth: Optional ground truth array to check for valid labels
        
    Returns:
        bool: True if the center is valid
    """
    x = center % image_height
    y = center // image_height

    # Check if the center is within valid bounds
    xmin = int(patch_size / 2)
    xmax = image_height - int(math.ceil(patch_size / 2))
    ymin = int(patch_size / 2)
    ymax = image_width - int(math.ceil(patch_size / 2))

    if y < ymin or y > ymax or x < xmin or x > xmax:
        return False
        
    if truth is not None and truth[center] <= 0:
        return False
        
    return True


def _save_patch(idx, patch, class_label, output_dir, rgb=False):
    """
    Save a patch to disk.
    
    Args:
        idx: Patch index
        patch: Patch data
        class_label: Class label
        output_dir: Output directory
        rgb: Whether to save as RGB image
        
    Returns:
        str: Path to saved patch
    """
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


def extract_patches(data, truth, centers, image_height, image_width, patch_size, 
                   output_dir, rgb=False, desc="Processing patches"):
    """
    Extract patches from multispectral data and save them to disk.
    
    Args:
        data: Multispectral image data
        truth: Ground truth labels
        centers: List of patch center indices
        image_height: Image height
        image_width: Image width
        patch_size: Patch size (assuming square patches)
        output_dir: Output directory
        rgb: Whether to save as RGB images
        desc: Description for progress bar
        
    Returns:
        list: List of [relative_path, class_label] pairs
    """
    labels = []
    
    for idx, center in enumerate(tqdm(centers, desc=desc)):
        if not is_valid_patch_center(center, image_height, image_width, patch_size, truth):
            continue

        x = center % image_height
        y = center // image_height

        # Extract patch
        patch = select_patch(data, patch_size, patch_size, x, y)
        class_label = truth[center] - 1  # Convert to zero-based index
        patch_path = _save_patch(idx, patch, class_label, output_dir, rgb)

        # Add the patch path and label to the labels list
        relative_path = os.path.relpath(patch_path, output_dir)
        labels.append([relative_path, class_label])

    return labels


def extract_and_save_patches(data, truth, centers, image_height, image_width, patch_size, 
                           output_dir, rgb=False, desc="Processing patches"):
    """
    Extract patches and save them along with a dataset.json file.
    
    Args:
        data: Multispectral image data
        truth: Ground truth labels
        centers: List of patch center indices
        image_height: Image height
        image_width: Image width
        patch_size: Patch size (assuming square patches)
        output_dir: Output directory
        rgb: Whether to save as RGB images
        desc: Description for progress bar
        
    Returns:
        str: Path to the dataset.json file
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract patches
    labels = extract_patches(data, truth, centers, image_height, image_width, 
                           patch_size, output_dir, rgb, desc)

    # Save the labels to a dataset.json file
    dataset_json = {"labels": labels}
    json_path = os.path.join(output_dir, "dataset.json")
    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=4)

    print(f"* Labels saved in: {json_path}")
    return json_path


def batch_extract_patches(data, truth, train_centers, val_centers, test_centers, 
                         image_height, image_width, patch_size, base_output_dir, 
                         rgb=False):
    """
    Extract patches for train, validation, and test sets.
    
    Args:
        data: Multispectral image data
        truth: Ground truth labels
        train_centers: Training patch centers
        val_centers: Validation patch centers
        test_centers: Test patch centers
        image_height: Image height
        image_width: Image width
        patch_size: Patch size
        base_output_dir: Base output directory
        rgb: Whether to save as RGB images
        
    Returns:
        dict: Dictionary with paths to dataset.json files for each split
    """
    splits = [
        ("train", train_centers, "Processing training patches"),
        ("validation", val_centers, "Processing validation patches"),
        ("test", test_centers, "Processing test patches")
    ]
    
    json_paths = {}
    
    for split_name, centers, desc in splits:
        split_dir = os.path.join(base_output_dir, split_name)
        json_path = extract_and_save_patches(
            data, truth, centers, image_height, image_width, 
            patch_size, split_dir, rgb, desc
        )
        json_paths[split_name] = json_path
    
    return json_paths


def load_patch_dataset(dataset_json_path):
    """
    Load patch dataset from a dataset.json file.
    
    Args:
        dataset_json_path: Path to the dataset.json file
        
    Returns:
        dict: Dictionary containing the dataset information
    """
    with open(dataset_json_path, 'r') as f:
        dataset = json.load(f)
    
    dataset_dir = os.path.dirname(dataset_json_path)
    
    # Convert relative paths to absolute paths
    for label_info in dataset['labels']:
        label_info[0] = os.path.join(dataset_dir, label_info[0])
    
    return dataset


def get_patch_statistics(dataset_json_path):
    """
    Get statistics for a patch dataset.
    
    Args:
        dataset_json_path: Path to the dataset.json file
        
    Returns:
        dict: Dictionary with dataset statistics
    """
    dataset = load_patch_dataset(dataset_json_path)
    labels = [label_info[1] for label_info in dataset['labels']]
    
    unique_labels = set(labels)
    class_counts = {label: labels.count(label) for label in unique_labels}
    
    stats = {
        'total_patches': len(labels),
        'num_classes': len(unique_labels),
        'class_counts': class_counts,
        'class_distribution': {label: count/len(labels) for label, count in class_counts.items()}
    }
    
    return stats