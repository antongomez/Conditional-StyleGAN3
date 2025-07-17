"""
Data splitting utilities for multispectral images.
"""

import math
import random
import json
import pickle
import numpy as np

def _filter_valid_patches(truth, centers, image_height, image_width, sizex, sizey):
    """Filter valid patches grouped by class."""
    patches_by_class = dict()

    # Calculate valid boundaries
    xmin = sizex // 2
    xmax = image_height - math.ceil(sizex / 2)
    ymin = sizey // 2
    ymax = image_width - math.ceil(sizey / 2)

    # Filter valid patches
    for patch_idx in centers:
        i = patch_idx // image_height
        j = patch_idx % image_height

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


def _calculate_sample_sizes(total_samples_per_class, train_ratio, val_ratio, batch_size=None):
    """
    Calculate the number of training and validation samples per class.
    
    Args:
        total_samples_per_class (dict): Dictionary with total samples per class
        train_ratio (float): Proportion of samples to use for training
        val_ratio (float): Proportion of samples to use for validation
        batch_size (int, optional): Batch size to align the total validation sample count
        
    Returns:
        tuple: (train_samples_per_class, val_samples_per_class, total_train, total_val)
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
                best_cls = max(val_raw.keys(), key=lambda k: val_raw[k] - val_samples[k])
                val_samples[best_cls] += 1

            total_val = sum(val_samples.values())

    total_train = sum(train_samples.values())
    return train_samples, val_samples, total_train, total_val


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
    print(f"  Total validation samples: {total_val} ({(total_val / total_samples) * 100:.2f}%) "
        f"({'batch_aligned' if batch_size else 'not batch_aligned'})"
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
            f"  Class {class_idx+1:2d} : {len(class_patches):7d} | {train_samples[class_idx]:5d} | "
            f"{val_samples[class_idx]:10d} | {train_samples[class_idx] / len(class_patches) * 100:6.2f}% | "
            f"{val_samples[class_idx] / len(class_patches) * 100:6.2f}%"
        )

    return train, validation, test


def split_dataset(truth, centers, image_height, image_width, patch_width, patch_height, 
                 train_size, val_size, seed=None, batch_size=None):
    """
    Split dataset into training, validation and test sets.

    Args:
        truth: Array with ground truth labels
        centers: List of patch center indices
        image_height, image_width: Image dimensions
        patch_width, patch_height: Patch sizes
        train_size: Training percentage (0-1)
        val_size: Validation percentage (0-1)
        seed: Seed for reproducibility
        batch_size: Optional batch size for alignment

    Returns:
        tuple: (train_indices, validation_indices, test_indices, label_map)
    """
    print("* Selecting training samples")

    # Filter valid patches
    valid_patches = _filter_valid_patches(truth, centers, image_height, image_width, patch_width, patch_height)
    nclases = len(valid_patches)
    print(f"  Total classes with valid patches: {nclases}")

    # Shuffle patches by class
    _shuffle_patches_by_class(valid_patches, seed)

    # Create label mapping
    label_map = _create_label_map(valid_patches)

    # Split into sets
    train, validation, test = _split_patches(valid_patches, train_size, val_size, batch_size)

    return train, validation, test, label_map


def save_split_info(train_indices, val_indices, test_indices, label_map, output_file, 
                   metadata=None, format='json'):
    """
    Save dataset split information to a file.
    
    Args:
        train_indices: List of training sample indices
        val_indices: List of validation sample indices  
        test_indices: List of test sample indices
        label_map: Dictionary mapping original classes to new labels
        output_file: Output file path
        metadata: Optional metadata dictionary
        format: File format ('json', 'pickle', 'npz')
    """
    split_info = {
        'train_indices': train_indices,
        'validation_indices': val_indices,
        'test_indices': test_indices,
        'label_map': label_map,
        'metadata': metadata or {},
        'split_stats': {
            'train_samples': len(train_indices),
            'validation_samples': len(val_indices),
            'test_samples': len(test_indices),
            'total_samples': len(test_indices),
            'num_classes': len(label_map)
        }
    }
    
    if format == 'json':
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        split_info = convert_numpy_types(split_info)
        
        with open(output_file, 'w') as f:
            json.dump(split_info, f, indent=4)
            
    elif format == 'pickle':
        with open(output_file, 'wb') as f:
            pickle.dump(split_info, f)
            
    elif format == 'npz':
        np.savez_compressed(output_file, **split_info)
        
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"* Split information saved to: {output_file}")


def load_split_info(input_file, format='json'):
    """
    Load dataset split information from a file.
    
    Args:
        input_file: Input file path
        format: File format ('json', 'pickle', 'npz')
        
    Returns:
        dict: Dictionary containing split information
    """
    if format == 'json':
        with open(input_file, 'r') as f:
            return json.load(f)
            
    elif format == 'pickle':
        with open(input_file, 'rb') as f:
            return pickle.load(f)
            
    elif format == 'npz':
        data = np.load(input_file, allow_pickle=True)
        return {key: data[key].item() if data[key].ndim == 0 else data[key] for key in data.files}
        
    else:
        raise ValueError(f"Unsupported format: {format}")


def get_split_statistics(split_info):
    """
    Calculate and display statistics for a dataset split.
    
    Args:
        split_info: Dictionary containing split information
        
    Returns:
        dict: Dictionary with calculated statistics
    """
    stats = split_info.get('split_stats', {})
    
    print("Dataset Split Statistics:")
    print(f"  Training samples: {stats.get('train_samples', 0)}")
    print(f"  Validation samples: {stats.get('validation_samples', 0)}")
    print(f"  Test samples: {stats.get('test_samples', 0)}")
    print(f"  Total samples: {stats.get('total_samples', 0)}")
    print(f"  Number of classes: {stats.get('num_classes', 0)}")
    
    if 'label_map' in split_info:
        print("Label mapping:")
        for original, new in split_info['label_map'].items():
            print(f"  Original class {original} -> New label {new}")
    
    return stats