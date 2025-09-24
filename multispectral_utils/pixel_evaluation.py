import os

from tqdm import tqdm
import numpy as np

import torch

import dnnlib
import legacy

from .data_readers import load_multispectral_dataset
from .data_splitter import load_split_info

def init_dataset_kwargs(data):
    dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data, use_labels=True, max_size=None, xflip=False)
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
    dataset_kwargs.resolution = dataset_obj.resolution # Be explicit about resolution.
    dataset_kwargs.use_labels = dataset_obj.has_labels # Be explicit about labels.
    dataset_kwargs.max_size = len(dataset_obj) # Be explicit about dataset size.
    return dataset_kwargs, dataset_obj.name

def build_test_dataset(test_dataset_kwargs: dict = None, data_loader_kwargs: dict = None, batch_size: int = 64):
    test_dataset = dnnlib.util.construct_class_by_name(
        **test_dataset_kwargs
    )  # subclass of training.dataset.Dataset
    print("Test dataset built")

    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        **data_loader_kwargs,
    )
    print("Test dataset iterator built with batch size:", batch_size)

    return test_dataset, test_dataloader

def build_discriminator(network_pkl: str, device: str = "cuda"):
    """Load a pretrained StyleGAN discriminator from a pickle file."""
    print(f'Loading networks from "{network_pkl}"...')
    device = torch.device(device)
    with dnnlib.util.open_url(network_pkl) as f:
        D = legacy.load_network_pkl(f)["D"].to(device)  # type: ignore
    D.eval()
    print("Discriminator network loaded.")
    return D, device

def load_dataset_and_split_info(input_dir, output_dir, filename, split_format, read_raw_data=True):
    """Load dataset information and data split configuration."""
    dataset = load_multispectral_dataset(input_dir, filename, read_raw_data=read_raw_data)
    split_file = os.path.join(output_dir, f"split_info.{split_format}")
    split_info = load_split_info(split_file, split_format)
    
    return {
        'dataset': dataset,
        'split_info': split_info
    }

def predict_patches_batch(dataloader, D, device, label_map=None, show_progress=False):
    """Predict classes for all patches using the discriminator."""
    predictions = []
    batch_size = dataloader.batch_size
    
    with torch.no_grad():
        if show_progress:
            dataloader = tqdm(dataloader, ncols=60, desc="Evaluating")
        
        for image_batch, label_batch in dataloader:
            # Normalization
            image_batch = image_batch.to(torch.float32) / 127.5 - 1
            actual_batch_size = image_batch.shape[0]
            
            # Padding if necessary
            if actual_batch_size < batch_size:
                pad_size = batch_size - actual_batch_size
                pad_images = torch.zeros((pad_size, *image_batch.shape[1:]), 
                                       dtype=image_batch.dtype, device=image_batch.device)
                image_batch = torch.cat([image_batch, pad_images], dim=0)
                
                pad_labels = torch.zeros((pad_size, *label_batch.shape[1:]), 
                                       dtype=label_batch.dtype, device=label_batch.device)
                label_batch = torch.cat([label_batch, pad_labels], dim=0)
            
            # Move to device
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)
            
            # Prediction
            _, classification_logits = D(image_batch, label_batch, update_emas=False)
            
            # Remove padding
            if actual_batch_size < batch_size:
                classification_logits = classification_logits[:actual_batch_size]
            
            # Get predictions
            predicted_classes = torch.argmax(classification_logits, dim=1).cpu().numpy()
            if label_map is not None:
                predicted_classes = np.array([label_map[x] for x in predicted_classes])
            predictions.extend(predicted_classes + 1)  # +1 to adjust indexing
    
    return np.array(predictions)

def generate_pixel_classification_map(predictions, test_indices, centers, segmentation_data, 
                                    train_indices, validation_indices, image_size, show_progress=False):
    """Generate pixel classification map and remove train/val centers."""
    pixel_output = np.zeros(image_size, dtype=np.uint8)
    
    # Assign predictions to test centers
    pixel_output[test_indices] = predictions

    # Generate full classification map by assigning center predictions to pixels
    if show_progress:
        dataloader = tqdm(range(image_size), desc="Generating classification map")
    else:
        dataloader = range(image_size)

    for i in dataloader:
        pixel_output[i] = pixel_output[centers[segmentation_data[i]]]
    
    # if show_progress:
    #     print("Generating classification map...")
    
    # # Create mask for non-center pixels
    # all_indices = np.arange(image_size)
    # center_mask = np.zeros(image_size, dtype=bool)
    # center_mask[centers] = True
    # non_center_indices = all_indices[~center_mask]
    
    # # Assign classifications using vectorized operations
    # pixel_output[non_center_indices] = pixel_output[centers[segmentation_data[non_center_indices]]]
    
    # Remove train and validation centers (vectorized operations)
    pixel_output[train_indices] = 0
    pixel_output[validation_indices] = 0
    
    return pixel_output

def calculate_metrics_vectorized(pixel_output, truth, num_classes, label_map=None, show_progress=False):
    """Calculate OA and AA metrics using vectorized operations."""
    if show_progress:
        print("Calculating metrics...")

    # Ensure truth is a numpy array
    if not isinstance(truth, np.ndarray):
        truth = np.array(truth)
    
    # Create masks for valid pixels
    valid_mask = (pixel_output != 0) & (truth != 0)
    valid_pixels = np.sum(valid_mask)

    if label_map:
        class_idxs = [class_idx+1 for class_idx in label_map.values()]
    else:
        class_idxs = list(range(1, num_classes + 1))
    
    if valid_pixels == 0:
        return 0.0, 0.0, dict((class_id, 0.0) for class_id in class_idxs)
    
    # Get predictions and truths for valid pixels
    valid_predictions = pixel_output[valid_mask]
    valid_truth = truth[valid_mask]
    
    # Calculate OA
    correct_predictions = (valid_predictions == valid_truth)
    OA = np.sum(correct_predictions) / valid_pixels
    
    # Calculate AA using vectorized operations
    class_accuracies = {}
    
    for class_id in class_idxs:
        class_mask = (valid_truth == class_id)
        class_total = np.sum(class_mask)
        
        if class_total > 0:
            class_correct = np.sum(correct_predictions & class_mask)
            class_accuracies[class_id] = class_correct / class_total
        else:
            class_accuracies[class_id] = 0.0
    
    # Calculate average AA (excluding class 0)
    AA = np.mean(list(class_accuracies.values()))
    
    return OA, AA, class_accuracies

import numpy as np
import matplotlib.pyplot as plt

def compare_pixel_outputs(arr1, arr2, shape=None, save_path=None, max_diff_report=10):
    """
    Compare two pixel output arrays and report differences.
    
    Args:
        arr1 (np.ndarray): First pixel output array.
        arr2 (np.ndarray): Second pixel output array.
        shape (tuple): Optional (height, width) to reshape for visualization.
        save_path (str): If provided, saves a mismatch heatmap image at this path.
        max_diff_report (int): Max number of individual mismatches to print.
    """
    if arr1.shape != arr2.shape:
        raise ValueError(f"Shape mismatch: {arr1.shape} vs {arr2.shape}")

    diff_mask = arr1 != arr2
    num_diffs = np.count_nonzero(diff_mask)

    if num_diffs == 0:
        print("✅ Pixel output arrays match exactly.")
        return

    print(f"❌ Found {num_diffs} differing pixels out of {arr1.size} ({num_diffs / arr1.size:.6%})")

    # Report individual differences (limited)
    diff_indices = np.where(diff_mask)[0]
    for i in diff_indices[:max_diff_report]:
        print(f"Index {i}: arr1={arr1[i]} arr2={arr2[i]}")

    # Class frequency of mismatches
    unique_diffs = np.unique(np.stack((arr1[diff_mask], arr2[diff_mask]), axis=1), axis=0)
    print("\nSample of class mismatches (arr1 vs arr2):")
    print(unique_diffs)

    # Save mismatch heatmap if requested
    if save_path and shape:
        mismatch_map = diff_mask.reshape(shape)
        plt.figure(figsize=(10, 8))
        plt.imshow(mismatch_map, cmap='Reds')
        plt.title("Mismatch Map (pixel_output ≠ pixel_output_2)")
        plt.colorbar(label="Mismatch")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"📸 Mismatch heatmap saved to {save_path}")



def calculate_pixel_accuracy(input_dir, output_dir, filename, split_format, dataloader, D, device, label_map=None, show_progress=False):
    """
    Computes pixel-wise accuracy from classifier predictions over image patches.
    """

    # Load multispectral image data and segmentation
    dataset = load_multispectral_dataset(input_dir, filename, read_raw_data=False)
    centers = dataset['centers']
    truth = dataset['truth']
    segmentation_data = dataset['segmentation_data']
    image_height = dataset['dimensions']['height']
    image_width = dataset['dimensions']['width']

    # Load train/val/test split info and metadata
    split_file = os.path.join(output_dir, f"split_info.{split_format}")
    split_info = load_split_info(split_file, split_format)
    train_indices = split_info['train_indices']
    validation_indices = split_info['validation_indices']
    test_indices = split_info['test_indices']
    num_classes = split_info['split_stats']['num_classes']

    # Get batch size from dataloader
    batch_size = dataloader.batch_size

    # Initialize output array
    total_centers = 0
    pixel_output = np.zeros(image_height * image_width, dtype=np.uint8)

    with torch.no_grad():
        if show_progress:
            dataloader = tqdm(dataloader, desc="Evaluating")

        for image_batch, label_batch in dataloader:
            # Normalize inputs
            image_batch = image_batch.to(torch.float32) / 127.5 - 1
            actual_batch_size = image_batch.shape[0]

            # Pad batch if smaller than expected
            if actual_batch_size < batch_size:
                pad_size = batch_size - actual_batch_size
                pad_images = torch.zeros((pad_size, *image_batch.shape[1:]), dtype=image_batch.dtype, device=image_batch.device)
                pad_labels = torch.zeros((pad_size, *label_batch.shape[1:]), dtype=label_batch.dtype, device=label_batch.device)
                image_batch = torch.cat([image_batch, pad_images], dim=0)
                label_batch = torch.cat([label_batch, pad_labels], dim=0)

            # Move to GPU
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)

            # Get predictions from discriminator
            _, classification_logits = D(image_batch, label_batch, update_emas=False)

            # Remove padded entries
            if actual_batch_size < batch_size:
                classification_logits = classification_logits[:actual_batch_size]
                label_batch = label_batch[:actual_batch_size]

            # Get predicted class indices
            predicted_classes = torch.argmax(classification_logits, dim=1)
            ground_truth = torch.argmax(label_batch, dim=1)

            # Store predictions in flat output map
            predicted = predicted_classes.cpu()
            for i in range(len(predicted)):
                pixel_output[test_indices[total_centers + i]] = np.uint8(predicted[i] + 1)
            total_centers += ground_truth.size(0)


    # Generate full classification map by assigning center predictions to pixels
    if show_progress:
        dataloader = tqdm(range(image_height * image_width), desc="Generating classif.map")
    else:
        dataloader = range(image_height * image_width)

    for i in dataloader:
        pixel_output[i] = pixel_output[centers[segmentation_data[i]]]

    # Remove training and validation labels from the output
    for i in train_indices:
        pixel_output[i] = 0
    for i in validation_indices:
        pixel_output[i] = 0

    # Compute accuracy metrics (OA, AA, class-wise)
    if show_progress:
        dataloader = tqdm(range(len(pixel_output)), desc="Calculating accuracy on pixels")
    else:
        dataloader = range(len(pixel_output))

    correct = 0
    total = 0
    AA = 0
    OA = 0
    class_correct = [0] * (num_classes + 1)
    class_total = [0] * (num_classes + 1)
    class_accuracies = [0] * (num_classes + 1)

    for i in dataloader:
        if pixel_output[i] == 0 or truth[i] == 0:
            continue
        pixel_truth = truth[i]
        total += 1
        class_total[pixel_truth] += 1
        if pixel_output[i] == pixel_truth:
            correct += 1
            class_correct[pixel_truth] += 1

    for i in range(1, num_classes + 1):
        if class_total[i] != 0:
            class_accuracies[i] = class_correct[i] / class_total[i]
        else:
            class_accuracies[i] = 0
        AA += class_accuracies[i]

    OA = correct / total
    AA = AA / num_classes

    return OA, AA, class_accuracies

def calculate_pixel_accuracy_optimized(input_dir, output_dir, filename, split_format, 
                                     dataloader, D, device, label_map=None, show_progress=False):
    """
    Optimized version of calculate_pixel_accuracy with vectorized operations
    and improved modularization.
    """
    # 1. Load data
    data_info = load_dataset_and_split_info(input_dir, output_dir, filename, split_format, read_raw_data=False)
    dataset = data_info['dataset']
    split_info = data_info['split_info']
    
    # Extract necessary information
    centers = dataset['centers']
    truth = dataset['truth']
    segmentation_data = dataset['segmentation_data']
    image_height = dataset['dimensions']['height']
    image_width = dataset['dimensions']['width']
    image_size = image_height * image_width
    
    train_indices = split_info['train_indices']
    validation_indices = split_info['validation_indices']
    test_indices = split_info['test_indices']
    num_classes = split_info['split_stats']['num_classes']
    
    # 2. Patch predictions
    predictions = predict_patches_batch(dataloader, D, device, label_map, show_progress)
    
    # 3. Generate pixel classification map
    pixel_output = generate_pixel_classification_map(
        predictions, test_indices, centers, segmentation_data,
        train_indices, validation_indices, image_size, show_progress
    )
    
    # 4. Calculate metrics
    OA, AA, class_accuracies = calculate_metrics_vectorized(
        pixel_output, truth, num_classes, label_map, show_progress
    )
    
    return OA, AA, class_accuracies

# Alternative version with pre-computed masks for greater efficiency
def calculate_pixel_accuracy_ultra_optimized(input_dir, output_dir, filename, split_format, 
                                           dataloader, D, device, label_map=None, show_progress=False):
    """
    Ultra-optimized version that pre-computes masks and uses fully 
    vectorized operations.
    """
    # Load data
    data_info = load_dataset_and_split_info(input_dir, output_dir, filename, split_format, read_raw_data=False)
    dataset = data_info['dataset']
    split_info = data_info['split_info']
    
    # Extract information
    centers = dataset['centers']
    truth = dataset['truth']
    segmentation_data = dataset['segmentation_data']
    image_size = dataset['dimensions']['height'] * dataset['dimensions']['width']
    
    test_indices = split_info['test_indices']
    train_val_indices = np.concatenate([split_info['train_indices'], 
                                       split_info['validation_indices']])
    num_classes = split_info['split_stats']['num_classes']
    
    # Predictions
    predictions = predict_patches_batch(dataloader, D, device, label_map, show_progress)
    
    # Create base map
    pixel_output = np.zeros(image_size, dtype=np.uint8)
    pixel_output[test_indices] = predictions
    
    # Pre-compute center mask to optimize mapping
    center_to_pixel_map = centers[segmentation_data]
    non_center_mask = np.ones(image_size, dtype=bool)
    non_center_mask[centers] = False
    non_center_indices = np.where(non_center_mask)[0]
    
    # Complete vectorized mapping
    pixel_output[non_center_indices] = pixel_output[center_to_pixel_map[non_center_indices]]
    
    # Remove train/val (vectorized)
    pixel_output[train_val_indices] = 0

    # Ensure truth is a numpy array
    if not isinstance(truth, np.ndarray):
        truth = np.array(truth)
    
    # Ultra-optimized metrics calculation
    valid_mask = (pixel_output != 0) & (truth != 0)
    if not np.any(valid_mask):
        return 0.0, 0.0, [0.0] * (num_classes + 1)
        
    valid_pred = pixel_output[valid_mask]
    valid_truth = np.array([truth[i] for i in np.where(valid_mask)[0]])
    
    # Vectorized metrics
    correct_mask = (valid_pred == valid_truth)
    OA = np.mean(correct_mask)
    
    # AA using broadcasting and vectorized operations
    classes = np.arange(1, num_classes + 1)
    class_masks = valid_truth[:, None] == classes[None, :]
    class_totals = np.sum(class_masks, axis=0)
    class_corrects = np.sum(correct_mask[:, None] & class_masks, axis=0)
    
    # Avoid division by zero
    class_accuracies = np.divide(class_corrects, class_totals, 
                                out=np.zeros_like(class_corrects, dtype=float), 
                                where=class_totals!=0)
    AA = np.mean(class_accuracies)
    
    # Compatible output format
    full_class_accuracies = [0.0] + class_accuracies.tolist()
    
    return OA, AA, full_class_accuracies
