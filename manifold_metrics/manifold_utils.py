import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler


class TensorDataset(torch.utils.data.Dataset):
    """Simple dataset wrapper for tensors."""

    def __init__(self, tensor):
        self.tensor = tensor

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, idx):
        return self.tensor[idx]


def get_train_features(
    dataset,
    num_images_per_class,
    judge_model,
    batch_size=128,
    num_workers=3,
    device="cpu",
    num_gpus=None,
    rank=None,
    seed=None,
):
    """
    Selects 'num_images_per_class' number of samples PER CLASS from the dataset.
    If a class has fewer samples than requested, it samples with replacement.

    If num_gpus and rank are provided, uses DistributedSampler for distributed loading.

    Args:
        dataset: The dataset object.
        num_images_per_class: Number of samples per class.
        batch_size: DataLoader batch size.
        num_workers: DataLoader workers.
        device: Device to use (e.g., "cpu" or "cuda").
        num_gpus: Number of GPUs for distributed sampling (optional).
        rank: Rank of the current process for distributed sampling (optional).
        seed: Random seed for reproducibility (optional).

    Returns:
        numpy.ndarray: Concatenated array of all extracted features.
    """
    if seed is not None:
        np.random.seed(seed)

    raw_labels = dataset._get_raw_labels()
    rev_label_map = dataset.get_rev_label_map()  # {raw_label: internal_index}

    dataset_indices = np.arange(len(dataset))
    selected_indices = []

    for raw_label, internal_idx in rev_label_map.items():
        class_mask = raw_labels == raw_label
        available_indices = dataset_indices[class_mask]

        if len(available_indices) == 0:
            print(f"Warning: No samples found for class {raw_label} (internal {internal_idx})")
            continue

        replace = len(available_indices) < num_images_per_class
        chosen = np.random.choice(available_indices, num_images_per_class, replace=replace)
        selected_indices.extend(chosen)

    if not selected_indices:
        return torch.tensor([])

    subset = Subset(dataset, selected_indices)

    # Use a DataLoader to fetch images efficiently
    if num_gpus is not None and rank is not None:
        sampler = DistributedSampler(subset, num_replicas=num_gpus, rank=rank, shuffle=False)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            sampler=sampler,
        )
    else:
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    judge_model.eval()
    features_list = []
    with torch.no_grad():
        for images, _ in loader:
            # Normalize to [-1, 1]
            images = images.to(device, dtype=torch.float32) / 127.5 - 1.0
            _, features = judge_model(images)
            features_list.append(features.cpu())

    torch.cuda.empty_cache()

    if not features_list:
        return np.array([])

    return torch.cat(features_list, dim=0).numpy()


def get_synthetic_features(
    pool,
    classes,
    num_images_per_class,
    judge_model,
    batch_size=128,
    num_workers=3,
    device="cpu",
    seed=None,
):
    """
    Selects 'num_images_per_class' number of samples PER CLASS from the synthetic pool.
    If a class has fewer samples than requested, it samples with replacement.
    Args:
        pool: A list or dict where each entry corresponds to a class and contains a tensor of images.
        classes: List of class indices to sample from.
        num_images_per_class: Number of samples per class.
        judge_model: The model used to extract features.
        batch_size: Batch size for feature extraction.
        num_workers: Number of DataLoader workers.
        device: Device to use (e.g., "cpu" or "cuda").
        seed: Random seed for reproducibility (optional).
    Returns:
        numpy.ndarray: Concatenated array of all extracted features.
    """
    if seed is not None:
        np.random.seed(seed)

    samples_list = []

    # First, collect all samples from all classes
    for class_idx in classes:
        # pool[class_idx] is a tensor of shape [Total_N, H, W, C] (NHWC)
        class_pool = pool[class_idx]
        total_available = class_pool.shape[0]

        # Randomly select indices
        # If we request more than available, we must replace.
        replace = num_images_per_class > total_available
        selected_indices = np.random.choice(total_available, num_images_per_class, replace=replace)

        # Index directly into the tensor (fast)
        selected_batch = class_pool[selected_indices]  # [num_samples, H, W, C]

        # Permute to [num_samples, C, H, W] (NCHW) expected by the judge model
        selected_batch = selected_batch.permute(0, 3, 1, 2)

        samples_list.append(selected_batch)

    if not samples_list:
        return np.array([])

    # Concatenate all samples
    all_samples = torch.cat(samples_list, dim=0)

    # Create dataset wrapper
    dataset = TensorDataset(all_samples)

    # Each GPU should process all images to get complete features for FID calculation
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    # Extract features in batches
    judge_model.eval()
    features_list = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, features = judge_model(batch)
            features_list.append(features.cpu())

    torch.cuda.empty_cache()

    return torch.cat(features_list, dim=0).numpy()
