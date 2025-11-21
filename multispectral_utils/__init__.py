"""
Multispectral image processing utilities.

This package provides tools for reading, processing, and splitting multispectral images
for machine learning applications.
"""

from .data_readers import load_multispectral_dataset, read_pgm, read_raw, read_seg, read_seg_centers
from .data_splitter import get_split_statistics, load_split_info, save_split_info, split_dataset
from .multispectral_utils import (
    create_dataset_report,
    load_processed_dataset,
    process_multispectral_dataset,
    validate_dataset_integrity,
)
from .patch_extractor import (
    batch_extract_patches,
    extract_and_save_patches,
    extract_patches,
    get_patch_statistics,
    is_valid_patch_center,
    load_patch_dataset,
    select_patch,
)
from .pixel_evaluation import (
    build_dataset,
    build_discriminator,
    calculate_pixel_accuracy,
    calculate_pixel_accuracy_optimized,
    calculate_pixel_accuracy_ultra_optimized,
    init_dataset_kwargs,
)

__version__ = "1.0.0"
__author__ = "Antón Gómez López"

__all__ = [
    # Data readers
    "read_raw",
    "read_seg",
    "read_pgm",
    "read_seg_centers",
    "load_multispectral_dataset",
    # Data splitter
    "split_dataset",
    "save_split_info",
    "load_split_info",
    "get_split_statistics",
    # Patch extractor
    "select_patch",
    "extract_patches",
    "extract_and_save_patches",
    "batch_extract_patches",
    "load_patch_dataset",
    "get_patch_statistics",
    "is_valid_patch_center",
    # Main utilities
    "process_multispectral_dataset",
    "load_processed_dataset",
    "create_dataset_report",
    "validate_dataset_integrity"
    # Pixel evaluation
    "build_discriminator",
    "init_dataset_kwargs",
    "build_dataset",
    "calculate_pixel_accuracy",
    "calculate_pixel_accuracy_optimized",
    "calculate_pixel_accuracy_ultra_optimized",
]
