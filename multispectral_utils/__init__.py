"""
Multispectral image processing utilities.

This package provides tools for reading, processing, and splitting multispectral images
for machine learning applications.
"""

from .data_readers import (
    read_raw,
    read_seg,
    read_pgm,
    read_seg_centers,
    load_multispectral_dataset
)

from .data_splitter import (
    split_dataset,
    save_split_info,
    load_split_info,
    get_split_statistics
)

from .patch_extractor import (
    select_patch,
    extract_patches,
    extract_and_save_patches,
    batch_extract_patches,
    load_patch_dataset,
    get_patch_statistics,
    is_valid_patch_center
)

from .multispectral_utils import (
    process_multispectral_dataset,
    load_processed_dataset,
    create_dataset_report,
    validate_dataset_integrity
)

__version__ = "1.0.0"
__author__ = "Antón Gómez López"

__all__ = [
    # Data readers
    'read_raw',
    'read_seg', 
    'read_pgm',
    'read_seg_centers',
    'load_multispectral_dataset',
    
    # Data splitter
    'split_dataset',
    'save_split_info',
    'load_split_info',
    'get_split_statistics',
    
    # Patch extractor
    'select_patch',
    'extract_patches',
    'extract_and_save_patches',
    'batch_extract_patches',
    'load_patch_dataset',
    'get_patch_statistics',
    'is_valid_patch_center',
    
    # Main utilities
    'process_multispectral_dataset',
    'load_processed_dataset',
    'create_dataset_report',
    'validate_dataset_integrity'
]