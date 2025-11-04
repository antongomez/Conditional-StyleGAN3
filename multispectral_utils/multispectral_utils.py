"""
Main utility module for multispectral image processing.
Provides high-level functions that combine functionality from other modules.
"""

import json
import os
from datetime import datetime

from .data_readers import load_multispectral_dataset
from .data_splitter import load_split_info, save_split_info, split_dataset
from .patch_extractor import batch_extract_patches, get_patch_statistics


def process_multispectral_dataset(input_dir, filename, output_dir, 
                                 train_size=0.15, val_size=0.05, 
                                 patch_size=32, rgb=False, seed=None, 
                                 batch_size=None, split_format='json'):
    """
    Complete processing pipeline for multispectral datasets.
    
    Args:
        input_dir: Directory containing input files
        filename: Base filename without extension
        output_dir: Output directory for processed data
        train_size: Training set proportion (0-1)
        val_size: Validation set proportion (0-1)
        patch_size: Square patch size
        rgb: Whether to save patches as RGB images
        seed: Random seed for reproducibility
        batch_size: Optional batch size for alignment
        split_format: Format for saving split info ('json', 'pickle', 'npz')
        
    Returns:
        dict: Dictionary with processing results and file paths
    """
    print(f"* Processing multispectral dataset: {filename}")
    
    # Load dataset
    dataset = load_multispectral_dataset(input_dir, filename)
    
    # Split dataset
    train_centers, val_centers, test_centers, label_map = split_dataset(
        dataset['truth'], 
        dataset['centers'],
        dataset['dimensions']['height'],
        dataset['dimensions']['width'],
        patch_size,
        patch_size,
        train_size,
        val_size,
        seed=seed,
        batch_size=batch_size
    )
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save split information
    split_metadata = {
        'dataset_name': filename,
        'patch_size': patch_size,
        'train_size': train_size,
        'val_size': val_size,
        'seed': seed,
        'batch_size': batch_size,
        'rgb': rgb,
        'processing_date': datetime.now().isoformat(),
        'dimensions': dataset['dimensions'],
        'input_dir': input_dir,
        'output_dir': output_dir
    }
    
    split_filename = "split_info" + (f"_{seed}" if seed is not None and seed != 0 else "") + f".{split_format}"
    split_file = os.path.join(output_dir, split_filename)
    save_split_info(train_centers, val_centers, test_centers, label_map, 
                   split_file, split_metadata, split_format)
    
    # Extract and save patches
    json_paths = batch_extract_patches(
        dataset['data'],
        dataset['truth'],
        train_centers,
        val_centers,
        test_centers,
        dataset['dimensions']['height'],
        dataset['dimensions']['width'],
        patch_size,
        output_dir,
        rgb
    )
    
    # Print label mapping
    print("* Label map:")
    for class_idx, label in label_map.items():
        print(f"  Class {class_idx:2d} => Label {label:2d}")

    dataset_info_summary = {
        'num_pixels': dataset['data'].shape[1] * dataset['data'].shape[2],
        'num_channels': dataset['data'].shape[0],
        'num_labeled_pixels': sum(1 for t in dataset['truth'] if t > 0),
        'num_centers': len(dataset['centers']),
        'dimensions': {
            'height': dataset['dimensions']['height'],
            'width': dataset['dimensions']['width'],
            'channels': dataset['dimensions']['channels']
        },
        'metadata': dataset['metadata'].copy()  
    }
    
    results = {
        'split_file': split_file,
        'dataset_json_paths': json_paths,
        'label_map': label_map,
        'split_metadata': split_metadata,
        'dataset_info': dataset_info_summary
    }
    
    # Save processing summary
    summary_file = os.path.join(output_dir, "processing_summary" + (f"_{seed}" if seed is not None and seed != 0 else "") + ".json")
    with open(summary_file, 'w') as f:
        # Convert numpy types for JSON serialization
        json_results = json.loads(json.dumps(results, default=str))
        json.dump(json_results, f, indent=4)
    
    print(f"* Processing complete. Results saved to: {output_dir}")
    return results


def load_processed_dataset(output_dir, seed=None, split_format='json'):
    """
    Load a previously processed multispectral dataset.
    
    Args:
        output_dir: Directory containing processed data
        seed: Random seed used during processing (if any)
        split_format: Format of split info file ('json', 'pickle', 'npz')
        
    Returns:
        dict: Dictionary with loaded dataset information
    """
    # Load split information
    split_filename = "split_info" + (f"_{seed}" if seed is not None and seed != 0 else "") + f".{split_format}"
    split_file = os.path.join(output_dir, split_filename)
    split_info = load_split_info(split_file, split_format)
    
    # Load dataset JSON files
    json_paths = {}
    for split_name in ['train', 'validation', 'test']:
        json_path = os.path.join(output_dir, split_name, 'dataset.json')
        if os.path.exists(json_path):
            json_paths[split_name] = json_path
    
    # Load processing summary if available
    summary_file = os.path.join(output_dir, "processing_summary" + (f"_{seed}" if seed is not None and seed != 0 else "") + ".json")
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            summary = json.load(f)
    
    return {
        'split_info': split_info,
        'dataset_json_paths': json_paths,
        'processing_summary': summary
    }

def create_split_distribution_table(dataset_json_paths):
    """
    Create a detailed table showing class distribution across train/validation splits.
    
    Args:
        dataset_json_paths: Dictionary with paths to dataset.json files for each split
        
    Returns:
        list: Lines of the formatted table
    """
    # Get statistics for each split
    train_stats = get_patch_statistics(dataset_json_paths['train']) if 'train' in dataset_json_paths else None
    val_stats = get_patch_statistics(dataset_json_paths['validation']) if 'validation' in dataset_json_paths else None
    test_stats = get_patch_statistics(dataset_json_paths['test']) if 'test' in dataset_json_paths else None
    
    if not test_stats:
        return ["Error: Test set not found. Cannot calculate percentages."]
    
    # Use test set as total (since it contains the entire dataset)
    total_per_class = test_stats['class_counts']
    
    # Calculate totals
    total_train = train_stats['total_patches'] if train_stats else 0
    total_val = val_stats['total_patches'] if val_stats else 0
    total_test = test_stats['total_patches']
    
    # Calculate percentages
    train_percent = (total_train / total_test) * 100 if total_test > 0 else 0
    val_percent = (total_val / total_test) * 100 if total_test > 0 else 0
    
    table_lines = []
    
    # Header with totals
    table_lines.append(f"  Total train samples: {total_train} ({train_percent:.2f}%)")
    table_lines.append(f"  Total validation samples: {total_val} ({val_percent:.2f}%) (batch_aligned)")
    
    # Table header
    table_lines.append("  Class    : seg.tot | train | validation | train % |   val %")
    
    # Get all unique classes and sort them
    all_classes = sorted(total_per_class.keys())
    
    # Generate table rows
    for class_label in all_classes:
        total_class = total_per_class[class_label]
        train_class = train_stats['class_counts'].get(class_label, 0) if train_stats else 0
        val_class = val_stats['class_counts'].get(class_label, 0) if val_stats else 0
        
        # Calculate percentages for this class
        train_class_percent = (train_class / total_class) * 100 if total_class > 0 else 0
        val_class_percent = (val_class / total_class) * 100 if total_class > 0 else 0
        
        # Format the row
        row = f"  Class {class_label:2d} : {total_class:7d} | {train_class:5d} | {val_class:10d} | {train_class_percent:6.2f}% | {val_class_percent:6.2f}%"
        table_lines.append(row)
    
    return table_lines


def create_dataset_report(output_dir, seed=None, split_format='json'):
    """
    Create a comprehensive report of the processed dataset.
    
    Args:
        output_dir: Directory containing processed data
        seed: Random seed used during processing (if any)
        split_format: Format of split info file
        
    Returns:
        str: Path to the generated report
    """
    from .data_splitter import get_split_statistics
    from .patch_extractor import get_patch_statistics

    # Load processed dataset
    dataset = load_processed_dataset(output_dir, seed, split_format)
    
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("MULTISPECTRAL DATASET PROCESSING REPORT")
    report_lines.append("=" * 60)
    
    # Basic information
    if dataset['processing_summary'] and 'split_metadata' in dataset['processing_summary']:
        metadata = dataset['processing_summary']['split_metadata']
        report_lines.append(f"Dataset Name: {metadata.get('dataset_name', 'Unknown')}")
        report_lines.append(f"Processing Date: {metadata.get('processing_date', 'Unknown')}")
        report_lines.append(f"Patch Size: {metadata.get('patch_size', 'Unknown')}")
        report_lines.append(f"RGB Mode: {metadata.get('rgb', 'Unknown')}")
        report_lines.append(f"Random Seed: {metadata.get('seed', 'None')}")
        report_lines.append("")
    
    # Split statistics
    report_lines.append("DATASET SPLIT STATISTICS")
    report_lines.append("-" * 30)
    split_stats = get_split_statistics(dataset['split_info'])
    report_lines.append(f"  Training samples: {split_stats.get('train_samples', 0)}")
    report_lines.append(f"  Validation samples: {split_stats.get('validation_samples', 0)}")
    report_lines.append(f"  Test samples: {split_stats.get('test_samples', 0)}")
    report_lines.append(f"  Total samples: {split_stats.get('total_samples', 0)}")
    report_lines.append(f"  Number of classes: {split_stats.get('num_classes', 0)}")
    report_lines.append("")

    # Add the detailed distribution table
    report_lines.append("DETAILED CLASS DISTRIBUTION")
    report_lines.append("-" * 30)
    distribution_table = create_split_distribution_table(dataset['dataset_json_paths'])
    report_lines.extend(distribution_table)
    report_lines.append("")
    
    # Patch statistics for each split
    for split_name, json_path in dataset['dataset_json_paths'].items():
        if os.path.exists(json_path):
            report_lines.append(f"{split_name.upper()} SET STATISTICS")
            report_lines.append("-" * 30)
            
            stats = get_patch_statistics(json_path)
            report_lines.append(f"Total patches: {stats['total_patches']}")
            report_lines.append(f"Number of classes: {stats['num_classes']}")
            report_lines.append("Class distribution:")
            
            for class_label, count in sorted(stats['class_counts'].items()):
                percentage = stats['class_distribution'][class_label] * 100
                report_lines.append(f"  Class {class_label}: {count} patches ({percentage:.2f}%)")
            
            report_lines.append("")
    
    # Save report
    report_file = os.path.join(output_dir, f"dataset_report{'_' + str(seed) if seed else ''}.txt")
    with open(report_file, 'w') as f:
        f.write("\n".join(report_lines))
    
    print(f"* Dataset report saved to: {report_file}")
    return report_file


def validate_dataset_integrity(output_dir, seed=None, split_format='json'):
    """
    Validate the integrity of a processed dataset.
    
    Args:
        output_dir: Directory containing processed data
        seed: Random seed used during processing (if any)
        split_format: Format of split info file
        
    Returns:
        dict: Validation results
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check if split info file exists
    split_filename = "split_info" + (f"_{seed}" if seed is not None and seed != 0 else "") + f".{split_format}"
    split_file = os.path.join(output_dir, split_filename)
    if not os.path.exists(split_file):
        results['valid'] = False
        results['errors'].append(f"Split info file not found: {split_file}")
        return results
    
    # Load split info
    try:
        split_info = load_split_info(split_file, split_format)
    except Exception as e:
        results['valid'] = False
        results['errors'].append(f"Error loading split info: {str(e)}")
        return results
    
    # Check dataset JSON files
    for split_name in ['train', 'validation', 'test']:
        json_path = os.path.join(output_dir, split_name, 'dataset.json')
        if not os.path.exists(json_path):
            results['warnings'].append(f"Dataset JSON not found for {split_name} split")
            continue
        
        # Check if patch files exist
        try:
            with open(json_path, 'r') as f:
                dataset = json.load(f)
            
            missing_patches = []
            for label_info in dataset['labels']:
                patch_path = os.path.join(output_dir, split_name, label_info[0])
                if not os.path.exists(patch_path):
                    missing_patches.append(label_info[0])
            
            if missing_patches:
                results['warnings'].append(f"Missing {len(missing_patches)} patch files in {split_name} split")
                
        except Exception as e:
            results['warnings'].append(f"Error validating {split_name} split: {str(e)}")
    
    # Check for overlapping indices
    train_set = set(split_info.get('train_indices', []))
    val_set = set(split_info.get('validation_indices', []))
    
    if train_set & val_set:
        results['errors'].append("Overlapping indices between train and validation sets")
        results['valid'] = False

    
    return results