"""
Script to extract patches from multispectral datasets and split them into training, validation, and test sets.
"""

import argparse
from multispectral_utils import (
    process_multispectral_dataset,
    load_processed_dataset,
    create_dataset_report,
    validate_dataset_integrity
)


def main():
    """Main function demonstrating the usage of multispectral utilities."""
    parser = argparse.ArgumentParser(description="Process multispectral images using the utilities package")
    
    # Add arguments
    parser.add_argument("--input-dir", type=str, default="data/oitaven/", 
                       help="Input directory containing the raw, pgm, and segment center files")
    parser.add_argument("--output-dir", type=str, default="data/oitaven/patches/", 
                       help="Output directory to save the patches")
    parser.add_argument("--filename", type=str, default="oitaven", 
                       help="Base filename (without extension)")
    parser.add_argument("--rgb", action="store_true", default=False, 
                       help="Whether to save patches as RGB images")
    parser.add_argument("--train-size", type=float, default=0.15, 
                       help="Size of the train set (default: 0.15)")
    parser.add_argument("--validation_size", type=float, default=0.05, 
                       help="Size of the validation set (default: 0.05)")
    parser.add_argument("--patch-size", type=int, default=32, 
                       help="Square patch size (default: 32)")
    parser.add_argument("--batch-size", type=int, default=None, 
                       help="Batch size for processing patches (default: None)")
    parser.add_argument("--seed", type=int, default=None, 
                       help="Random seed for reproducibility (default: None)")
    parser.add_argument("--split-format", type=str, default="json", 
                       choices=["json", "pickle", "npz"],
                       help="Format for saving split information (default: json)")
    
    # Mode selection
    parser.add_argument("--mode", type=str, default="process", 
                       choices=["process", "load", "report", "validate"],
                       help="Mode of operation (default: process)")
    
    args = parser.parse_args()
    
    if args.mode == "process":
        print("=" * 60)
        print("PROCESSING MULTISPECTRAL DATASET")
        print("=" * 60)
        
        # Process the dataset
        results = process_multispectral_dataset(
            input_dir=args.input_dir,
            filename=args.filename,
            output_dir=args.output_dir,
            train_size=args.train_size,
            val_size=args.validation_size,
            patch_size=args.patch_size,
            rgb=args.rgb,
            seed=args.seed,
            batch_size=args.batch_size,
            split_format=args.split_format
        )
        
        # Generate report
        create_dataset_report(args.output_dir, args.split_format)
        
        # Validate integrity
        validation_results = validate_dataset_integrity(args.output_dir, args.split_format)
        if validation_results['valid']:
            print("✓ Dataset integrity validation passed")
        else:
            print("✗ Dataset integrity validation failed")
            for error in validation_results['errors']:
                print(f"  Error: {error}")
            for warning in validation_results['warnings']:
                print(f"  Warning: {warning}")
    
    elif args.mode == "load":
        print("=" * 60)
        print("LOADING PROCESSED DATASET")
        print("=" * 60)
        
        # Load previously processed dataset
        dataset = load_processed_dataset(args.output_dir, args.split_format)
        
        print(f"Loaded dataset from: {args.output_dir}")
        print(f"Available splits: {list(dataset['dataset_json_paths'].keys())}")
        
        # Display split statistics
        from multispectral_utils import get_split_statistics
        get_split_statistics(dataset['split_info'])
    
    elif args.mode == "report":
        print("=" * 60)
        print("GENERATING DATASET REPORT")
        print("=" * 60)
        
        # Generate comprehensive report
        report_file = create_dataset_report(args.output_dir, args.split_format)
        print(f"Report generated: {report_file}")
        
        # Display report content
        with open(report_file, 'r') as f:
            print("\n" + f.read())
    
    elif args.mode == "validate":
        print("=" * 60)
        print("VALIDATING DATASET INTEGRITY")
        print("=" * 60)
        
        # Validate dataset integrity
        validation_results = validate_dataset_integrity(args.output_dir, args.split_format)
        
        if validation_results['valid']:
            print("✓ Dataset integrity validation passed")
        else:
            print("✗ Dataset integrity validation failed")
        
        if validation_results['errors']:
            print("\nErrors:")
            for error in validation_results['errors']:
                print(f"  - {error}")
        
        if validation_results['warnings']:
            print("\nWarnings:")
            for warning in validation_results['warnings']:
                print(f"  - {warning}")


def example_workflow():
    """Example workflow showing how to use the utilities step by step."""
    print("=" * 60)
    print("EXAMPLE WORKFLOW")
    print("=" * 60)
    
    # Example parameters
    input_dir = "data/oitaven/"
    output_dir = "data/oitaven/patches/"
    filename = "oitaven"
    
    # 1. Process dataset
    print("1. Processing dataset...")
    results = process_multispectral_dataset(
        input_dir=input_dir,
        filename=filename,
        output_dir=output_dir,
        train_size=0.15,
        val_size=0.05,
        patch_size=32,
        rgb=False,
        seed=42,
        split_format='json'
    )
    
    # 2. Load processed dataset
    print("\n2. Loading processed dataset...")
    dataset = load_processed_dataset(output_dir, 'json')
    
    # 3. Generate report
    print("\n3. Generating report...")
    report_file = create_dataset_report(output_dir, 'json')
    
    # 4. Validate integrity
    print("\n4. Validating integrity...")
    validation = validate_dataset_integrity(output_dir, 'json')
    
    print(f"\nWorkflow complete! Results saved to: {output_dir}")
    return results, dataset, report_file, validation


if __name__ == "__main__":
    main()