"""
Script to evaluate a pretrained StyleGAN discriminator on multispectral images.
"""

import argparse
import time
import os
from datetime import datetime
from pathlib import Path

import dnnlib
from multispectral_utils import (
    build_discriminator, 
    init_dataset_kwargs, 
    build_test_dataset, 
    calculate_pixel_accuracy,
    calculate_pixel_accuracy_optimized,
    calculate_pixel_accuracy_ultra_optimized
)


def print_accuracy_results(OA, AA, class_accuracies, title="Accuracy Results"):
    """
    Print accuracy results in a formatted way.
    
    Args:
        OA (float): Overall Accuracy
        AA (float): Average Accuracy  
        class_accuracies (list): List of class accuracies
        title (str): Title for the results section
    """
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")
    print(f"Overall Accuracy (OA): {OA:.4f} ({OA*100:.2f}%)")
    print(f"Average Accuracy (AA): {AA:.4f} ({AA*100:.2f}%)")
    print(f"{'-'*50}")
    print("Class-wise Accuracies:")
    print(f"{'-'*50}")
    
    # Skip class 0 as requested
    for class_idx, accuracy in enumerate(class_accuracies[1:], start=1):
        print(f"  Class {class_idx:2d}: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print(f"{'='*50}\n")


def write_report(output_dir, filename, OA, AA, class_accuracies, execution_time, network_path, method="optimized"):
    """
    Write evaluation results to a report file.
    
    Args:
        output_dir (str): Directory to save the report
        filename (str): Base filename for the report
        OA (float): Overall Accuracy
        AA (float): Average Accuracy
        class_accuracies (list): List of class accuracies
        execution_time (float): Time taken for evaluation in seconds
        network_path (str): Path to the discriminator network
        method (str): Method used for evaluation
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build a identifier for the network using only the necessary parts of the relative path
    path = Path(network_path)
    network_id = f"{path.parent.name}_{path.stem}"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if network_path:
        report_filename = f"{filename}_evaluation_report_{network_id}.txt"
    else:
        report_filename = f"{filename}_evaluation_report_{timestamp}.txt"
    report_path = os.path.join(output_dir, report_filename)
    
    with open(report_path, 'w') as f:
        f.write(f"Discriminator Evaluation Report\n")
        f.write(f"{'='*50}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Network: {network_path if network_path else 'Unknown'}\n")
        f.write(f"Method: {method}\n")
        f.write(f"Execution Time: {execution_time:.4f} seconds\n")
        f.write(f"{'='*50}\n\n")
        
        f.write(f"ACCURACY RESULTS:\n")
        f.write(f"{'-'*30}\n")
        f.write(f"Overall Accuracy (OA): {OA:.4f} ({OA*100:.2f}%)\n")
        f.write(f"Average Accuracy (AA): {AA:.4f} ({AA*100:.2f}%)\n")
        f.write(f"\nClass-wise Accuracies:\n")
        f.write(f"{'-'*30}\n")
        
        # Skip class 0 as requested
        for class_idx, accuracy in enumerate(class_accuracies[1:], start=1):
            f.write(f"Class {class_idx:2d}: {accuracy:.4f} ({accuracy*100:6.2f}%)\n")
    
    print(f"Report saved to: {report_path}")


def compare_optimization_methods(input_dir, output_dir, filename, split_format, dataloader, D, device):
    """
    Compare execution times of different optimization methods.
    
    Args:
        input_dir (str): Input directory path
        output_dir (str): Output directory path
        filename (str): Base filename
        split_format (str): Format for split information
        dataloader: Data loader iterator
        D: Discriminator model
        device: Device to run on
        
    Returns:
        tuple: Results and times for each method
    """
    methods = [
        ("Original", calculate_pixel_accuracy),
        ("Optimized", calculate_pixel_accuracy_optimized), 
        ("Ultra-Optimized", calculate_pixel_accuracy_ultra_optimized)
    ]
    
    results = {}
    
    print(f"\n{'='*60}")
    print(f"{'PERFORMANCE COMPARISON':^60}")
    print(f"{'='*60}")
    
    for method_name, method_func in methods:
        print(f"\nRunning {method_name} method...")
        
        start_time = time.time()
        OA, AA, class_accuracies = method_func(
            input_dir=input_dir,
            output_dir=output_dir,
            filename=filename,
            split_format=split_format,
            dataloader=dataloader,
            D=D,
            device=device,
            show_progress=True
        )
        end_time = time.time()
        execution_time = end_time - start_time
        
        results[method_name] = {
            'OA': OA,
            'AA': AA,
            'class_accuracies': class_accuracies,
            'time': execution_time
        }
        
        print(f"{method_name} completed in {execution_time:.4f} seconds")
        print_accuracy_results(OA, AA, class_accuracies, f"{method_name} Results")
    
    # Print timing comparison
    print(f"{'='*60}")
    print(f"{'TIMING COMPARISON':^60}")
    print(f"{'='*60}")
    
    base_time = results["Original"]["time"]
    for method_name in results:
        method_time = results[method_name]["time"]
        speedup = base_time / method_time if method_time > 0 else float('inf')
        print(f"{method_name:15}: {method_time:8.4f}s (Speedup: {speedup:6.2f}x)")
    
    print(f"{'='*60}\n")
    
    return results


def main():
    """Main function for discriminator evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a pretrained StyleGAN discriminator on multispectral images")
    
    # Required arguments
    parser.add_argument("--network", dest="network_pkl", type=str, required=True,
                       help="Discriminator pickle filename")
    parser.add_argument("--data-zip", type=str, required=True,
                       help="Path to the zip file with images to evaluate")
    parser.add_argument("--input-dir", type=str, required=True,
                       help="Input directory containing the raw, pgm, and segment center files")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory to save the patches")
    parser.add_argument("--filename", type=str, required=True,
                       help="Base filename (without extension)")
    
    # Optional arguments
    parser.add_argument("--split-format", type=str, default="json",
                       choices=["json", "pickle", "npz"],
                       help="Format for saving split information (default: json)")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size for evaluation (default: 64)")
    
    # Mode selection
    parser.add_argument("--compare-times", action="store_true",
                       help="Compare execution times of different optimization methods")
    parser.add_argument("--write-report", action="store_true",
                       help="Write evaluation results to a report file")
    
    args = parser.parse_args()
    
    print("Loading discriminator...")
    # Load discriminator
    D, device = build_discriminator(args.network_pkl)
    
    print("Initializing dataset...")
    # Initialize dataset
    validation_set_kwargs, _ = init_dataset_kwargs(data=args.data_zip)
    data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2, num_workers=0)
    
    # Build test dataset and iterator
    _test_dataset, test_dataset_iterator = build_test_dataset(
        test_dataset_kwargs=validation_set_kwargs,
        data_loader_kwargs=data_loader_kwargs,
        batch_size=args.batch_size,
    )
    
    if args.compare_times:
        # Compare different optimization methods
        results = compare_optimization_methods(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            filename=args.filename,
            split_format=args.split_format,
            dataloader=test_dataset_iterator,
            D=D,
            device=device
        )
        
        # Use optimized results for report if needed
        if args.write_report:
            optimized_results = results["Optimized"]
            write_report(
                output_dir=args.output_dir,
                filename=args.filename,
                OA=optimized_results["OA"],
                AA=optimized_results["AA"],
                class_accuracies=optimized_results["class_accuracies"],
                execution_time=optimized_results["time"],
                method="optimized",
                network_path=args.network_pkl
            )
    
    else:
        # Run single evaluation using optimized method
        print("Computing pixel-level accuracy...")
        
        start_time = time.time()
        OA, AA, class_accuracies = calculate_pixel_accuracy_optimized(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            filename=args.filename,
            split_format=args.split_format,
            dataloader=test_dataset_iterator,
            D=D,
            device=device,
            show_progress=True
        )
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Print results
        print_accuracy_results(OA, AA, class_accuracies, "Evaluation Results")
        print(f"Execution time: {execution_time:.4f} seconds")
        
        # Write report if requested
        if args.write_report:
            write_report(
                output_dir=args.output_dir,
                filename=args.filename,
                OA=OA,
                AA=AA,
                class_accuracies=class_accuracies,
                execution_time=execution_time,
                method="optimized",
                network_path=args.network_pkl
            )


if __name__ == '__main__':
    main()