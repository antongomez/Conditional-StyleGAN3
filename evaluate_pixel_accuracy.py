"""
Script to evaluate a pretrained StyleGAN discriminator on multispectral images.
"""

import argparse
import time
import os
import warnings
import json
import glob
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
from visualization_utils import read_jsonl, extract_best_tick


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
    for class_idx, accuracy in class_accuracies.items():
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
        for class_idx, accuracy in class_accuracies.items():
            f.write(f"Class {class_idx:2d}: {accuracy:.4f} ({accuracy*100:6.2f}%)\n")

    print(f"Report saved to: {report_path}")


def compare_optimization_methods(input_dir, output_dir, filename, split_format, dataloader, D, device, label_map=None):
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
            label_map=label_map,
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

def output_csv_line(output_dir, output_filename, experiment_name, dataset_name, training_options, best_tick_kimg, oa_test, aa_test, oa_val, aa_val, class_accuracies):
    """
    Write a line to the results CSV file.

    Args:
        output_dir (str): Directory to save the CSV file
        output_filename (str): Name of the CSV file
        experiment_name (str): Directory name of the experiment so it can be traced back
        dataset_name (str): Name of the dataset
        training_options (dict): Dictionary of training options
        oa_test (float): Overall Accuracy on test set
        aa_test (float): Average Accuracy on test set
        oa_val (float): Overall Accuracy on validation set
        aa_val (float): Average Accuracy on validation set
        class_accuracies (list): List of class accuracies
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, output_filename)

    # Check valid training options keys
    valid_keys = {"uniform_class", "disc_on_gen", "autoencoder"}
    for key in training_options.keys():
        if key not in valid_keys:
            raise ValueError(f"Invalid training option key: {key}. Valid keys are: {valid_keys}")

    # Create header if file does not exist
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            headers = ["experiment_name", "dataset"] + list(training_options.keys()) + ["best_tick_kimg", "oa_test", "aa_test", "oa_val", "aa_val"] + [str(i) for i in range(1, 11)]  # Assuming max 10 classes
            f.write(",".join(headers) + "\n")

    # Fill class accuracies to ensure consistent number of columns
    class_accuracies_list = [None] * 10
    for i in range(1, 11):
        if i in class_accuracies:
            class_accuracies_list[i - 1] = f"{class_accuracies[i]:.6f}"

    # Write data line
    with open(csv_path, 'a') as f:
        values = [experiment_name, dataset_name] + list(training_options.values()) + [f"{float(best_tick_kimg):.3f}", f"{oa_test:.6f}", f"{aa_test:.6f}", f"{oa_val:.6f}", f"{aa_val:.6f}"] + class_accuracies_list
        f.write(",".join(map(str, values)) + "\n")

def main():
    """Main function for discriminator evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a pretrained StyleGAN discriminator on multispectral images")

    # Required arguments
    parser.add_argument("--network", dest="network_pkl", type=str, default=None,
                       help="Discriminator pickle filename")
    parser.add_argument("--experiment-dir", type=str, default=None,
                       help="Directory containing the experiment results (if network not provided)")
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
    parser.add_argument("--batch-size", type=int, default=512,
                       help="Batch size for evaluation (default: 512)")
    parser.add_argument("--remove", action="store_true",
                       help="Remove other .pkl files in the experiment directory after selecting the best one")

    # Mode selection
    parser.add_argument("--compare-times", action="store_true",
                       help="Compare execution times of different optimization methods")
    parser.add_argument("--write-report", action="store_true",
                       help="Write evaluation results to a report file")
    parser.add_argument("--output-csv", type=str, default="results.csv",
                       help="Output CSV filename for results (default: results.csv)")

    args = parser.parse_args()

    # Check if network path is provided, if not extract the best network based in validation AA
    if args.network_pkl is None and args.experiment_dir is None:
        raise ValueError("Either --network or --experiment-dir must be provided.")
    if args.network_pkl is not None and args.experiment_dir is not None:
        args.experiment_dir = None
        warnings.warn("Both --network and --experiment-dir are provided. The network will be used.", UserWarning)

    if args.experiment_dir is not None:
        jsonl_data = read_jsonl(os.path.join(args.experiment_dir, "stats.jsonl"))

        with open(os.path.join(args.output_dir, "processing_summary.json"), "r") as f:
            summary = json.load(f)
        label_map = summary.get("label_map", {})
        class_labels = [int(label) for label in label_map.keys()]
        print(f"Class labels found: {class_labels}")

        best_tick_performance = extract_best_tick(jsonl_data, class_labels, performance_key="avg", verbose=False, only_tick_with_pkl=True, network_snapshot_ticks=1)
        args.network_pkl = os.path.join(args.experiment_dir, f"network-snapshot-{int(best_tick_performance['kimg']):06d}.pkl")
        print(f"Selected network: {args.network_pkl}")
        print(f"Best tick based on validation AA: {int(best_tick_performance['tick'])} with AA: {best_tick_performance['avg_accuracy_val']:.4f} and OA: {best_tick_performance['overall_accuracy_val']:.4f}")

        if args.remove:           
            print("Removing other .pkl files...")
            deleted_count = 0
            freed_bytes = 0

            for pkl_file in glob.glob(os.path.join(args.experiment_dir, "*.pkl")):
                if os.path.abspath(pkl_file) != os.path.abspath(args.network_pkl):
                    file_size = os.path.getsize(pkl_file) 
                    os.remove(pkl_file)
                    deleted_count += 1
                    freed_bytes += file_size
            print(f"Cleanup complete. Deleted {deleted_count} files, freeing {freed_bytes / (1024 * 1024 * 1024):.2f} GB.")


    print("Loading discriminator...")
    # Load discriminator
    D, device = build_discriminator(args.network_pkl)

    print("Initializing dataset...")
    # Initialize dataset
    test_set_kwargs, _ = init_dataset_kwargs(data=args.data_zip)
    test_set_kwargs.use_label_map = True
    data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2, num_workers=0)

    # Build test dataset and dataloader
    test_dataset, test_dataloader = build_test_dataset(
        test_dataset_kwargs=test_set_kwargs,
        data_loader_kwargs=data_loader_kwargs,
        batch_size=args.batch_size,
    )

    # label map will be None if it is not defined in training
    label_map = test_dataset.get_label_map()

    if args.compare_times:
        # Compare different optimization methods
        results = compare_optimization_methods(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            filename=args.filename,
            split_format=args.split_format,
            dataloader=test_dataloader,
            D=D,
            device=device,
            label_map=test_dataset.get_label_map()
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
            dataloader=test_dataloader,
            D=D,
            device=device,
            label_map=test_dataset.get_label_map(),
            show_progress=True,

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

        # Write to CSV if requested
        if args.output_csv:
            if args.experiment_dir is None: 
                # Get the dir from the network path
                args.experiment_dir = os.path.dirname(args.network_pkl)

                # Also, get best tick performance from the jsonl file if possible
                jsonl_data = read_jsonl(os.path.join(args.experiment_dir, "stats.jsonl"))
                print(f"Extracting label map from {os.path.join(args.experiment_dir, 'processing_summary.json')}...")

                with open(os.path.join(args.output_dir, "processing_summary.json"), "r") as f:
                    summary = json.load(f)
                label_map = summary.get("label_map", {})
                class_labels = [int(label) for label in label_map.keys()]
                print(f"Class labels found: {class_labels}")

                best_tick_performance = extract_best_tick(jsonl_data, class_labels, performance_key="avg", verbose=False, only_tick_with_pkl=True, network_snapshot_ticks=1)

            with open(os.path.join(args.experiment_dir, "training_options.json"), "r") as f:
                training_options = json.load(f)
            training_options = {
                "uniform_class": training_options.get("uniform_class_labels", None),
                "disc_on_gen": training_options.get("disc_on_gen", None),
                "autoencoder": training_options.get("autoencoder", None)
            }
            output_dir = "."
            output_csv_line(
                output_dir=output_dir,
                output_filename=args.output_csv,
                experiment_name=os.path.basename(args.experiment_dir),
                dataset_name=args.filename,
                training_options=training_options,
                best_tick_kimg=best_tick_performance['kimg'],
                oa_test=OA,
                aa_test=AA,
                oa_val=best_tick_performance.get('overall_accuracy_val', None),
                aa_val=best_tick_performance.get('avg_accuracy_val', None),
                class_accuracies=class_accuracies
            )
            print(f"Results written to {os.path.join(output_dir, args.output_csv)}")

if __name__ == '__main__':
    main()