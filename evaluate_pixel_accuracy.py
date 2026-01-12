"""
Script to evaluate a pretrained StyleGAN discriminator on multispectral images.
"""

import argparse
import glob
import json
import math
import os
import re
import time
import warnings
from datetime import datetime
from pathlib import Path

import dnnlib
from multispectral_utils import (
    build_dataset,
    build_discriminator,
    calculate_pixel_accuracy,
    calculate_pixel_accuracy_optimized,
    calculate_pixel_accuracy_ultra_optimized,
    init_dataset_kwargs,
)
from visualization_utils import extract_best_tick, read_jsonl


def get_train_size(split_info_path):
    with open(split_info_path, "r") as f:
        split_info = json.load(f)
    train_size = int(split_info.get("split_stats", {}).get("train_samples"))
    return train_size


def compute_autoencoder_epochs(autoencoder_kimg, train_size):
    if autoencoder_kimg is None:
        return None
    return math.ceil(autoencoder_kimg * 1000 / train_size)


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

    with open(report_path, "w") as f:
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
        ("Ultra-Optimized", calculate_pixel_accuracy_ultra_optimized),
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
            show_progress=True,
        )
        end_time = time.time()
        execution_time = end_time - start_time

        results[method_name] = {"OA": OA, "AA": AA, "class_accuracies": class_accuracies, "time": execution_time}

        print(f"{method_name} completed in {execution_time:.4f} seconds")
        print_accuracy_results(OA, AA, class_accuracies, f"{method_name} Results")

    # Print timing comparison
    print(f"{'='*60}")
    print(f"{'TIMING COMPARISON':^60}")
    print(f"{'='*60}")

    base_time = results["Original"]["time"]
    for method_name in results:
        method_time = results[method_name]["time"]
        speedup = base_time / method_time if method_time > 0 else float("inf")
        print(f"{method_name:15}: {method_time:8.4f}s (Speedup: {speedup:6.2f}x)")

    print(f"{'='*60}\n")

    return results


def output_csv_line(
    output_dir,
    output_filename,
    experiment_name,
    dataset_name,
    training_options,
    best_tick_kimg,
    oa_test,
    aa_test,
    oa_val,
    aa_val,
    class_accuracies,
):
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
    valid_keys = {"uniform_class", "disc_on_gen", "autoencoder_epochs"}
    for key in training_options.keys():
        if key not in valid_keys:
            raise ValueError(f"Invalid training option key: {key}. Valid keys are: {valid_keys}")

    # Create header if file does not exist
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            headers = (
                ["experiment_name", "dataset"]
                + list(training_options.keys())
                + ["best_tick_kimg", "oa_test", "aa_test", "oa_val", "aa_val"]
                + [str(i) for i in range(1, 11)]
            )  # Assuming max 10 classes
            f.write(",".join(headers) + "\n")

    # Fill class accuracies to ensure consistent number of columns
    class_accuracies_list = [None] * 10
    for i in range(1, 11):
        if i in class_accuracies:
            class_accuracies_list[i - 1] = f"{class_accuracies[i]:.6f}"

    # Write data line
    with open(csv_path, "a") as f:
        values = (
            [experiment_name, dataset_name]
            + list(training_options.values())
            + [f"{float(best_tick_kimg):.3f}", f"{oa_test:.6f}", f"{aa_test:.6f}", f"{oa_val:.6f}", f"{aa_val:.6f}"]
            + class_accuracies_list
        )
        f.write(",".join(map(str, values)) + "\n")


def get_best_tick_performance(experiment_dir, output_dir, seed=None):
    """
    Extracts the best tick performance from the experiment results.

    Args:
        experiment_dir (str): Directory containing the experiment results.
        output_dir (str): Directory containing the processing summary.
    Returns:
        tuple: Best tick performance data and class labels.
    """

    jsonl_data = read_jsonl(os.path.join(experiment_dir, "stats.jsonl"))

    with open(
        os.path.join(
            output_dir, "processing_summary" + (f"_{seed}" if seed is not None and seed != 0 else "") + ".json"
        ),
        "r",
    ) as f:
        summary = json.load(f)
    label_map = summary.get("label_map", {})
    class_labels = [int(label) for label in label_map.keys()]

    best_tick_performance = extract_best_tick(
        jsonl_data,
        class_labels,
        performance_key="avg",
        verbose=False,
        only_tick_with_pkl=True,
        network_snapshot_ticks=1,
    )

    return best_tick_performance, class_labels


def select_network_snapshot(
    experiment_dir, output_dir, selection_method="best_val_aa", remove_other_snapshots=False, seed=None
):
    """
    Selects a network snapshot from an experiment directory based on the specified method.

    Args:
        experiment_dir (str): Directory containing the experiment results.
        output_dir (str): Directory containing the processing summary.
        selection_method (str): Method for selecting the snapshot ('best_val_aa' or 'last').
        remove_other_snapshots (bool): If True, removes other .pkl files.

    Returns:
        tuple: Path to the selected network snapshot and its performance data.
    """

    best_tick_performance, class_labels = get_best_tick_performance(experiment_dir, output_dir, seed=seed)
    print(f"Class labels found: {class_labels}")

    if selection_method == "best_val_aa":
        network_pkl = os.path.join(experiment_dir, f"network-snapshot-{int(best_tick_performance['kimg']):06d}.pkl")
        print(f"Selected network based on best validation AA: {network_pkl}")
        print(
            f"Best tick: {int(best_tick_performance['tick'])} with AA: {best_tick_performance['avg_accuracy_val']:.4f} and OA: {best_tick_performance['overall_accuracy_val']:.4f}"
        )
    elif selection_method == "last":
        pkl_files = sorted(glob.glob(os.path.join(experiment_dir, "network-snapshot-*.pkl")))
        if not pkl_files:
            raise ValueError(f"No network snapshots found in {experiment_dir}")
        network_pkl = pkl_files[-1]
        best_tick_performance = None  # No performance data for 'last' method
        print(f"Selected the last network snapshot: {network_pkl}")
    else:
        raise ValueError(f"Invalid selection method: {selection_method}. Choose 'best_val_aa' or 'last'.")

    if remove_other_snapshots:
        print(">>> Removing other .pkl files...")
        deleted_count = 0
        freed_bytes = 0

        all_pkl_files = glob.glob(os.path.join(experiment_dir, "*.pkl"))

        # Find the last snapshot to keep it as well
        last_snapshot = sorted(glob.glob(os.path.join(experiment_dir, "network-snapshot-*.pkl")))[-1]

        for pkl_file in all_pkl_files:
            if os.path.abspath(pkl_file) not in [os.path.abspath(network_pkl), os.path.abspath(last_snapshot)]:
                file_size = os.path.getsize(pkl_file)
                os.remove(pkl_file)
                deleted_count += 1
                freed_bytes += file_size

        print(
            f">>> Cleanup complete. Deleted {deleted_count} files, freeing {freed_bytes / (1024 * 1024 * 1024):.2f} GB."
        )

    return network_pkl, best_tick_performance


def main():
    """Main function for discriminator evaluation."""
    # fmt: off
    parser = argparse.ArgumentParser(description="Evaluate a pretrained StyleGAN discriminator on multispectral images")

    # Required arguments
    parser.add_argument("--network", dest="network_pkl", type=str, default=None, help="Discriminator pickle filename")
    parser.add_argument("--experiment-dir", default=None, help="Directory containing the experiment results (if network not provided)")
    parser.add_argument("--data-zip", type=str, required=True, help="Path to the zip file with images to evaluate")
    parser.add_argument("--input-path", type=str, default="./data", help="Path to the input multispectral dataset")
    parser.add_argument("--filename", type=str, required=True, help="Base filename (without extension)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (default: None)")

    # Optional arguments
    parser.add_argument("--split-format", type=str, default="json", choices=["json", "pickle", "npz"], help="Format for saving split information (default: json)")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for evaluation (default: 256)")
    parser.add_argument("--remove", action="store_true", help="Remove other .pkl files in the experiment directory after selecting the best one")
    parser.add_argument("--selection-method", type=str, default="last", choices=["best_val_aa", "last"], help="Method for selecting the network snapshot (default: last)")

    # Mode selection
    parser.add_argument("--write-report", action="store_true", help="Write evaluation results to a report file")                                               
    parser.add_argument("--output-csv", type=str, default="results.csv", help="Output CSV filename for results (default: results.csv)")
    # fmt: on

    args = parser.parse_args()
    input_dir = os.path.join(args.input_path, args.filename)
    output_dir = os.path.join(args.input_path, args.filename, "patches")

    # Check if network path is provided, if not extract the best network based in validation AA
    if args.network_pkl is None and args.experiment_dir is None:
        raise ValueError("Either --network or --experiment-dir must be provided.")
    if args.network_pkl is not None and args.experiment_dir is not None:
        args.experiment_dir = None
        warnings.warn("Both --network and --experiment-dir are provided. The network will be used.", UserWarning)

    best_tick_performance = None
    if args.experiment_dir is not None:
        args.network_pkl, best_tick_performance = select_network_snapshot(
            experiment_dir=args.experiment_dir,
            output_dir=output_dir,
            selection_method=args.selection_method,
            remove_other_snapshots=args.remove,
            seed=args.seed,
        )

    # Load discriminator
    D, device = build_discriminator(args.network_pkl)

    # Initialize dataset
    test_set_kwargs, _ = init_dataset_kwargs(data=args.data_zip)
    test_set_kwargs.use_label_map = True
    data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2, num_workers=3)

    # Build test dataset and dataloader
    test_dataset, test_dataloader = build_dataset(
        dataset_kwargs=test_set_kwargs,
        data_loader_kwargs=data_loader_kwargs,
        batch_size=args.batch_size,
    )

    start_time = time.time()
    oa, aa, class_accuracies = calculate_pixel_accuracy_optimized(
        input_dir=input_dir,
        output_dir=output_dir,
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
    print_accuracy_results(oa, aa, class_accuracies, "Evaluation Results")
    print(f"Execution time: {execution_time:.4f} seconds")

    # Write report if requested
    if args.write_report:
        write_report(
            output_dir=output_dir,
            filename=args.filename,
            OA=oa,
            AA=aa,
            class_accuracies=class_accuracies,
            execution_time=execution_time,
            method="optimized",
            network_path=args.network_pkl,
        )

    # Write to CSV if requested
    if args.output_csv:
        if args.experiment_dir is None or best_tick_performance is None:
            # Get the dir from the network path
            args.experiment_dir = os.path.dirname(args.network_pkl)
            best_tick_performance, _ = get_best_tick_performance(args.experiment_dir, output_dir)

        train_size = get_train_size(
            os.path.join(
                os.path.dirname(args.data_zip),
                "patches",
                "split_info" + (f"_{args.seed}" if args.seed is not None and args.seed != 0 else "") + ".json",
            )
        )
        with open(os.path.join(args.experiment_dir, "training_options.json"), "r") as f:
            training_options = json.load(f)

        training_options = {
            "uniform_class": training_options.get("uniform_class_labels", None),
            "disc_on_gen": training_options.get("disc_on_gen", None),
            "autoencoder_epochs": compute_autoencoder_epochs(
                training_options.get("autoencoder_kimg", None), train_size
            ),
        }

        output_csv_line(
            output_dir=".",
            output_filename=args.output_csv,
            experiment_name=os.path.basename(args.experiment_dir),
            dataset_name=args.filename,
            training_options=training_options,
            best_tick_kimg=best_tick_performance["kimg"],
            oa_test=oa,
            aa_test=aa,
            oa_val=best_tick_performance.get("overall_accuracy_val"),
            aa_val=best_tick_performance.get("avg_accuracy_val"),
            class_accuracies=class_accuracies,
        )
        print(f"Results written to {os.path.join(output_dir, args.output_csv)}")


if __name__ == "__main__":
    main()
