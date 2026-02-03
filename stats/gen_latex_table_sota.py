#!/usr/bin/env python3
"""
Script to generate LaTeX tables comparing SOTA models.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

dataset_names_map = {
    "oitaven": "Oitavén River",
    "xesta": "Xesta Basin",
    "eiras": "Eiras Dam",
    "ermidas": "Ermidas Creek",
    "ferreiras": "Ferreiras River",
    "mestas": "Das Mestas River",
    "mera": "Mera River",
    "ulla": "Ulla River",
}

# Desired order of networks/models in the table columns
NETWORK_ORDER = [
    "CNN",
    "ResNet",
    "ViT",
    "ResBaGAN",
    "EffBaGAN Base",
    "EffBaGAN Small",
    "StyleGAN3",
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a LaTeX table comparing SOTA models (OA and AA)"
    )
    parser.add_argument("-c", "--csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("-o", "--output", type=str, default=None, help="Path to output file (default: stdout)")
    return parser.parse_args()


def load_and_group_data(csv_path):
    """
    Load CSV and group data by dataset and network.

    Args:
        csv_path: Path to CSV file

    Returns:
        DataFrame grouped by dataset and network
    """
    df = pd.read_csv(csv_path)

    # Verify that required columns exist
    required_cols = ["dataset", "network", "oa", "aa"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV must contain the column '{col}'")

    # Group by dataset and network
    grouped = df.groupby(["dataset", "network"])

    return grouped


def calculate_statistics(grouped_data):
    """
    Calculate mean and standard deviation for each combination of dataset and network.

    Args:
        grouped_data: Data grouped by dataset and network

    Returns:
        DataFrame with columns: dataset, network, oa_mean, oa_std, aa_mean, aa_std, n_experiments
    """
    stats = []

    for (dataset_name, network_name), group in grouped_data:
        oa_values = group["oa"].values
        aa_values = group["aa"].values

        stat_entry = {
            "dataset": dataset_name,
            "network": network_name,
            "oa_mean": np.mean(oa_values),
            "oa_std": np.std(oa_values, ddof=1) if len(oa_values) > 1 else 0.0,
            "aa_mean": np.mean(aa_values),
            "aa_std": np.std(aa_values, ddof=1) if len(aa_values) > 1 else 0.0,
            "n_experiments": len(oa_values),
        }

        stats.append(stat_entry)

    return pd.DataFrame(stats).sort_values(["dataset", "network"])


def format_cell(mean, std, is_best, percent=True):
    """
    Format a table cell with mean ± deviation.
    If is_best is True, mark in bold.
    """
    if percent:
        formatted = f"{mean*100:.1f} ± {std*100:.1f}"
    else:
        formatted = f"{mean:.3f} ± {std:.3f}"
    if is_best:
        formatted = f"\\textbf{{{formatted}}}"
    return formatted


def generate_latex_table(stats_df):
    """
    Generate LaTeX code for the table.

    Args:
        stats_df: DataFrame with statistics
    """
    # Get list of unique datasets and unique networks
    datasets = sorted(stats_df["dataset"].unique())
    
    # Get unique networks from data
    available_networks = set(stats_df["network"].unique())
    
    # Use predefined order, but only include networks that exist in the data
    networks = [net for net in NETWORK_ORDER if net in available_networks]

    # Determine number of columns (Dataset + Metric + N networks)
    col_format = "c" + "l" + "c" * len(networks)

    # Start table
    latex = []
    latex.append("\\begin{table*}[tbp]")
    latex.append("\\centering")
    latex.append("\\resizebox{\\textwidth}{!}{")
    latex.append(f"\\begin{{tabular}}{{{col_format}}}")
    latex.append("        \\toprule")

    # Header
    header = "        \\thead{Dataset} &    "
    for network in networks:
        header += f" & \\thead{{{network}}}"
    header += " \\\\"
    latex.append(header)
    latex.append("        \\midrule")

    # For each dataset, generate rows for OA and AA
    for dataset_idx, dataset in enumerate(datasets):
        dataset_data = stats_df[stats_df["dataset"] == dataset]

        # Calculate best values for this dataset (only for networks in NETWORK_ORDER)
        # For each network, get OA and AA
        dataset_data_filtered = dataset_data[dataset_data["network"].isin(networks)]
        max_oa_mean = dataset_data_filtered["oa_mean"].max()
        max_aa_mean = dataset_data_filtered["aa_mean"].max()

        # Generate rows for OA and AA
        for metric_idx, metric_name in enumerate(["OA", "AA"]):
            # First column: dataset name (only in first row)
            if metric_idx == 0:
                dataset_name = dataset_names_map.get(dataset, dataset)
                dataset_name_lines = dataset_name.split()
                if len(dataset_name_lines) > 1:
                    dataset_name = " ".join(dataset_name_lines[:-1]) + "\\\\" + dataset_name_lines[-1]
                else:
                    dataset_name = dataset_name.capitalize()
                row = f"\\multirow{{2}}{{*}}{{\\shortstack[c]{{{dataset_name}}}}} & {metric_name} \\%"
            else:
                row = f" & {metric_name} \\%"

            # Value columns for each network
            for network in networks:
                network_data = dataset_data[dataset_data["network"] == network]

                if len(network_data) == 0:
                    # No data for this network in this dataset
                    row += " & -"
                else:
                    row_data = network_data.iloc[0]

                    # Select the correct metric
                    if metric_idx == 0:  # OA
                        mean_val = row_data["oa_mean"]
                        std_val = row_data["oa_std"]
                        is_best = mean_val == max_oa_mean
                    else:  # AA
                        mean_val = row_data["aa_mean"]
                        std_val = row_data["aa_std"]
                        is_best = mean_val == max_aa_mean

                    cell = format_cell(mean_val, std_val, is_best, percent=True)
                    row += f" & {cell}"

            row += " \\\\"
            latex.append("        " + row)

        # Add horizontal line after each dataset (except the last one)
        if dataset_idx < len(datasets) - 1:
            latex.append("        \\midrule")

    # Add double horizontal line before average
    latex.append("        \\midrule")
    latex.append("        \\midrule")

    # Calculate average across all datasets for each network
    for metric_idx, metric_name in enumerate(["OA", "AA"]):
        if metric_idx == 0:
            row = f"\\multirow{{2}}{{*}}{{\\shortstack[c]{{\\textbf{{Average}}}}}} & {metric_name} \\%"
        else:
            row = f" & {metric_name} \\%"

        # Collect average values for each network to determine the best
        avg_values = []
        for network in networks:
            network_data = stats_df[stats_df["network"] == network]
            if len(network_data) > 0:
                if metric_idx == 0:  # OA
                    values = network_data["oa_mean"].values
                else:  # AA
                    values = network_data["aa_mean"].values
                avg_values.append(np.mean(values))
            else:
                avg_values.append(None)

        # Find best average value
        valid_avg_values = [v for v in avg_values if v is not None]
        best_avg = max(valid_avg_values) if valid_avg_values else None

        # For each network, calculate average and std across all datasets
        for network_idx, network in enumerate(networks):
            network_data = stats_df[stats_df["network"] == network]

            if len(network_data) == 0:
                row += " & -"
            else:
                # Select the correct metric
                if metric_idx == 0:  # OA
                    values = network_data["oa_mean"].values
                else:  # AA
                    values = network_data["aa_mean"].values

                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0

                # Determine if this is the best value for the average row
                is_best = (avg_values[network_idx] == best_avg) if best_avg is not None else False

                cell = format_cell(mean_val, std_val, is_best, percent=True)
                row += f" & {cell}"

        row += " \\\\"
        latex.append("        " + row)

    # Close table
    latex.append("        \\bottomrule")
    latex.append("    \\end{tabular}}")
    latex.append(
        "\\caption{Comparison of the proposed method (StyleGAN3) against state-of-the-art architectures. Benchmarks include standard discriminative models (CNN \\cite{choi2017pytorch}, ResNet \\cite{he2016}, ViT \\cite{dosovitskiy2020}), GAN-based frameworks (ResBaGAN \\cite{dieste2023}, EffBaGAN \\cite{vilela2025}), and the diffusion-based EMViT-DDPM (ViT) with EBDA ($p=2$) \\cite{barreiro2025}. OA and AA denote Overall Accuracy and Average Accuracy, respectively (mean $\\pm$ standard deviation). \\colorbox{green!25}{Green}, \\colorbox{yellow!25}{yellow}, and \\colorbox{orange!25}{orange} backgrounds highlight the first, second, and third best results for each metric and dataset.}"
    )
    latex.append("\\label{tab:sota_comparison}")
    latex.append("\\end{table*}")

    return "\n".join(latex)


def main():
    """Main function."""
    args = parse_args()

    # Verify that file exists
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {args.csv}")

    # Load and process data
    print(f"Loading data from {args.csv}...", flush=True)
    grouped_data = load_and_group_data(args.csv)

    print("Calculating statistics...", flush=True)
    stats_df = calculate_statistics(grouped_data)

    # Show information grouped by dataset
    print(f"\nNumber of experiments per dataset and network:")
    for dataset in sorted(stats_df["dataset"].unique()):
        print(f"\n  Dataset: {dataset}")
        dataset_data = stats_df[stats_df["dataset"] == dataset]
        for _, row in dataset_data.iterrows():
            print(f"    {row['network']}: {row['n_experiments']} experiments")

    # Generate LaTeX table
    print("\nGenerating LaTeX table...", flush=True)
    latex_table = generate_latex_table(stats_df)

    # Save or print
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(latex_table)
        print(f"\nTable saved to: {args.output}")
    else:
        print("\n" + "=" * 70)
        print("LaTeX TABLE:")
        print("=" * 70)
        print(latex_table)
        print("=" * 70)


if __name__ == "__main__":
    main()
