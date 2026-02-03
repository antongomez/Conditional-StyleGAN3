#!/usr/bin/env python3
"""
Script to generate LaTeX tables comparing SOTA models.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Color definitions for top results
COLOR_FIRST = "green!25"   # Verde para el primer puesto
COLOR_SECOND = "yellow!25"  # Amarillo para el segundo puesto
COLOR_THIRD = "orange!25"   # Naranja para el tercer puesto

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
    "EMViT-DDPM ViT",
    "StyleGAN3",
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a LaTeX table comparing SOTA models (OA and AA)"
    )
    parser.add_argument("-c", "--csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("-o", "--output", type=str, default=None, help="Path to output file (default: stdout)")
    parser.add_argument(
        "--show-both-methods",
        action="store_true",
        help="Show both methods (Cl. Aug and Prop. Aug) for EMViT networks. By default only Prop. Aug is shown."
    )
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
    required_cols = ["dataset", "network", "method", "oa", "aa"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV must contain the column '{col}'")

    # Replace NaN and empty strings in method column with "None" for proper grouping
    df["method"] = df["method"].fillna("None")
    df.loc[df["method"] == "", "method"] = "None"

    # Group by dataset, network and method
    grouped = df.groupby(["dataset", "network", "method"])

    return grouped


def calculate_statistics(grouped_data):
    """
    Calculate mean and standard deviation for each combination of dataset, network and method.

    Args:
        grouped_data: Data grouped by dataset, network and method

    Returns:
        DataFrame with columns: dataset, network, method, oa_mean, oa_std, aa_mean, aa_std, n_experiments
    """
    stats = []

    for (dataset_name, network_name, method_name), group in grouped_data:
        oa_values = group["oa"].values
        aa_values = group["aa"].values

        stat_entry = {
            "dataset": dataset_name,
            "network": network_name,
            "method": method_name,
            "oa_mean": np.mean(oa_values),
            "oa_std": np.std(oa_values, ddof=1) if len(oa_values) > 1 else 0.0,
            "aa_mean": np.mean(aa_values),
            "aa_std": np.std(aa_values, ddof=1) if len(aa_values) > 1 else 0.0,
            "n_experiments": len(oa_values),
        }

        stats.append(stat_entry)

    return pd.DataFrame(stats).sort_values(["dataset", "network", "method"])


def format_cell(mean, std, rank=None, percent=True):
    """
    Format a table cell with mean ± deviation.
    If rank is 1, 2, or 3, apply background color.
    
    Args:
        mean: Mean value
        std: Standard deviation
        rank: Ranking (1=first/best, 2=second, 3=third, None=no color)
        percent: If True, format as percentage
    """
    if percent:
        formatted = f"{mean*100:.1f} ± {std*100:.1f}"
    else:
        formatted = f"{mean:.3f} ± {std:.3f}"
    
    if rank == 1:
        formatted = f"\\cellcolor{{{COLOR_FIRST}}}{formatted}"
    elif rank == 2:
        formatted = f"\\cellcolor{{{COLOR_SECOND}}}{formatted}"
    elif rank == 3:
        formatted = f"\\cellcolor{{{COLOR_THIRD}}}{formatted}"
    
    return formatted


def generate_latex_table(stats_df, show_both_methods=False):
    """
    Generate LaTeX code for the table.

    Args:
        stats_df: DataFrame with statistics
        show_both_methods: If True, show both Cl. Aug and Prop. Aug for EMViT networks.
                          If False (default), show only Prop. Aug.
    """
    # Get list of unique datasets and unique networks
    datasets = sorted(stats_df["dataset"].unique())
    
    # Get unique networks from data
    available_networks = set(stats_df["network"].unique())
    
    # Use predefined order, but only include networks that exist in the data
    networks = [net for net in NETWORK_ORDER if net in available_networks]

    # Networks with multiple methods (EMViT)
    emvit_networks = ["EMViT-DDPM swinT", "EMViT-DDPM ViT", "EMViT-DDPM CNN"]
    
    # Calculate number of columns: Dataset + Metric + columns per network
    if show_both_methods:
        # EMViT networks have 2 columns (Cl. Aug and Prop. Aug), StyleGAN3 has 1
        num_data_cols = sum(2 if net in emvit_networks else 1 for net in networks)
    else:
        # All networks have 1 column (Prop. Aug for EMViT, None for StyleGAN3)
        num_data_cols = len(networks)
    col_format = "c" + "l" + "c" * num_data_cols

    # Start table
    latex = []
    latex.append("\\begin{table*}[tbp]")
    latex.append("\\centering")
    latex.append("\\resizebox{\\textwidth}{!}{%")
    latex.append(f"\\begin{{tabular}}{{{col_format}}}")
    latex.append("        \\toprule")

    # Header - First row with network names
    header1 = "        \\thead{Dataset} &    "
    for network in networks:
        network_header = "\\\\".join(network.split(" "))
        if show_both_methods and network in emvit_networks:
            # For EMViT networks with both methods, span 2 columns
            header1 += f" & \\multicolumn{{2}}{{c}}{{\\thead{{{network_header}}}}}"
        else:
            # Single column for all other cases
            header1 += f" & \\thead{{{network_header}}}"
    header1 += " \\\\"
    latex.append(header1)
    
    # Header - Second row with method names (only if showing both methods)
    if show_both_methods:
        header2 = "         & "
        for network in networks:
            if network in emvit_networks:
                # For EMViT networks, show Cl. Aug and Prop. Aug
                header2 += " & \\scriptsize{Cl. Aug} & \\scriptsize{Prop. Aug}"
            else:
                # For StyleGAN3, empty cell
                header2 += " & "
        header2 += " \\\\"
        latex.append(header2)
    latex.append("        \\midrule")

    # Networks with multiple methods (EMViT)
    emvit_networks = ["EMViT-DDPM swinT", "EMViT-DDPM ViT", "EMViT-DDPM CNN"]

    # For each dataset, generate rows for OA and AA
    for dataset_idx, dataset in enumerate(datasets):
        dataset_data = stats_df[stats_df["dataset"] == dataset]

        # Calculate top 3 values for this dataset (only for networks in NETWORK_ORDER)
        # Consider all methods for each network
        dataset_data_filtered = dataset_data[dataset_data["network"].isin(networks)]
        
        # Get all OA and AA values with their indices for ranking
        oa_values_sorted = dataset_data_filtered["oa_mean"].sort_values(ascending=False)
        aa_values_sorted = dataset_data_filtered["aa_mean"].sort_values(ascending=False)
        
        # Get top 3 values for OA and AA
        top3_oa = oa_values_sorted.head(3).values if len(oa_values_sorted) >= 3 else oa_values_sorted.values
        top3_aa = aa_values_sorted.head(3).values if len(aa_values_sorted) >= 3 else aa_values_sorted.values

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

                if network in emvit_networks:
                    if show_both_methods:
                        # EMViT networks have two methods: Cl. Aug and Prop. Aug
                        for method in ["Cl. Aug", "Prop. Aug"]:
                            method_data = network_data[network_data["method"] == method]
                            
                            if len(method_data) == 0:
                                row += " & -"
                            else:
                                row_data = method_data.iloc[0]

                                # Select the correct metric
                                if metric_idx == 0:  # OA
                                    mean_val = row_data["oa_mean"]
                                    std_val = row_data["oa_std"]
                                    # Determine rank (1, 2, 3, or None)
                                    rank = None
                                    for i, top_val in enumerate(top3_oa, 1):
                                        if np.isclose(mean_val, top_val):
                                            rank = i
                                            break
                                else:  # AA
                                    mean_val = row_data["aa_mean"]
                                    std_val = row_data["aa_std"]
                                    # Determine rank (1, 2, 3, or None)
                                    rank = None
                                    for i, top_val in enumerate(top3_aa, 1):
                                        if np.isclose(mean_val, top_val):
                                            rank = i
                                            break

                                cell = format_cell(mean_val, std_val, rank, percent=True)
                                row += f" & {cell}"
                    else:
                        # Only show Prop. Aug for EMViT
                        method_data = network_data[network_data["method"] == "Prop. Aug"]
                        
                        if len(method_data) == 0:
                            row += " & -"
                        else:
                            row_data = method_data.iloc[0]

                            # Select the correct metric
                            if metric_idx == 0:  # OA
                                mean_val = row_data["oa_mean"]
                                std_val = row_data["oa_std"]
                                # Determine rank (1, 2, 3, or None)
                                rank = None
                                for i, top_val in enumerate(top3_oa, 1):
                                    if np.isclose(mean_val, top_val):
                                        rank = i
                                        break
                            else:  # AA
                                mean_val = row_data["aa_mean"]
                                std_val = row_data["aa_std"]
                                # Determine rank (1, 2, 3, or None)
                                rank = None
                                for i, top_val in enumerate(top3_aa, 1):
                                    if np.isclose(mean_val, top_val):
                                        rank = i
                                        break

                            cell = format_cell(mean_val, std_val, rank, percent=True)
                            row += f" & {cell}"
                else:
                    # StyleGAN3 has method "None"
                    method_data = network_data[network_data["method"] == "None"]
                    
                    if len(method_data) == 0:
                        row += " & -"
                    else:
                        row_data = method_data.iloc[0]

                        # Select the correct metric
                        if metric_idx == 0:  # OA
                            mean_val = row_data["oa_mean"]
                            std_val = row_data["oa_std"]
                            # Determine rank (1, 2, 3, or None)
                            rank = None
                            for i, top_val in enumerate(top3_oa, 1):
                                if np.isclose(mean_val, top_val):
                                    rank = i
                                    break
                        else:  # AA
                            mean_val = row_data["aa_mean"]
                            std_val = row_data["aa_std"]
                            # Determine rank (1, 2, 3, or None)
                            rank = None
                            for i, top_val in enumerate(top3_aa, 1):
                                if np.isclose(mean_val, top_val):
                                    rank = i
                                    break

                        cell = format_cell(mean_val, std_val, rank, percent=True)
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

        # Collect average values for each network/method combination to determine the best
        avg_values = []
        for network in networks:
            if network in emvit_networks:
                if show_both_methods:
                    # For EMViT, collect averages for both methods
                    for method in ["Cl. Aug", "Prop. Aug"]:
                        method_data = stats_df[(stats_df["network"] == network) & (stats_df["method"] == method)]
                        if len(method_data) > 0:
                            if metric_idx == 0:  # OA
                                values = method_data["oa_mean"].values
                            else:  # AA
                                values = method_data["aa_mean"].values
                            avg_values.append(np.mean(values))
                        else:
                            avg_values.append(None)
                else:
                    # Only Prop. Aug for EMViT
                    method_data = stats_df[(stats_df["network"] == network) & (stats_df["method"] == "Prop. Aug")]
                    if len(method_data) > 0:
                        if metric_idx == 0:  # OA
                            values = method_data["oa_mean"].values
                        else:  # AA
                            values = method_data["aa_mean"].values
                        avg_values.append(np.mean(values))
                    else:
                        avg_values.append(None)
            else:
                # For StyleGAN3, method is "None"
                network_data = stats_df[(stats_df["network"] == network) & (stats_df["method"] == "None")]
                if len(network_data) > 0:
                    if metric_idx == 0:  # OA
                        values = network_data["oa_mean"].values
                    else:  # AA
                        values = network_data["aa_mean"].values
                    avg_values.append(np.mean(values))
                else:
                    avg_values.append(None)

        # Find top 3 average values
        valid_avg_values = [v for v in avg_values if v is not None]
        sorted_avg_values = sorted(valid_avg_values, reverse=True)
        top3_avg = sorted_avg_values[:3] if len(sorted_avg_values) >= 3 else sorted_avg_values

        # For each network, calculate average and std across all datasets
        avg_idx = 0
        for network in networks:
            if network in emvit_networks:
                if show_both_methods:
                    # For EMViT, show both methods
                    for method in ["Cl. Aug", "Prop. Aug"]:
                        method_data = stats_df[(stats_df["network"] == network) & (stats_df["method"] == method)]
                        
                        if len(method_data) == 0:
                            row += " & -"
                        else:
                            # Select the correct metric
                            if metric_idx == 0:  # OA
                                values = method_data["oa_mean"].values
                            else:  # AA
                                values = method_data["aa_mean"].values

                            mean_val = np.mean(values)
                            std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0

                            # Determine rank for the average row
                            rank = None
                            if avg_values[avg_idx] is not None:
                                for i, top_val in enumerate(top3_avg, 1):
                                    if np.isclose(avg_values[avg_idx], top_val):
                                        rank = i
                                        break

                            cell = format_cell(mean_val, std_val, rank, percent=True)
                            row += f" & {cell}"
                        
                        avg_idx += 1
                else:
                    # Only Prop. Aug for EMViT
                    method_data = stats_df[(stats_df["network"] == network) & (stats_df["method"] == "Prop. Aug")]
                    
                    if len(method_data) == 0:
                        row += " & -"
                    else:
                        # Select the correct metric
                        if metric_idx == 0:  # OA
                            values = method_data["oa_mean"].values
                        else:  # AA
                            values = method_data["aa_mean"].values

                        mean_val = np.mean(values)
                        std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0

                        # Determine rank for the average row
                        rank = None
                        if avg_values[avg_idx] is not None:
                            for i, top_val in enumerate(top3_avg, 1):
                                if np.isclose(avg_values[avg_idx], top_val):
                                    rank = i
                                    break

                        cell = format_cell(mean_val, std_val, rank, percent=True)
                        row += f" & {cell}"
                    
                    avg_idx += 1
            else:
                # For StyleGAN3, method is "None"
                network_data = stats_df[(stats_df["network"] == network) & (stats_df["method"] == "None")]
                
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

                    # Determine rank for the average row
                    rank = None
                    if avg_values[avg_idx] is not None:
                        for i, top_val in enumerate(top3_avg, 1):
                            if np.isclose(avg_values[avg_idx], top_val):
                                rank = i
                                break

                    cell = format_cell(mean_val, std_val, rank, percent=True)
                    row += f" & {cell}"
                
                avg_idx += 1

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
    latex_table = generate_latex_table(stats_df, show_both_methods=args.show_both_methods)

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
