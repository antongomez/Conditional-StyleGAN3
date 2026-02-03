#!/usr/bin/env python3
"""
Script to generate LaTeX tables with OA and AA statistics grouped by classification_weight.
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a LaTeX table with OA and AA statistics grouped by classification_weight"
    )
    parser.add_argument("-c", "--csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument(
        "-d",
        "--dataset-type",
        type=str,
        choices=["val", "test"],
        default="test",
        help="Type of dataset to use: val (validation) or test (default: test)",
    )
    parser.add_argument(
        "-m",
        "--manifold-metrics",
        action="store_true",
        help="Use manifold metrics (FID, Precision, Recall) instead of OA and AA",
    )
    parser.add_argument(
        "-a",
        "--average-only",
        action="store_true",
        help="Generate only average results table with lambda values as rows and metrics as columns",
    )
    parser.add_argument("-o", "--output", type=str, default=None, help="Path to output file (default: stdout)")
    return parser.parse_args()


def load_and_group_data(csv_path, dataset_type="test", use_manifold=False):
    """
    Load CSV and group data by dataset and classification_weight.

    Args:
        csv_path: Path to CSV file
        dataset_type: 'test' or 'val' to choose dataset (ignored if use_manifold=True)
        use_manifold: If True, use manifold metrics (FID, Precision, Recall)

    Returns:
        tuple: (grouped_data, col1, col2, col3) or (grouped_data, oa_col, aa_col, None)
    """
    df = pd.read_csv(csv_path)

    # Verify that dataset column exists
    if "dataset" not in df.columns:
        raise ValueError("CSV must contain the 'dataset' column")

    if use_manifold:
        # Use manifold metrics
        col1 = "mean_fid"
        col2 = "mean_precision"
        col3 = "mean_recall"
        required_cols = ["classification_weight", "dataset", col1, col2, col3]
    else:
        # Use classification metrics
        col1 = f"oa_{dataset_type}"
        col2 = f"aa_{dataset_type}"
        col3 = None
        required_cols = ["classification_weight", "dataset", col1, col2]

    # Verify that required columns exist
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV must contain the column '{col}'")

    # Group by dataset and classification_weight
    grouped = df.groupby(["dataset", "classification_weight"])

    return grouped, col1, col2, col3


def calculate_statistics(grouped_data, col1, col2, col3=None):
    """
    Calculate mean and standard deviation for each combination of dataset and classification_weight.

    Args:
        grouped_data: Data grouped by dataset and classification_weight
        col1: Name of first column (oa or fid)
        col2: Name of second column (aa or precision)
        col3: Name of third column (None or recall)

    Returns:
        DataFrame with columns: dataset, classification_weight, metric1_mean, metric1_std, metric2_mean, metric2_std, [metric3_mean, metric3_std]
    """
    stats = []

    for (dataset_name, weight), group in grouped_data:
        values1 = group[col1].values
        values2 = group[col2].values

        stat_entry = {
            "dataset": dataset_name,
            "classification_weight": weight,
            "metric1_mean": np.mean(values1),
            "metric1_std": np.std(values1, ddof=1) if len(values1) > 1 else 0.0,
            "metric2_mean": np.mean(values2),
            "metric2_std": np.std(values2, ddof=1) if len(values2) > 1 else 0.0,
            "n_experiments": len(values1),
        }

        # If there is a third metric (recall for manifold)
        if col3 is not None:
            values3 = group[col3].values
            stat_entry["metric3_mean"] = np.mean(values3)
            stat_entry["metric3_std"] = np.std(values3, ddof=1) if len(values3) > 1 else 0.0

        stats.append(stat_entry)

    return pd.DataFrame(stats).sort_values(["dataset", "classification_weight"])


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


def generate_latex_table(stats_df, dataset_type="test", use_manifold=False):
    """
    Generate LaTeX code for the table.

    Args:
        stats_df: DataFrame with statistics
        dataset_type: Type of dataset used ('test' or 'val')
        use_manifold: If True, use manifold metrics (FID, Precision, Recall)
    """
    # Determine metrics
    if use_manifold:
        metric_names = ["FID", "Precision", "Recall"]
        dataset_name = "Manifold Metrics"
    else:
        metric_names = ["OA", "AA"]
        dataset_name = "Test" if dataset_type == "test" else "Validation"

    # Get list of unique datasets and unique classification_weights
    datasets = sorted(stats_df["dataset"].unique())
    weights = sorted(stats_df["classification_weight"].unique())

    # Determine number of columns (Dataset + Metric + N weights)
    col_format = "c" + "l" + "c" * len(weights)

    # Start table
    latex = []
    latex.append("\\begin{table*}[tbp]")
    latex.append("\\centering")
    latex.append(f"\\begin{{tabular}}{{{col_format}}}")
    latex.append("        \\toprule")

    # Header
    header = "        \\thead{Dataset} &    "
    for weight in weights:
        header += f" & $\\boldsymbol{{\\lambda}}_{{\\mathbf{{cls}}}} = \\mathbf{{{weight}}}$"
    header += " \\\\"
    latex.append(header)
    latex.append("        \\midrule")

    # For each dataset, generate rows for each metric
    for dataset_idx, dataset in enumerate(datasets):
        dataset_data = stats_df[stats_df["dataset"] == dataset]

        # Calculate best values for this dataset
        if use_manifold:
            # FID: lower is better
            min_metric1_idx = dataset_data["metric1_mean"].idxmin()
            # Precision and Recall: higher is better
            max_metric2_idx = dataset_data["metric2_mean"].idxmax()
            max_metric3_idx = dataset_data["metric3_mean"].idxmax()
            best_indices = [min_metric1_idx, max_metric2_idx, max_metric3_idx]
        else:
            # OA and AA: higher is better
            max_metric1_idx = dataset_data["metric1_mean"].idxmax()
            max_metric2_idx = dataset_data["metric2_mean"].idxmax()
            best_indices = [max_metric1_idx, max_metric2_idx]

        # Generate rows for each metric
        for metric_idx, metric_name in enumerate(metric_names):
            # First column: dataset name (only in first row)
            if metric_idx == 0:
                dataset_name = dataset_names_map.get(dataset, dataset)
                dataset_name_lines = dataset_name.split()
                if len(dataset_name_lines) > 1:
                    dataset_name = " ".join(dataset_name_lines[:-1]) + "\\\\" + dataset_name_lines[-1]
                else:
                    dataset_name = dataset_name.capitalize()
                row = f"\\multirow{{2}}{{*}}{{\\shortstack[c]{{{dataset_name}}}}} & {metric_name}"
            else:
                row = f" & {metric_name}"

            # Value columns for each weight
            for weight in weights:
                weight_data = dataset_data[dataset_data["classification_weight"] == weight]

                if len(weight_data) == 0:
                    # No data for this weight in this dataset
                    row += " & -"
                else:
                    row_data = weight_data.iloc[0]
                    is_best = row_data.name == best_indices[metric_idx]

                    # Select the correct metric
                    if metric_idx == 0:
                        mean_val = row_data["metric1_mean"]
                        std_val = row_data["metric1_std"]
                    elif metric_idx == 1:
                        mean_val = row_data["metric2_mean"]
                        std_val = row_data["metric2_std"]
                    else:  # metric_idx == 2
                        mean_val = row_data["metric3_mean"]
                        std_val = row_data["metric3_std"]

                    # For manifold metrics: FID is not percent, Precision and Recall are percent
                    # For classification metrics: both OA and AA are percent
                    use_percent = not use_manifold or metric_idx > 0
                    cell = format_cell(mean_val, std_val, is_best, percent=use_percent)
                    row += f" & {cell}"

            row += " \\\\"
            latex.append("        " + row)

        # Add horizontal line after each dataset (except the last one)
        if dataset_idx < len(datasets) - 1:
            latex.append("        \\midrule")

    # Add double horizontal line before average
    latex.append("        \\midrule")
    latex.append("        \\midrule")

    # Calculate average across all datasets for each weight
    for metric_idx, metric_name in enumerate(metric_names):
        if metric_idx == 0:
            row = f"\\multirow{{2}}{{*}}{{\\shortstack[c]{{\\textbf{{Average}}}}}} & {metric_name}"
        else:
            row = f" & {metric_name}"

        # For each weight, calculate average and std across all datasets
        for weight in weights:
            weight_data = stats_df[stats_df["classification_weight"] == weight]

            if len(weight_data) == 0:
                row += " & -"
            else:
                # Select the correct metric
                if metric_idx == 0:
                    values = weight_data["metric1_mean"].values
                elif metric_idx == 1:
                    values = weight_data["metric2_mean"].values
                else:  # metric_idx == 2
                    values = weight_data["metric3_mean"].values

                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0

                # Determine if this is the best value for the average row
                # Get all average values for this metric across all weights
                avg_values = []
                for w in weights:
                    wd = stats_df[stats_df["classification_weight"] == w]
                    if len(wd) > 0:
                        if metric_idx == 0:
                            avg_values.append(np.mean(wd["metric1_mean"].values))
                        elif metric_idx == 1:
                            avg_values.append(np.mean(wd["metric2_mean"].values))
                        else:
                            avg_values.append(np.mean(wd["metric3_mean"].values))

                # Determine best value
                if use_manifold and metric_idx == 0:
                    # FID: lower is better
                    is_best = mean_val == min(avg_values)
                else:
                    # OA, AA, Precision, Recall: higher is better
                    is_best = mean_val == max(avg_values)

                # For manifold metrics: FID is not percent, Precision and Recall are percent
                # For classification metrics: both OA and AA are percent
                use_percent = not use_manifold or metric_idx > 0
                cell = format_cell(mean_val, std_val, is_best, percent=use_percent)
                row += f" & {cell}"

        row += " \\\\"
        latex.append("        " + row)

    # Close table
    latex.append("        \\bottomrule")
    latex.append("    \\end{tabular}")
    if use_manifold:
        latex.append(
            f"\\caption{{Manifold metrics (FID, Precision, Recall) as a function of the classification weight $\\lambda_{{cls}}$ across different datasets. Lower FID and higher Precision and Recall values indicate better performance. Bold values represent the best result for each metric within each dataset.}}"
        )
    else:
        latex.append(
            f"\\caption{{Classification performance metrics (OA and AA) as a function of the classification weight $\\lambda_{{cls}}$ across different datasets. Higher values indicate better performance. Bold values represent the best result for each metric within each dataset.}}"
        )
    latex.append(f"\\label{{tab:cls_weight_{dataset_type}_results}}")
    latex.append("\\end{table*}")

    return "\n".join(latex)


def generate_latex_table_average_only(stats_df, dataset_type="test", use_manifold=False):
    """
    Generate LaTeX code for a simplified table with average results only.
    Rows are lambda values, columns are metrics (OA, AA or FID, Precision, Recall).

    Args:
        stats_df: DataFrame with statistics
        dataset_type: Type of dataset used ('test' or 'val')
        use_manifold: If True, use manifold metrics (FID, Precision, Recall)
    """
    # Determine metrics
    if use_manifold:
        metric_names = ["FID", "Precision", "Recall"]
        num_metrics = 3
    else:
        metric_names = ["OA", "AA"]
        num_metrics = 2

    # Get list of unique classification_weights
    weights = sorted(stats_df["classification_weight"].unique())

    # Column format: Lambda + metrics
    col_format = "c" + "c" * num_metrics

    # Start table
    latex = []
    latex.append("\\begin{table}[tbp]")
    latex.append("\\centering")
    latex.append(f"\\begin{{tabular}}{{{col_format}}}")
    latex.append("        \\toprule")

    # Header
    header = "        \\thead{$\\boldsymbol{\\lambda}_{\\mathbf{cls}}$}"
    for metric_name in metric_names:
        pct = ' (\\%)' if metric_name != 'FID' else ''
        header += f" & \\thead{{{metric_name}{pct}}}"
    header += " \\\\"
    latex.append(header)
    latex.append("        \\midrule")

    # For each lambda value, calculate average across all datasets
    all_avg_values = {}
    for metric_idx in range(num_metrics):
        all_avg_values[metric_idx] = []
        for weight in weights:
            weight_data = stats_df[stats_df["classification_weight"] == weight]
            if len(weight_data) > 0:
                if metric_idx == 0:
                    values = weight_data["metric1_mean"].values
                elif metric_idx == 1:
                    values = weight_data["metric2_mean"].values
                else:  # metric_idx == 2
                    values = weight_data["metric3_mean"].values
                all_avg_values[metric_idx].append(np.mean(values))

    # Generate rows for each lambda value
    for weight_idx, weight in enumerate(weights):
        weight_data = stats_df[stats_df["classification_weight"] == weight]

        if len(weight_data) == 0:
            row = f"        {weight}"
            for _ in range(num_metrics):
                row += " & -"
        else:
            row = f"        {weight}"

            # For each metric, calculate average and std across all datasets
            for metric_idx in range(num_metrics):
                if metric_idx == 0:
                    values = weight_data["metric1_mean"].values
                elif metric_idx == 1:
                    values = weight_data["metric2_mean"].values
                else:  # metric_idx == 2
                    values = weight_data["metric3_mean"].values

                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0

                # Determine if this is the best value for this metric
                if use_manifold and metric_idx == 0:
                    # FID: lower is better
                    is_best = mean_val == min(all_avg_values[metric_idx])
                else:
                    # OA, AA, Precision, Recall: higher is better
                    is_best = mean_val == max(all_avg_values[metric_idx])

                # For manifold metrics: FID is not percent, Precision and Recall are percent
                # For classification metrics: both OA and AA are percent
                use_percent = not use_manifold or metric_idx > 0
                cell = format_cell(mean_val, std_val, is_best, percent=use_percent)
                row += f" & {cell}"

        row += " \\\\"
        latex.append(row)

    # Close table
    latex.append("        \\bottomrule")
    latex.append("    \\end{tabular}")
    if use_manifold:
        latex.append(
            f"\\caption{{Resultados medios de todas as bacías para as métricas de variedade (FID, Precision, Recall) en función do peso de clasificación $\\lambda_{{cls}}$. Valores máis baixos de FID e valores máis altos de Precision e Recall indican mellor rendemento.}}"
        )
    else:
        latex.append(
            f"\\caption{{Resultados medios de todas as bacías para as métricas de clasificación (OA e AA) en función do peso de clasificación $\\lambda_{{cls}}$. Valores máis altos indican mellor rendemento.}}"
        )
    latex.append(f"\\label{{tab:cls_weight_{dataset_type}_average}}")
    latex.append("\\end{table}")

    return "\n".join(latex)


def main():
    """Main function."""
    args = parse_args()

    # Verify that file exists
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {args.csv}")

    # Load and process data
    if args.manifold_metrics:
        print(f"Loading data from {args.csv} (using manifold metrics: FID, Precision, Recall)...", flush=True)
    else:
        dataset_name = "test" if args.dataset_type == "test" else "validation"
        print(f"Loading data from {args.csv} (using dataset: {dataset_name})...", flush=True)

    grouped_data, col1, col2, col3 = load_and_group_data(args.csv, args.dataset_type, args.manifold_metrics)

    print("Calculating statistics...", flush=True)
    stats_df = calculate_statistics(grouped_data, col1, col2, col3)

    # Show information grouped by dataset
    print(f"\nNumber of experiments per dataset and classification_weight:")
    for dataset in sorted(stats_df["dataset"].unique()):
        print(f"\n  Dataset: {dataset}")
        dataset_data = stats_df[stats_df["dataset"] == dataset]
        for _, row in dataset_data.iterrows():
            print(f"    {row['classification_weight']}: {row['n_experiments']} experiments")

    # Generate LaTeX table
    print("\nGenerating LaTeX table...", flush=True)
    if args.average_only:
        latex_table = generate_latex_table_average_only(stats_df, args.dataset_type, args.manifold_metrics)
    else:
        latex_table = generate_latex_table(stats_df, args.dataset_type, args.manifold_metrics)

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
