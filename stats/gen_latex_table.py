import argparse
import re

import pandas as pd
from stats_utils import calculate_anova, calculate_group_diffs, calculate_stats


def parse_args():
    parser = argparse.ArgumentParser(description="Generate LaTeX results from experiments CSV")
    parser.add_argument(
        "-c",
        "--csv",
        dest="csv_path",
        default="results_last.csv",
        help="Path to the experiments CSV file (default: 'results_last.csv')",
    )
    parser.add_argument(
        "-d",
        "--dataset-type",
        dest="dataset_type",
        default="test",
        choices=["test", "val"],
        help="Dataset to use for calculations (default: 'test')",
    )
    parser.add_argument(
        "--ref-config",
        dest="ref_config",
        default=None,
        help="Reference configuration for difference calculations (default: None)",
    )
    parser.add_argument(
        "-t",
        "--test",
        dest="test_type",
        default="ttest",
        choices=["dunnett", "ttest", "wilcoxon"],
        help="Type of statistical test to use (default: 'ttest')",
    )
    parser.add_argument(
        "--display-mode",
        type=str,
        nargs="+",
        default=["std"],
        choices=["std", "symbols", "pvalues", "all"],  # Valid options, several can be combined
        help="Chooses how to display the results: 'std' for mean ± std, 'symbols' for significance symbols, 'pvalues' for p-values, 'all' for all combined (default: 'std')",
    )
    parser.add_argument(
        "--pvalue-threshold",
        type=float,
        default=0.05,
        help="Threshold for p-value significance (default: 0.05)",
    )
    return parser.parse_args()


# Statistical significance symbols
NOT_SIGNIFICANT = "."
SIGNIFICANT_POSITIVE = "$+$"
SIGNIFICANT_NEGATIVE = "$-$"


def parse_pvalue(statistic, p_value, threshold=0.05):
    """Formats the p-value for LaTeX output."""
    if p_value > threshold:
        return NOT_SIGNIFICANT
    elif statistic > 0:
        return SIGNIFICANT_POSITIVE
    else:
        return SIGNIFICANT_NEGATIVE


def format_pvalue(p_value, include_equal=False):
    """Formats the p-value for LaTeX output."""
    if p_value < 0.001:
        return "$<$0.001"
    else:
        return f"{'=' if include_equal else ''}{p_value:.3f}"


def generate_anova_table(anova_results):
    """Generates a LaTeX table for ANOVA results."""
    datasets = sorted(anova_results.keys())
    variables = sorted(anova_results[datasets[0]].keys())

    print("\\begin{tabular}{l" + " ".join(["c"] * len(datasets)) + "}")
    print("        \\toprule")
    print("         & " + " & ".join([f"\\textbf{{{d.capitalize()}}}" for d in datasets]) + " \\\\")
    print("        \\midrule")

    for var in variables:
        row = [f"\\textbf{{{var.capitalize()}}}"]
        for dataset in datasets:
            # Find the p-value for the current variable and dataset
            f_statistic, p_value = anova_results[dataset][var]
            row.append(f"{format_pvalue(p_value)} {parse_pvalue(f_statistic, p_value)}")
        print(" & ".join(row) + " \\\\")

    print("        \\bottomrule")
    print("    \\end{tabular}")


def generate_stats_table(
    stats,
    display_mode,
    dataset_type="test",
    ref_config=None,
    pvalue_threshold=0.05,
):
    """Generates a LaTeX table from the computed statistics."""
    configs = sorted(
        {config for dataset_results in stats.values() for config in dataset_results.keys()},
        reverse=True,
    )
    if ref_config is not None and ref_config in configs:
        configs.remove(ref_config)

    print("\\begin{tabular}{l l " + " ".join(["c"] * len(configs)) + "}")
    print("        \\toprule")
    print("        \\textbf{Dataset} &    & " + " & ".join([f"\\textbf{{{config}}}" for config in configs]) + " \\\\")
    print("        \\midrule")

    datasets = sorted(stats.keys())
    metric_suffix = f"_{dataset_type}_diff" if ref_config is not None else f"_{dataset_type}"

    for i, dataset in enumerate(datasets):
        dataset_results = stats[dataset]
        dataset_name = dataset.capitalize()

        # Compute the best values to highlight them
        valid_oa = [dataset_results[config][f"oa{metric_suffix}"][0] for config in configs if config in dataset_results]
        valid_aa = [dataset_results[config][f"aa{metric_suffix}"][0] for config in configs if config in dataset_results]
        best_oa_val = max(valid_oa) if valid_oa else None
        best_aa_val = max(valid_aa) if valid_aa else None

        # format cells
        cells_content = {"oa": [], "aa": []}
        for config in configs:
            if config in dataset_results:
                if ref_config is not None and config == ref_config:
                    continue

                for metric in cells_content.keys():
                    # Extract mean and std
                    metric_mean, metric_std = dataset_results[config][f"{metric}{metric_suffix}"]
                    if "std" in display_mode:
                        formatted_cell_text = f"{metric_mean:.2f} $\\pm$ {metric_std:.2f}"
                    else:
                        formatted_cell_text = f"{metric_mean:.2f}"

                    # Check if is the best value to bold
                    if metric_mean == (best_oa_val if metric == "oa" else best_aa_val):
                        formatted_cell_text = f"\\textbf{{{formatted_cell_text}}}"

                    if "symbols" in display_mode:
                        statistic = dataset_results[config].get(f"{metric}_{dataset_type}_statistic")
                        p_value = dataset_results[config].get(f"{metric}_{dataset_type}_pvalue")
                        symbol = parse_pvalue(statistic, p_value, threshold=pvalue_threshold)
                        formatted_cell_text += f" {symbol}"

                    if "pvalues" in display_mode:
                        p_value = dataset_results[config].get(f"{metric}_{dataset_type}_pvalue")
                        formatted_cell_text += f" (p{format_pvalue(p_value, include_equal=True)})"

                    cells_content[metric].append(formatted_cell_text)
            else:
                cells_content["oa"].append("-")
                cells_content["aa"].append("-")

        print(f'        {dataset_name:<16} & OA & {" & ".join(cells_content["oa"])} \\\\')
        print(f'        {"":<16} & AA & {" & ".join(cells_content["aa"])} \\\\')

        if i < len(datasets) - 1:
            print("        \\midrule")

    print("        \\bottomrule")
    print("    \\end{tabular}")


def extract_seed(experiment_name):
    """Extracts the seed number from the experiment name."""
    match = re.search(r"_train_(\d+)", experiment_name)
    if match:
        return int(match.group(1))
    return None


def check_args(args):
    """Checks the validity of the provided arguments."""
    if not (0 < args.pvalue_threshold < 1):
        raise ValueError("pvalue_threshold must be between 0 and 1.")
    if args.ref_config is None and (len(args.display_mode) > 1 or "std" not in args.display_mode):
        raise ValueError("When using ref_config, display_mode must be 'std'.")


def complete_df_with_seed_and_config(df):
    """Adds seed and config columns to the DataFrame."""
    df["seed"] = df["experiment_name"].apply(extract_seed)
    df["seed"] = df["seed"].astype(int)
    df["config"] = df["uniform_class"].astype(int).astype(str) + df["disc_on_gen"].astype(int).astype(str)
    return df


def preprocess_data(csv_path):
    """Loads and preprocesses the experiment data from a CSV file."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
        return None

    df = complete_df_with_seed_and_config(df)
    df.dropna(subset=["oa_test", "aa_test"], inplace=True)
    return df


def main():
    args = parse_args()
    check_args(args)

    # Complement display_mode if 'all' is selected
    if args.display_mode == ["all"]:
        args.display_mode = ["std", "symbols", "pvalues"]

    df = preprocess_data(args.csv_path)
    if df is None:
        return

    # Calculate an ANOVA table to check for significant differences
    if args.test_type == "dunnett":
        print("\n--- ANOVA Results ---")
        anova_results = calculate_anova(df, model_vars=["config", "seed"], dependent_var=f"aa_{args.dataset_type}")
        generate_anova_table(anova_results)

    if args.ref_config is not None:
        # Calculate differences from the reference configuration
        df = (
            df.groupby(["dataset", "seed"])
            .apply(
                calculate_group_diffs, ref_config=args.ref_config, dataset_type=args.dataset_type, include_groups=False
            )
            .reset_index()
        )
        df.dropna(subset=[f"oa_{args.dataset_type}_diff", f"aa_{args.dataset_type}_diff"], inplace=True)

    # Group by dataset and config for stats calculation
    grouped = df.groupby(["dataset", "config"])

    # Calculate statistics (mean, std and significance)
    stats = calculate_stats(
        grouped, dataset_type=args.dataset_type, ref_config=args.ref_config, test_type=args.test_type
    )

    print("\n--- Mean Values with Stats ---")
    generate_stats_table(
        stats,
        display_mode=args.display_mode,
        dataset_type=args.dataset_type,
        ref_config=args.ref_config,
        pvalue_threshold=args.pvalue_threshold,
    )


if __name__ == "__main__":
    main()
