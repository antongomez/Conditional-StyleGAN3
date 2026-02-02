import argparse
import re

import pandas as pd
from stats_utils import calculate_anova, calculate_group_diffs, calculate_stats

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

config_to_header_map = {
    "00": ("Original Dist.", "Real Only"),
    "01": ("Original Dist.", "Real + Syn."),
    "10": ("Balanced Dist.", "Real Only"),
    "11": ("Balanced Dist.", "Real + Syn."),
}


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
    parser.add_argument(
        "-m",
        "--manifold",
        dest="manifold",
        action="store_true",
        help="Use manifold metrics (FID, Precision, Recall) instead of OA and AA",
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


def calculate_manifold_stats(grouped):
    """Computes mean and std for manifold metrics (FID, Precision, Recall) for each dataset and configuration."""
    from collections import defaultdict

    import numpy as np

    stats = defaultdict(lambda: defaultdict(dict))

    for (dataset, config), group in grouped:
        # Calculate statistics for each manifold metric
        fid_mean = group["mean_fid"].mean()
        fid_std = group["mean_fid"].std(ddof=1) if len(group) > 1 else 0.0

        precision_mean = group["mean_precision"].mean()
        precision_std = group["mean_precision"].std(ddof=1) if len(group) > 1 else 0.0

        recall_mean = group["mean_recall"].mean()
        recall_std = group["mean_recall"].std(ddof=1) if len(group) > 1 else 0.0

        stats[dataset][config]["fid"] = (fid_mean, fid_std)
        stats[dataset][config]["precision"] = (precision_mean, precision_std)
        stats[dataset][config]["recall"] = (recall_mean, recall_std)

    # Calculate global averages across all datasets for each configuration
    global_stats = defaultdict(dict)
    all_configs = set()
    for dataset_results in stats.values():
        all_configs.update(dataset_results.keys())

    for config in all_configs:
        fid_values = []
        precision_values = []
        recall_values = []

        for dataset in stats.keys():
            if config in stats[dataset]:
                fid_mean, _ = stats[dataset][config]["fid"]
                precision_mean, _ = stats[dataset][config]["precision"]
                recall_mean, _ = stats[dataset][config]["recall"]

                fid_values.append(fid_mean)
                precision_values.append(precision_mean)
                recall_values.append(recall_mean)

        if fid_values:
            global_stats[config]["fid"] = (
                np.mean(fid_values),
                np.std(fid_values, ddof=1) if len(fid_values) > 1 else 0.0,
            )
        if precision_values:
            global_stats[config]["precision"] = (
                np.mean(precision_values),
                np.std(precision_values, ddof=1) if len(precision_values) > 1 else 0.0,
            )
        if recall_values:
            global_stats[config]["recall"] = (
                np.mean(recall_values),
                np.std(recall_values, ddof=1) if len(recall_values) > 1 else 0.0,
            )

    stats["__global__"] = global_stats

    return stats


def generate_manifold_stats_table(stats, display_mode):
    """Generates a LaTeX table for manifold metrics (FID, Precision, Recall)."""
    configs = sorted(
        {config for dataset_results in stats.values() for config in dataset_results.keys()},
        reverse=True,
    )

    print("\\begin{tabular}{c l " + " ".join(["c"] * len(configs)) + "}")
    print("        \\toprule")
    print(
        "        \\textbf{Dataset} &    & "
        + " & ".join(
            [f"\\textbf{{\\shortstack[c]{{{'\\\\'.join(config_to_header_map.get(config))}}}}}" for config in configs]
        )
        + " \\\\"
    )
    print("        \\midrule")

    datasets = sorted([d for d in stats.keys() if d != "__global__"])

    for i, dataset in enumerate(datasets):
        dataset_results = stats[dataset]
        dataset_name = dataset_names_map.get(dataset, dataset)
        dataset_name_lines = dataset_name.split()
        if len(dataset_name_lines) > 1:
            dataset_name = " ".join(dataset_name_lines[:-1]) + "\\\\" + dataset_name_lines[-1]
        else:
            dataset_name = dataset_name.capitalize()

        # Compute the best values to highlight them
        # FID: lower is better, Precision and Recall: higher is better
        valid_fid = [dataset_results[config]["fid"][0] for config in configs if config in dataset_results]
        valid_precision = [dataset_results[config]["precision"][0] for config in configs if config in dataset_results]
        valid_recall = [dataset_results[config]["recall"][0] for config in configs if config in dataset_results]
        best_fid_val = min(valid_fid) if valid_fid else None
        best_precision_val = max(valid_precision) if valid_precision else None
        best_recall_val = max(valid_recall) if valid_recall else None

        # Format cells for FID, Precision, Recall
        cells_content = {"fid": [], "precision": [], "recall": []}
        for config in configs:
            if config in dataset_results:
                for metric in cells_content.keys():
                    metric_mean, metric_std = dataset_results[config][metric]
                    if "std" in display_mode:
                        formatted_cell_text = f"{metric_mean:.1f} $\\pm$ {metric_std:.1f}"
                    else:
                        formatted_cell_text = f"{metric_mean:.1f}"

                    # Check if is the best value to bold
                    if metric == "fid":
                        is_best = metric_mean == best_fid_val
                    elif metric == "precision":
                        is_best = metric_mean == best_precision_val
                    else:  # recall
                        is_best = metric_mean == best_recall_val

                    if is_best:
                        formatted_cell_text = f"\\textbf{{{formatted_cell_text}}}"

                    cells_content[metric].append(formatted_cell_text)
            else:
                cells_content["fid"].append("-")
                cells_content["precision"].append("-")
                cells_content["recall"].append("-")

        print(
            f'        \\multirow{{3}}{{*}}{{\\shortstack[c]{{{dataset_name}}}}} & FID & {" & ".join(cells_content["fid"])} \\\\'
        )
        print(f'         & Precision & {" & ".join(cells_content["precision"])} \\\\')
        print(f'         & Recall & {" & ".join(cells_content["recall"])} \\\\')

        if i < len(datasets) - 1:
            print("        \\midrule")

    # Add global average row
    if "__global__" in stats:
        print("        \\midrule")
        print("        \\midrule")

        global_results = stats["__global__"]

        # Calculate best values for global averages
        valid_fid = [global_results[config]["fid"][0] for config in configs if config in global_results]
        valid_precision = [global_results[config]["precision"][0] for config in configs if config in global_results]
        valid_recall = [global_results[config]["recall"][0] for config in configs if config in global_results]
        best_fid_val = min(valid_fid) if valid_fid else None
        best_precision_val = max(valid_precision) if valid_precision else None
        best_recall_val = max(valid_recall) if valid_recall else None

        # Format cells for global averages
        cells_content = {"fid": [], "precision": [], "recall": []}
        for config in configs:
            if config in global_results:
                for metric in cells_content.keys():
                    metric_mean, metric_std = global_results[config][metric]
                    if "std" in display_mode:
                        formatted_cell_text = f"{metric_mean:.1f} $\\pm$ {metric_std:.1f}"
                    else:
                        formatted_cell_text = f"{metric_mean:.1f}"

                    # Check if is the best value to bold
                    if metric == "fid":
                        is_best = metric_mean == best_fid_val
                    elif metric == "precision":
                        is_best = metric_mean == best_precision_val
                    else:  # recall
                        is_best = metric_mean == best_recall_val

                    if is_best:
                        formatted_cell_text = f"\\textbf{{{formatted_cell_text}}}"

                    cells_content[metric].append(formatted_cell_text)
            else:
                cells_content["fid"].append("-")
                cells_content["precision"].append("-")
                cells_content["recall"].append("-")

        print(
            f'        \\multirow{{3}}{{*}}{{\\shortstack[c]{{\\textbf{{Average}}}}}} & FID & {" & ".join(cells_content["fid"])} \\\\'
        )
        print(f'         & Precision & {" & ".join(cells_content["precision"])} \\\\')
        print(f'         & Recall & {" & ".join(cells_content["recall"])} \\\\')

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

    print("\\begin{tabular}{c l " + " ".join(["c"] * len(configs)) + "}")
    print("        \\toprule")
    print(
        "        \\textbf{Dataset} &    & "
        + " & ".join([f"\\thead{{{'\\\\'.join(config_to_header_map.get(config))}}}" for config in configs])
        + " \\\\"
    )
    print("        \\midrule")

    datasets = sorted([d for d in stats.keys() if d != "__global__"])
    metric_suffix = f"_{dataset_type}_diff" if ref_config is not None else f"_{dataset_type}"

    for i, dataset in enumerate(datasets):
        dataset_results = stats[dataset]
        dataset_name = dataset_names_map.get(dataset, dataset)
        dataset_name_lines = dataset_name.split()
        if len(dataset_name_lines) > 1:
            dataset_name = " ".join(dataset_name_lines[:-1]) + "\\\\" + dataset_name_lines[-1]
        else:
            dataset_name = dataset_name.capitalize()

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
                        formatted_cell_text = f"{metric_mean:.1f} $\\pm$ {metric_std:.1f}"
                    else:
                        formatted_cell_text = f"{metric_mean:.1f}"

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

        print(
            f'        \\multirow{{2}}{{*}}{{\\shortstack[c]{{{dataset_name}}}}} & OA & {" & ".join(cells_content["oa"])} \\\\'
        )
        print(f'         & AA & {" & ".join(cells_content["aa"])} \\\\')

        if i < len(datasets) - 1:
            print("        \\midrule")

    # Add global average row
    if "__global__" in stats:
        print("        \\midrule")
        print("        \\midrule")

        global_results = stats["__global__"]

        best_oa_val = max(
            [global_results[config][f"oa{metric_suffix}"][0] for config in configs if config in global_results]
        )
        best_aa_val = max(
            [global_results[config][f"aa{metric_suffix}"][0] for config in configs if config in global_results]
        )

        # Format cells for global averages
        cells_content = {"oa": [], "aa": []}
        for config in configs:
            if config in global_results:
                for metric in cells_content.keys():
                    metric_mean, metric_std = global_results[config][f"{metric}{metric_suffix}"]
                    if "std" in display_mode:
                        formatted_cell_text = f"{metric_mean:.1f} $\\pm$ {metric_std:.1f}"
                    else:
                        formatted_cell_text = f"{metric_mean:.1f}"

                    # Check if is the best value to bold
                    if metric_mean == (best_oa_val if metric == "oa" else best_aa_val):
                        formatted_cell_text = f"\\textbf{{{formatted_cell_text}}}"

                    cells_content[metric].append(formatted_cell_text)
            else:
                cells_content["oa"].append("-")
                cells_content["aa"].append("-")

        print(
            f'        \\multirow{{2}}{{*}}{{\\shortstack[c]{{\\textbf{{Average}}}}}} & OA & {" & ".join(cells_content["oa"])} \\\\'
        )
        print(f'         & AA & {" & ".join(cells_content["aa"])} \\\\')

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


def preprocess_data(csv_path, manifold=False):
    """Loads and preprocesses the experiment data from a CSV file."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
        return None

    df = complete_df_with_seed_and_config(df)
    if manifold:
        df.dropna(subset=["mean_fid", "mean_precision", "mean_recall"], inplace=True)
    else:
        df.dropna(subset=["oa_test", "aa_test"], inplace=True)
    return df


def main():
    args = parse_args()
    check_args(args)

    # Complement display_mode if 'all' is selected
    if args.display_mode == ["all"]:
        args.display_mode = ["std", "symbols", "pvalues"]

    df = preprocess_data(args.csv_path, manifold=args.manifold)
    if df is None:
        return

    # If manifold metrics are requested, use a separate flow
    if args.manifold:
        # Group by dataset and config for stats calculation
        grouped = df.groupby(["dataset", "config"])

        # Calculate manifold statistics
        stats = calculate_manifold_stats(grouped)

        print("\n--- Manifold Metrics (FID, Precision, Recall) ---")
        generate_manifold_stats_table(stats, display_mode=args.display_mode)
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
