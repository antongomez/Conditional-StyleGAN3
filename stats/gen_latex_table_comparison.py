#!/usr/bin/env python3
"""
Script to generate LaTeX table comparing StyleGAN3 and MViT-DDPM across datasets.
"""

import argparse
from pathlib import Path

import pandas as pd

# Dataset names mapping
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
    parser = argparse.ArgumentParser(description="Generate LaTeX table comparing StyleGAN3 and MViT-DDPM")
    parser.add_argument("-c", "--csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("-o", "--output", type=str, default=None, help="Path to output file (default: stdout)")
    return parser.parse_args()


def load_data(csv_path):
    """Load CSV data and filter for StyleGAN3 and MViT-DDPM."""
    df = pd.read_csv(csv_path)

    # Filter only StyleGAN3 and MViT-DDPM (ignore ResBaGAN)
    df = df[df["network"].isin(["StyleGAN3", "MViT-DDPM"])].copy()

    return df


def format_value(mean, std, as_percentage=False):
    """Format a value as mean ± std."""
    if as_percentage:
        return f"{mean*100:.1f} $\\pm$ {std*100:.1f}"
    else:
        return f"{mean:.1f} $\\pm$ {std:.1f}"


def generate_latex_table(df):
    """Generate LaTeX code for the comparison table."""

    # Sort datasets in alphabetical order
    dataset_order = ["eiras", "ermidas", "ferreiras", "mera", "mestas", "oitaven", "ulla", "xesta"]

    # Begin LaTeX table
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Comparison of FID, Precision, and Recall for StyleGAN3 and MViT-DDPM across datasets}")
    latex.append("\\label{tab:model_comparison}")
    latex.append("\\begin{tabular}{llcc}")
    latex.append("\\toprule")

    # Header row with model names
    latex.append("\\textbf{Dataset} &  & \\textbf{MViT-DDPM} & \\textbf{StyleGAN3} \\\\")
    latex.append("\\midrule")

    # Data rows - each dataset has 3 rows (FID, Precision, Recall)
    for idx, dataset in enumerate(dataset_order):
        dataset_df = df[df["dataset"] == dataset]

        if len(dataset_df) == 0:
            continue

        # Get data for each network
        mvit_data = dataset_df[dataset_df["network"] == "MViT-DDPM"]
        sg3_data = dataset_df[dataset_df["network"] == "StyleGAN3"]

        # Get display name
        display_name = dataset_names_map.get(dataset, dataset.capitalize())

        # FID row
        fid_mvit = (
            format_value(mvit_data.iloc[0]["fid_mean"], mvit_data.iloc[0]["fid_std"]) if not mvit_data.empty else "--"
        )
        fid_sg3 = (
            format_value(sg3_data.iloc[0]["fid_mean"], sg3_data.iloc[0]["fid_std"]) if not sg3_data.empty else "--"
        )
        latex.append(f"\\multirow{{3}}{{*}}{{{display_name}}} & FID $\\downarrow$ & {fid_mvit} & {fid_sg3} \\\\")

        # Precision row
        prec_mvit = (
            format_value(mvit_data.iloc[0]["precision_mean"], mvit_data.iloc[0]["precision_std"], as_percentage=True)
            if not mvit_data.empty
            else "--"
        )
        prec_sg3 = (
            format_value(sg3_data.iloc[0]["precision_mean"], sg3_data.iloc[0]["precision_std"], as_percentage=True)
            if not sg3_data.empty
            else "--"
        )
        latex.append(f" & Precision \\% $\\uparrow$ & {prec_mvit} & {prec_sg3} \\\\")

        # Recall row
        rec_mvit = (
            format_value(mvit_data.iloc[0]["recall_mean"], mvit_data.iloc[0]["recall_std"], as_percentage=True)
            if not mvit_data.empty
            else "--"
        )
        rec_sg3 = (
            format_value(sg3_data.iloc[0]["recall_mean"], sg3_data.iloc[0]["recall_std"], as_percentage=True)
            if not sg3_data.empty
            else "--"
        )
        latex.append(f" & Recall \\% $\\uparrow$ & {rec_mvit} & {rec_sg3} \\\\")

        # Add midrule between datasets (except after the last one)
        if idx < len(dataset_order) - 1:
            latex.append("\\midrule")

    # End table
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    return "\n".join(latex)


def main():
    """Main function."""
    args = parse_args()

    # Load data
    df = load_data(args.csv)

    # Generate LaTeX table
    latex_table = generate_latex_table(df)

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(latex_table)
        print(f"LaTeX table written to {output_path}")
    else:
        print(latex_table)


if __name__ == "__main__":
    main()
