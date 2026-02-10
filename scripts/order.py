import argparse

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Order experiments CSV file")
    parser.add_argument(
        "-i",
        "--input",
        dest="input_path",
        default="experiments_accuracies.csv",
        help="Path to the input CSV file (default: 'experiments_accuracies.csv')",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        default="results_aa.csv",
        help="Path to the output ordered CSV file (default: 'results_aa.csv')",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input_path)

    df["uniform_class"] = df["uniform_class"].astype(str).str.strip().map({"True": True, "False": False})
    df["disc_on_gen"] = df["disc_on_gen"].astype(str).str.strip().map({"True": True, "False": False})

    df_sorted = df.sort_values(
        by=["dataset", "uniform_class", "disc_on_gen", "classification_weight", "experiment_name"],
        ascending=[True, False, False, True, True],
    )

    df_sorted.to_csv(args.output_path, index=False)
    print(f"✅ File '{args.output_path}' ordered successfully.")


if __name__ == "__main__":
    main()
