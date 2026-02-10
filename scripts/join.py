#!/usr/bin/env python3
"""
Script to join two CSV files using a join on key columns.
Fills missing rows in one of the datasets with 0s.
"""

import argparse
import os
import sys

import pandas as pd


def join_csvs(csv1_path, csv2_path, output_path):
    """
    Joins two CSV files using an outer join on key columns.

    Args:
        csv1_path: Path to the first CSV file
        csv2_path: Path to the second CSV file
        output_path: Path to the output CSV file
    """
    # Key columns for the join
    key_columns = [
        "experiment_name",
        "dataset",
        "uniform_class",
        "disc_on_gen",
        "autoencoder_epochs",
        "classification_weight",
    ]

    try:
        # Read the CSV files
        print(f"Reading {csv1_path}...")
        df1 = pd.read_csv(csv1_path)

        print(f"Reading {csv2_path}...")
        df2 = pd.read_csv(csv2_path)

        # Convert experiment_name to basename in both dataframes
        if "experiment_name" in df1.columns:
            print("Converting experiment_name to basename in CSV1...")
            df1["experiment_name"] = df1["experiment_name"].apply(lambda x: os.path.basename(str(x)))

        if "experiment_name" in df2.columns:
            print("Converting experiment_name to basename in CSV2...")
            df2["experiment_name"] = df2["experiment_name"].apply(lambda x: os.path.basename(str(x)))

        # Verify that the key columns exist in both dataframes
        for col in key_columns:
            if col not in df1.columns:
                print(f"ERROR: Column '{col}' does not exist in {csv1_path}")
                sys.exit(1)
            if col not in df2.columns:
                print(f"ERROR: Column '{col}' does not exist in {csv2_path}")
                sys.exit(1)

        # Perform an outer join to keep all rows from both datasets
        print("Performing the join...")
        df_merged = pd.merge(df1, df2, on=key_columns, how="outer", suffixes=("", "_dup"))

        # Fill missing values with 0s
        print("Filling missing values with 0s...")
        df_merged = df_merged.fillna(0)

        # Remove duplicate columns if they exist
        df_merged = df_merged.loc[:, ~df_merged.columns.str.endswith("_dup")]

        # Sort the dataframe
        print("Sorting the dataframe...")
        sort_columns = [
            "dataset",
            "uniform_class",
            "disc_on_gen",
            "classification_weight",
            "experiment_name",
        ]
        # Make sure the sort columns exist
        for col in sort_columns:
            if col not in df_merged.columns:
                print(f"WARNING: Sort column '{col}' does not exist in the dataframe. Skipping.")
                sort_columns.remove(col)

        # Convert boolean columns to sort correctly
        if "uniform_class" in df_merged.columns:
            df_merged["uniform_class"] = (
                df_merged["uniform_class"]
                .astype(str)
                .str.strip()
                .map({"True": True, "False": False, "1": True, "0": False, "1.0": True, "0.0": False})
            )
        if "disc_on_gen" in df_merged.columns:
            df_merged["disc_on_gen"] = (
                df_merged["disc_on_gen"]
                .astype(str)
                .str.strip()
                .map({"True": True, "False": False, "1": True, "0": False, "1.0": True, "0.0": False})
            )

        df_merged = df_merged.sort_values(
            by=sort_columns,
            ascending=[True, False, False, True, True],
        )

        # Save the result
        print(f"Saving result to {output_path}...")
        df_merged.to_csv(output_path, index=False)

        print(f"✓ Join completed successfully!")
        print(f"  - Rows in CSV1: {len(df1)}")
        print(f"  - Rows in CSV2: {len(df2)}")
        print(f"  - Rows in result: {len(df_merged)}")
        print(f"  - Columns in result: {len(df_merged.columns)}")

    except FileNotFoundError as e:
        print(f"ERROR: File not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


def main():
    """Main function that processes arguments and executes the join."""
    parser = argparse.ArgumentParser(
        description="Joins two CSV files using a join on key columns.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--csv1", "-1", type=str, required=True, help="Path to the first CSV file")
    parser.add_argument("--csv2", "-2", type=str, required=True, help="Path to the second CSV file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to the output CSV file")
    args = parser.parse_args()

    # Execute the join
    join_csvs(args.csv1, args.csv2, args.output)


if __name__ == "__main__":
    main()
