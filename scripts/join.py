#!/usr/bin/env python3
"""
Script to join two CSV files using a join on key columns.
Fills missing rows in one of the datasets with 0s.

The operation is detected automatically from the column structure:
  concat - if the extra columns of one file are a subset of the other's
  join   - if both files have disjoint extra columns
"""

import argparse
import os
import sys

import pandas as pd

BASE_COLUMNS = {
    "experiment_name",
    "dataset",
    "uniform_class",
    "disc_on_gen",
    "autoencoder_epochs",
    "classification_weight",
    "best_tick_kimg",
}


def _read_and_normalise(path):
    """Read a CSV and convert experiment_name to basename."""
    print(f"Reading {path}...")
    df = pd.read_csv(path)
    if "experiment_name" in df.columns:
        print("Converting experiment_name to basename...")
        df["experiment_name"] = df["experiment_name"].apply(
            lambda x: os.path.basename(str(x))
        )
    return df


def _sort_dataframe(df):
    """Sort dataframe by the standard columns, handling boolean conversions."""
    sort_columns = [
        "dataset",
        "uniform_class",
        "disc_on_gen",
        "classification_weight",
        "experiment_name",
    ]
    sort_columns = [col for col in sort_columns if col in df.columns]

    if "uniform_class" in df.columns:
        df["uniform_class"] = (
            df["uniform_class"]
            .astype(str)
            .str.strip()
            .map({"True": True, "False": False, "1": True, "0": False, "1.0": True, "0.0": False})
        )
    if "disc_on_gen" in df.columns:
        df["disc_on_gen"] = (
            df["disc_on_gen"]
            .astype(str)
            .str.strip()
            .map({"True": True, "False": False, "1": True, "0": False, "1.0": True, "0.0": False})
        )

    ascending = [True] * len(sort_columns)
    # uniform_class and disc_on_gen descending (True before False)
    for i, col in enumerate(sort_columns):
        if col in ("uniform_class", "disc_on_gen"):
            ascending[i] = False

    print("Sorting the dataframe...")
    return df.sort_values(by=sort_columns, ascending=ascending)


def concat_csvs(df1, df2, output_path):
    """
    Concatenates two DataFrames by stacking rows vertically.
    Columns absent in one file (e.g. generation metrics) are filled with null
    automatically by pd.concat.
    """
    print("Concatenating rows...")
    df_merged = pd.concat([df1, df2], ignore_index=True)

    df_merged = _sort_dataframe(df_merged)

    print(f"Saving result to {output_path}...")
    df_merged.to_csv(output_path, index=False)

    print("Concat completed successfully!")
    print(f"  - Rows in CSV1: {len(df1)}")
    print(f"  - Rows in CSV2: {len(df2)}")
    print(f"  - Rows in result: {len(df_merged)}")
    print(f"  - Columns in result: {len(df_merged.columns)}")


def join_csvs(df1, df2, output_path):
    """
    Joins two DataFrames using an outer join on key columns.

    Args:
        df1: First DataFrame (already loaded and normalised)
        df2: Second DataFrame (already loaded and normalised)
        output_path: Path to the output CSV file
    """
    key_columns = [
        "experiment_name",
        "dataset",
        "uniform_class",
        "disc_on_gen",
        "autoencoder_epochs",
        "classification_weight",
    ]

    # Verify that the key columns exist in both dataframes
    for col in key_columns:
        if col not in df1.columns:
            print(f"ERROR: Column '{col}' does not exist in CSV1")
            sys.exit(1)
        if col not in df2.columns:
            print(f"ERROR: Column '{col}' does not exist in CSV2")
            sys.exit(1)

    # Perform an outer join to keep all rows from both datasets
    print("Performing the join...")
    df_merged = pd.merge(df1, df2, on=key_columns, how="outer", suffixes=("", "_dup"))

    # Fill missing values with 0s
    print("Filling missing values with 0s...")
    df_merged = df_merged.fillna(0)

    # Remove duplicate columns if they exist
    df_merged = df_merged.loc[:, ~df_merged.columns.str.endswith("_dup")]

    df_merged = _sort_dataframe(df_merged)

    print(f"Saving result to {output_path}...")
    df_merged.to_csv(output_path, index=False)

    print("Join completed successfully!")
    print(f"  - Rows in CSV1: {len(df1)}")
    print(f"  - Rows in CSV2: {len(df2)}")
    print(f"  - Rows in result: {len(df_merged)}")
    print(f"  - Columns in result: {len(df_merged.columns)}")


def detect_mode(df1, df2):
    """
    Detect whether to concat or join based on extra (non-base) columns.

    Returns 'concat' if the extra columns of one file are a subset of the
    other's extra columns; returns 'join' if both files have disjoint extras.
    """
    extra1 = set(df1.columns) - BASE_COLUMNS
    extra2 = set(df2.columns) - BASE_COLUMNS

    if extra1 <= extra2 or extra2 <= extra1:
        return "concat"
    return "join"


def main():
    """Main function that processes arguments and executes the join or concat."""
    parser = argparse.ArgumentParser(
        description="Joins or concatenates two CSV files (operation is auto-detected).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--csv1", "-1", type=str, required=True, help="Path to the first CSV file")
    parser.add_argument("--csv2", "-2", type=str, required=True, help="Path to the second CSV file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to the output CSV file")
    args = parser.parse_args()

    df1 = _read_and_normalise(args.csv1)
    df2 = _read_and_normalise(args.csv2)

    mode = detect_mode(df1, df2)
    print(f"Auto-detected mode: {mode}")

    if mode == "concat":
        concat_csvs(df1, df2, args.output)
    else:
        join_csvs(df1, df2, args.output)


if __name__ == "__main__":
    main()
