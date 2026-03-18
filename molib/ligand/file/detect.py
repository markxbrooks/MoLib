"""
A set of utility functions for handling CSV input files.

This module provides functions to detect delimiters used in CSV files and
identify specific columns, such as SMILES and Name columns, in pandas DataFrames.
"""

import pandas as pd


def detect_delimiter(file_path):
    """Detect delimiter"""
    with open(file_path, "r", encoding="utf-8") as f:
        sample = f.read(1024)
        f.seek(0)
        try:
            delimiter = pd.io.common._get_delimiter(sample)
        except Exception:
            delimiter = ","
    return delimiter


def detect_smiles_and_name_columns(df):
    """Detect SMILES and Name columns"""
    smiles_col = next((c for c in df.columns if "smiles" in c.lower()), None)
    name_col = next(
        (c for c in df.columns if "name" in c.lower() and "smiles" not in c.lower()),
        None,
    )
    if not smiles_col:
        raise ValueError("No SMILES column found in CSV parser")
    return name_col, smiles_col
