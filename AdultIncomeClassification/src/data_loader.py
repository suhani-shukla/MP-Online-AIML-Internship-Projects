"""
data_loader.py
Loads the Adult Income dataset and does minimal, non-destructive cleanup.
"""

import pandas as pd
from pathlib import Path

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income"
]

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "adult.csv"


def load_raw_data(path: str = DEFAULT_DATA_PATH, has_header: bool = True) -> pd.DataFrame:
    """
    Load the Adult Income dataset from CSV.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    has_header : bool
        Set False if using the raw UCI file with no header row.

    Returns
    -------
    pd.DataFrame
    """
    if has_header:
        df = pd.read_csv(path, skipinitialspace=True)
    else:
        df = pd.read_csv(path, names=COLUMNS, header=None, skipinitialspace=True)

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names, replace '?' with NaN, strip whitespace,
    normalize the target column.
    """
    df = df.copy()

    # Normalize column names: dots -> hyphens (Kaggle version uses dots)
    df.columns = [col.replace(".", "-") for col in df.columns]

    # Replace '?' with NaN
    df.replace("?", pd.NA, inplace=True)

    # Strip whitespace from string columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # Normalize target column (Kaggle version sometimes has a trailing '.')
    if "income" in df.columns:
        df["income"] = df["income"].str.replace(".", "", regex=False)

    return df

    
def load_clean_data(path: str = DEFAULT_DATA_PATH, has_header: bool = True) -> pd.DataFrame:
    """Convenience wrapper: load + clean in one call."""
    df = load_raw_data(path, has_header=has_header)
    df = clean_data(df)
    return df


if __name__ == "__main__":
    df = load_clean_data()
    print(df.shape)
    print(df.isnull().sum())
    print(df["income"].value_counts())