import pandas as pd
from pathlib import Path
from typing import Sequence


def require_columns(df: pd.DataFrame, columns: Sequence[str], file_path: Path) -> None:
    """Check that a loaded CSV/Excel file has the columns a dataset reads.

    Args:
        df (pd.DataFrame): The file's contents.
        columns (Sequence[str]): Column names that must be present.
        file_path (Path): The file, for the error message.

    Raises:
        ValueError: If any of ``columns`` is missing.
    """
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"'{file_path}' must have the column(s) {list(columns)}; missing: {missing} "
            f"(columns found: {list(df.columns)})."
        )


def load_csv_or_excel(file_path: Path) -> pd.DataFrame:
    """Load CSV or Excel file.

    Args:
        file_path (Path): Path of the file.

    Returns:
        pd.DataFrame: File contents in a DataFrame.
    """
    if file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path)
    elif file_path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)
    else:
        raise ValueError("Only .csv and .xlsx/.xls files are supported.")
