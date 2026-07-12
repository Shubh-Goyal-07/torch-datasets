import pandas as pd
from pathlib import Path


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
