import os
import warnings
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd

from torchdatasets._internal.labels import split_labels


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


def load_csv_or_excel(file_path: Path, *, as_text: bool = False) -> pd.DataFrame:
    """Load CSV or Excel file.

    ``.xlsx`` files are read with openpyxl (installed with the package); old ``.xls`` files
    need ``pip install xlrd``.

    Args:
        file_path (Path): Path of the file.
        as_text (bool, optional): Read every cell as text, exactly as written: no type guessing
            (``"007"`` stays ``"007"``) and no missing-value words (``"NA"`` or ``"None"`` stay
            labels); only empty cells are empty. Defaults to False (pandas' usual parsing).

    Returns:
        pd.DataFrame: File contents in a DataFrame.
    """
    options = {"dtype": str, "keep_default_na": False} if as_text else {}
    if file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path, **options)
    elif file_path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(file_path, **options)
    else:
        raise ValueError("Only .csv and .xlsx/.xls files are supported.")


def is_blank(cell: Any) -> bool:
    """Whether a CSV/Excel cell is empty (missing value, or only whitespace).

    Args:
        cell (Any): The cell value.

    Returns:
        bool: True if the cell holds nothing.
    """
    if isinstance(cell, str):
        return not cell.strip()
    try:
        return bool(pd.isna(cell))
    except (TypeError, ValueError):
        return False


def cell_text(cell: Any) -> str:
    """A cell as text, without surrounding spaces; whole floats lose their ``.0``.

    Files are read as text (see :func:`load_csv_or_excel`), but a float can still arrive from
    a DataFrame read with type guessing, where an integer column with empty cells turns into
    floats and a label ``3`` would otherwise become ``"3.0"``.

    Args:
        cell (Any): The cell value.

    Returns:
        str: The text.
    """
    if isinstance(cell, float) and cell.is_integer():
        cell = int(cell)
    return str(cell).strip()


def folder_of(file_path: Path) -> Path:
    """The absolute folder holding a CSV/Excel file: relative paths in the file start there.

    Args:
        file_path (Path): The CSV/Excel file (absolute, or relative to the working directory).

    Returns:
        Path: The folder, as an absolute path.
    """
    return Path(os.path.normpath(Path(file_path).absolute().parent))


def resolve_path(cell: str, base_dir: Path) -> Path:
    """The file a path cell of a CSV/Excel file points to, as an absolute path.

    Absolute paths are used as written; relative ones start from ``base_dir`` (the CSV/Excel
    file's folder), so the working directory doesn't matter. The result is normalised
    (``imgs/../a.png`` becomes ``a.png``, separators follow the OS) without following
    symlinks, so ``return_path`` gives back an absolute path as written in the file.

    Args:
        cell (str): The cell's text.
        base_dir (Path): Folder that relative paths start from.

    Returns:
        Path: The absolute path.
    """
    path = Path(cell)
    if not path.is_absolute():
        path = Path(base_dir).absolute() / path
    return Path(os.path.normpath(path))


def row_name(position: int) -> str:
    """How a warning names a data row: its row number in a spreadsheet (the header is row 1).

    Args:
        position (int): 0-based position of the row among the data rows.

    Returns:
        str: E.g. ``"row 2"`` for the first data row.
    """
    return f"row {position + 2}"


def labelled_rows(
    df: pd.DataFrame,
    path_col: str,
    label_col: str,
    label_sep: Optional[str],
    extensions: Iterable[str],
    base_dir: Path,
) -> Tuple[List[Tuple[Path, Union[str, List[str]]]], List[str]]:
    """Rows that name an existing file with an allowed extension and have a label.

    Args:
        df (pd.DataFrame): The CSV/Excel contents.
        path_col (str): Column with the file paths (absolute, or relative to ``base_dir``).
        label_col (str): Column with the labels.
        label_sep (Optional[str]): Separator for several labels per row, or None for one label.
        extensions (Iterable[str]): Allowed file extensions (compared case-insensitively).
        base_dir (Path): Folder that relative paths start from (the CSV/Excel file's folder).

    Returns:
        Tuple[List[Tuple[Path, Union[str, List[str]]]], List[str]]: ``(absolute path, label)`` pairs
        (a list of labels when label_sep is set), and a description of each row that was left out
        (``"row 5: /data/imgs/x.png"``).
    """
    allowed = {ext.lower() for ext in extensions}
    rows: List[Tuple[Path, Union[str, List[str]]]] = []
    skipped: List[str] = []
    for position, (path_cell, label_cell) in enumerate(zip(df[path_col], df[label_col])):
        if is_blank(path_cell) or is_blank(label_cell):
            skipped.append(f"{row_name(position)}: empty cell")
            continue
        path = resolve_path(cell_text(path_cell), base_dir)
        if label_sep is None:
            label: Union[str, List[str]] = cell_text(label_cell)
        else:
            label = split_labels(cell_text(label_cell), label_sep)
        if not label:
            skipped.append(f"{row_name(position)}: no label")
        elif path.suffix.lower() in allowed and path.is_file():
            rows.append((path, label))
        else:
            skipped.append(f"{row_name(position)}: {path}")
    return rows, skipped


def no_usable_rows(file_path: Path, skipped: List[str], total: int, requirement: str) -> FileNotFoundError:
    """The error to raise when none of a file's rows can be used.

    Args:
        file_path (Path): The file.
        skipped (List[str]): A description of each skipped row.
        total (int): Number of rows in the file.
        requirement (str): What a row needs in order to be used.

    Returns:
        FileNotFoundError: The error, ready to raise.
    """
    examples = f"; first ones: {'; '.join(skipped[:3])}" if skipped else ""
    return FileNotFoundError(
        f"No valid entries in '{file_path}': none of its {total} row(s) can be used "
        f"({requirement}){examples}."
    )


def warn_skipped_rows(file_path: Path, skipped: List[str], total: int, requirement: str) -> None:
    """Warn that some rows of a CSV/Excel file were left out.

    Args:
        file_path (Path): The file.
        skipped (List[str]): A description of each skipped row.
        total (int): Number of rows in the file.
        requirement (str): What a row needs in order to be used.
    """
    if skipped:
        warnings.warn(
            f"Skipped {len(skipped)} of {total} row(s) in '{file_path}' ({requirement}); "
            f"first ones: {'; '.join(skipped[:3])}",
            stacklevel=4,  # dataset constructor -> _load_samples -> here: point at the caller
        )
