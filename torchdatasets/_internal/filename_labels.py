"""Helpers for datasets that read each sample's class from its file name (``cat_001.jpg``)."""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from torchdatasets._internal.files import list_files


def label_from_filename(path: Path, delimiter: str) -> str:
    """Return the class name at the start of a file name.

    Args:
        path (Path): The file.
        delimiter (str): Separator between the class name and the rest of the name.

    Returns:
        str: The part of the name (without extension) before the first ``delimiter``,
        or the whole name if it has no ``delimiter``.
    """
    return path.stem.split(delimiter, 1)[0]


def samples_from_filenames(
    root: Path,
    extensions: Iterable[str],
    delimiter: str,
    label_map: Optional[Dict[str, int]] = None,
) -> Tuple[List[Tuple[Path, int]], Dict[str, int]]:
    """Index a folder of files whose names start with their class name.

    Only files directly inside ``root`` are read (sub-folders and hidden files are ignored).

    Args:
        root (Path): Folder holding the files.
        extensions (Iterable[str]): Allowed file extensions (compared case-insensitively).
        delimiter (str): Separator between the class name and the rest of the file name.
        label_map (Optional[Dict[str, int]], optional): Class name -> class index. If None, the
            class names found are sorted and numbered 0..K-1. Defaults to None.

    Raises:
        ValueError: If ``delimiter`` is empty, a file name has nothing before the delimiter,
            or ``label_map`` has no entry for a class found in the folder.
        FileNotFoundError: If ``root`` is not a directory or has no file with an allowed extension.

    Returns:
        Tuple[List[Tuple[Path, int]], Dict[str, int]]: ``(path, class index)`` pairs sorted by
        path, and the class name -> index mapping.
    """
    if not delimiter:
        raise ValueError("delimiter must be a non-empty string.")
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: '{root}'")

    allowed = sorted({ext.lower() for ext in extensions})
    files = list_files(root, allowed)  # sorted, hidden files (e.g. ._cat_001.jpg) left out
    if not files:
        raise FileNotFoundError(f"No files with extensions {allowed} found in '{root}'.")

    named = [(path, label_from_filename(path, delimiter)) for path in files]

    unnamed = [path.name for path, name in named if not name]
    if unnamed:
        raise ValueError(
            f"{len(unnamed)} file name(s) in '{root}' have no class name before "
            f"'{delimiter}', e.g. {unnamed[:3]}."
        )

    names = {name for _, name in named}
    if label_map is None:
        class_to_idx = {name: idx for idx, name in enumerate(sorted(names))}
    else:
        class_to_idx = dict(label_map)
        missing = sorted(names - set(class_to_idx))
        if missing:
            raise ValueError(f"label_map has no entry for class(es) {missing} found in '{root}'.")

    return [(path, class_to_idx[name]) for path, name in named], class_to_idx
