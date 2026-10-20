"""Helpers for datasets with one sub-folder per class (``root/cat/001.jpg``)."""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

from torchdatasets._internal.files import list_files, list_subfolders
from torchdatasets._internal.labels import split_labels


def samples_from_subfolders(
    root: Path,
    extensions: Iterable[str],
    label_sep: Optional[str] = None,
) -> Tuple[List[Tuple[Path, Union[int, List[int]]]], Dict[str, int]]:
    """Index a folder with one sub-folder per class.

    Only visible sub-folders count (hidden ones such as ``.ipynb_checkpoints`` are skipped), and
    only the visible files directly inside them that have an allowed extension. Every sub-folder
    defines its classes, even if it holds no files.

    Args:
        root (Path): Folder holding the class folders.
        extensions (Iterable[str]): Allowed file extensions (compared case-insensitively).
        label_sep (Optional[str], optional): Separator between several labels in one folder name,
            e.g. ``","`` for ``cat,dog/`` (multi-label). If None, each folder name is one class.
            Defaults to None.

    Raises:
        ValueError: If ``label_sep`` is an empty string, or a folder name has no label between the
            separators.
        FileNotFoundError: If ``root`` is not a directory, has no class folders, or no files with an
            allowed extension.

    Returns:
        Tuple[List[Tuple[Path, Union[int, List[int]]]], Dict[str, int]]: ``(path, label)`` pairs sorted
        by folder and then file name, where the label is a class index (or a list of class indices
        when ``label_sep`` is set), and the class name -> index mapping (class names sorted).
    """
    if label_sep is not None and not label_sep:
        raise ValueError("label_sep must be a non-empty string, or None for one class per folder.")
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: '{root}'")

    folders = list_subfolders(root)  # sorted by name, hidden folders left out
    if not folders:
        raise FileNotFoundError(f"No class folders found in '{root}' (expected one sub-folder per class).")

    if label_sep is None:
        folder_labels = {folder.name: [folder.name] for folder in folders}
    else:
        folder_labels = {}
        for folder in folders:
            labels = split_labels(folder.name, label_sep)
            if not labels:
                raise ValueError(
                    f"Folder '{folder.name}' in '{root}' has no class name between the '{label_sep}' separators."
                )
            folder_labels[folder.name] = labels

    classes = sorted({label for labels in folder_labels.values() for label in labels})
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    samples: List[Tuple[Path, Union[int, List[int]]]] = []
    for folder in folders:
        indices = [class_to_idx[label] for label in folder_labels[folder.name]]
        for path in list_files(folder, extensions):  # sorted, hidden files (e.g. ._x.jpg) left out
            samples.append((path, list(indices) if label_sep is not None else indices[0]))

    if not samples:
        raise FileNotFoundError(f"No files with extensions {sorted(extensions)} found in the class folders of '{root}'.")
    return samples, class_to_idx
