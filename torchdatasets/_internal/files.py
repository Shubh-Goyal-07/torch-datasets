"""Listing dataset files: sorted, and without hidden files or folders."""

from pathlib import Path
from typing import Iterable, List


def is_hidden(path: Path) -> bool:
    """Whether a file or folder is hidden: its name starts with ``"."``.

    This covers macOS ``._photo.jpg`` sidecar files (which match image extensions but aren't
    images), ``.DS_Store`` and folders like ``.ipynb_checkpoints`` or ``.git``.

    Args:
        path (Path): The file or folder.

    Returns:
        bool: True if hidden.
    """
    return path.name.startswith(".")


def _by_name(path: Path) -> str:
    # Sort on the name string: sorting Path objects ignores case on Windows only, which would
    # give a different order (and different class indices) there than on Linux and macOS.
    return path.name


def list_files(folder: Path, extensions: Iterable[str]) -> List[Path]:
    """Visible files directly inside ``folder`` with an allowed extension, sorted by name.

    Names are compared as plain strings (case-sensitive, so ``"B.png"`` comes before
    ``"a.png"``), giving the same order on every OS.

    Args:
        folder (Path): The folder to list.
        extensions (Iterable[str]): Allowed extensions (compared case-insensitively).

    Returns:
        List[Path]: The files.
    """
    allowed = {ext.lower() for ext in extensions}
    files = (p for p in folder.iterdir() if p.is_file() and not is_hidden(p) and p.suffix.lower() in allowed)
    return sorted(files, key=_by_name)


def list_subfolders(folder: Path) -> List[Path]:
    """Visible folders directly inside ``folder``, sorted by name (as in :func:`list_files`).

    Args:
        folder (Path): The folder to list.

    Returns:
        List[Path]: The sub-folders.
    """
    return sorted((p for p in folder.iterdir() if p.is_dir() and not is_hidden(p)), key=_by_name)
