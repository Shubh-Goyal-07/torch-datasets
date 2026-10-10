"""Tests for torchdatasets._internal.files — listing dataset files."""

from torchdatasets._internal.files import is_hidden, list_files, list_subfolders


def test_is_hidden(tmp_path):
    assert is_hidden(tmp_path / ".DS_Store")
    assert is_hidden(tmp_path / "._cat_001.jpg")  # macOS AppleDouble sidecar
    assert is_hidden(tmp_path / ".ipynb_checkpoints")
    assert not is_hidden(tmp_path / "cat_001.jpg")


def test_list_files_skips_hidden_files_folders_and_other_extensions(tmp_path):
    for name in ["b.png", "a.PNG", "._a.png", ".hidden.png", "notes.txt"]:
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "folder.png").mkdir()  # a folder named like an image
    assert [p.name for p in list_files(tmp_path, [".png"])] == ["a.PNG", "b.png"]


def test_list_files_compares_extensions_case_insensitively(tmp_path):
    (tmp_path / "a.png").write_bytes(b"")
    assert [p.name for p in list_files(tmp_path, [".PNG"])] == ["a.png"]


def test_list_subfolders_skips_hidden_folders_and_files(tmp_path):
    for name in ["dog", "cat", ".ipynb_checkpoints", ".git"]:
        (tmp_path / name).mkdir()
    (tmp_path / "readme.txt").write_text("x")
    assert [p.name for p in list_subfolders(tmp_path)] == ["cat", "dog"]


def test_sorted_by_name_with_upper_case_first_on_every_os(tmp_path):
    """Sorting Path objects ignores case on Windows only; names must sort the same everywhere."""
    for name in ["b.png", "B2.png", "a.png", "A1.png"]:
        (tmp_path / name).write_bytes(b"")
    for name in ["dog", "Cat", "bird", "Zebra"]:
        (tmp_path / name).mkdir()
    assert [p.name for p in list_files(tmp_path, [".png"])] == ["A1.png", "B2.png", "a.png", "b.png"]
    assert [p.name for p in list_subfolders(tmp_path)] == ["Cat", "Zebra", "bird", "dog"]
