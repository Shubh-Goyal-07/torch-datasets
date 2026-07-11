"""
Shared test fixtures for torch-datasets.

All fixtures create synthetic data in pytest's tmp_path — nothing is committed
to the repository and every test run starts with a clean slate.
"""

import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Image fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_image_factory(tmp_path: Path):
    """Factory that creates tiny RGB test images on disk.

    Usage:
        path = tmp_image_factory("cats/img_0.png")
        path = tmp_image_factory("dogs/img_1.jpg", size=(16, 16), color=(0, 255, 0))
    """

    def _create(
        filename: str,
        size: tuple = (8, 8),
        color: tuple = (255, 0, 0),
    ) -> Path:
        img = Image.new("RGB", size, color)
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path)
        return path

    return _create


@pytest.fixture
def tmp_mask_factory(tmp_path: Path):
    """Factory that creates tiny grayscale test masks on disk.

    Usage:
        path = tmp_mask_factory("masks/mask_0.png")
        path = tmp_mask_factory("masks/mask_1.png", size=(16, 16), value=200)
    """

    def _create(
        filename: str,
        size: tuple = (8, 8),
        value: int = 128,
    ) -> Path:
        arr = np.full(size, value, dtype=np.uint8)
        img = Image.fromarray(arr, mode="L")
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path)
        return path

    return _create


# ---------------------------------------------------------------------------
# CSV / tabular fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_csv_factory(tmp_path: Path):
    """Factory that creates test CSV files from a list of row dicts.

    Usage:
        path = tmp_csv_factory("data.csv", [{"path": "a.png", "label": "cat"}, ...])
    """

    def _create(filename: str, rows: List[Dict]) -> Path:
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    return _create


# ---------------------------------------------------------------------------
# Composite fixtures (common directory layouts)
# ---------------------------------------------------------------------------


@pytest.fixture
def image_subdir_tree(tmp_path: Path, tmp_image_factory):
    """Creates a class-per-subdirectory image tree.

    Returns (root_path, class_counts_dict).
    """
    classes = {"cat": 3, "dog": 2}
    for cls_name, count in classes.items():
        for i in range(count):
            tmp_image_factory(f"{cls_name}/img_{i}.png")
    return tmp_path, classes


@pytest.fixture
def image_seg_paired_dirs(tmp_path: Path, tmp_image_factory, tmp_mask_factory):
    """Creates paired image/mask directories for segmentation multidir tests.

    Returns (image_dir, mask_dir, expected_count).
    """
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    names = ["sample_0", "sample_1", "sample_2"]
    for name in names:
        tmp_image_factory(f"images/{name}.png")
        tmp_mask_factory(f"masks/{name}.png")
    return img_dir, mask_dir, len(names)


@pytest.fixture
def image_seg_single_dir(tmp_path: Path, tmp_image_factory, tmp_mask_factory):
    """Creates a single directory with images and their mask counterparts
    identified by a suffix (e.g. ``img`` and ``img_mask``).

    Returns (src_dir, suffix, expected_count).
    """
    suffix = "_mask"
    names = ["a", "b"]
    for name in names:
        tmp_image_factory(f"seg/{name}.png")
        tmp_mask_factory(f"seg/{name}{suffix}.png")
    src_dir = tmp_path / "seg"
    return src_dir, suffix, len(names)
