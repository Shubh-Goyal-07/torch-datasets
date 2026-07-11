"""Tests for ImageSubdirDataset — loads images from class-per-subdirectory layout."""

import pytest

from torchdatasets.image.classification.from_subdir import ImageSubdirDataset


class TestBasicLoading:
    def test_length(self, image_subdir_tree):
        root, classes = image_subdir_tree
        ds = ImageSubdirDataset(root_dir=root)
        assert len(ds) == sum(classes.values())

    def test_getitem_returns_2_tuple(self, image_subdir_tree):
        root, _ = image_subdir_tree
        ds = ImageSubdirDataset(root_dir=root)
        result = ds[0]
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_return_path(self, image_subdir_tree):
        root, _ = image_subdir_tree
        ds = ImageSubdirDataset(root_dir=root, return_path=True)
        result = ds[0]
        assert len(result) == 3


class TestClassMapping:
    def test_class_to_idx_keys(self, image_subdir_tree):
        root, classes = image_subdir_tree
        ds = ImageSubdirDataset(root_dir=root)
        assert set(ds.class_to_idx.keys()) == set(classes.keys())

    def test_inverse_consistency(self, image_subdir_tree):
        root, _ = image_subdir_tree
        ds = ImageSubdirDataset(root_dir=root)
        for cls, idx in ds.class_to_idx.items():
            assert ds.idx_to_class[idx] == cls

    def test_class_count(self, image_subdir_tree):
        root, classes = image_subdir_tree
        ds = ImageSubdirDataset(root_dir=root)
        for cls, expected in classes.items():
            idx = ds.class_to_idx[cls]
            assert ds.class_count[idx] == expected


class TestExtensionFilter:
    def test_filters_by_extension(self, tmp_path, tmp_image_factory):
        tmp_image_factory("animals/a.png")
        tmp_image_factory("animals/b.jpg")
        ds = ImageSubdirDataset(root_dir=tmp_path, extensions=[".png"])
        assert len(ds) == 1

    def test_case_insensitive_extension(self, tmp_path, tmp_image_factory):
        tmp_image_factory("animals/a.PNG")
        ds = ImageSubdirDataset(root_dir=tmp_path, extensions=[".png"])
        assert len(ds) == 1


class TestErrorCases:
    def test_empty_dir_raises(self, tmp_path):
        (tmp_path / "empty_class").mkdir()
        with pytest.raises(AssertionError, match="No valid samples"):
            ImageSubdirDataset(root_dir=tmp_path)

    def test_no_matching_extensions(self, tmp_path, tmp_image_factory):
        tmp_image_factory("cls/a.png")
        with pytest.raises(AssertionError, match="No valid samples"):
            ImageSubdirDataset(root_dir=tmp_path, extensions=[".bmp"])
