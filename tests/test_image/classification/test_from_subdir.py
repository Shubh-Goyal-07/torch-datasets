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


class TestCaseSensitiveOrder:
    def test_class_indices_follow_case_sensitive_name_order(self, tmp_path, tmp_image_factory):
        """Same indices as torchvision's ImageFolder, on every OS (Windows included)."""
        for name in ["bird/0.png", "Cat/0.png", "dog/0.png"]:
            tmp_image_factory(name)
        assert ImageSubdirDataset(tmp_path).class_to_idx == {"Cat": 0, "bird": 1, "dog": 2}


class TestDefaultExtensions:
    def test_tif_and_webp_are_loaded_by_default(self, tmp_path, tmp_image_factory):
        for name in ["cat/a.tif", "cat/b.tiff", "dog/c.webp", "dog/d.jpeg"]:
            tmp_image_factory(name)
        ds = ImageSubdirDataset(tmp_path)
        assert len(ds) == 4
        assert all(ds[i][0].shape == (3, 8, 8) for i in range(4))


class TestErrorCases:
    def test_empty_dir_raises(self, tmp_path):
        (tmp_path / "empty_class").mkdir()
        with pytest.raises(FileNotFoundError, match="No files with extensions"):
            ImageSubdirDataset(root_dir=tmp_path)

    def test_no_matching_extensions(self, tmp_path, tmp_image_factory):
        tmp_image_factory("cls/a.png")
        with pytest.raises(FileNotFoundError, match=r"No files with extensions \['.bmp'\]"):
            ImageSubdirDataset(root_dir=tmp_path, extensions=[".bmp"])

    def test_no_class_folders_raises(self, tmp_path, tmp_image_factory):
        tmp_image_factory("loose.png")
        with pytest.raises(FileNotFoundError, match="No class folders"):
            ImageSubdirDataset(root_dir=tmp_path)

    def test_missing_root_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Directory not found"):
            ImageSubdirDataset(root_dir=tmp_path / "nope")


class TestOrdering:
    def test_samples_are_sorted_within_each_class(self, tmp_path, tmp_image_factory):
        for name in ["c.png", "a.png", "b.png"]:
            tmp_image_factory(f"cls/{name}")
        ds = ImageSubdirDataset(root_dir=tmp_path)
        assert [p.name for p, _ in ds.samples] == ["a.png", "b.png", "c.png"]


class TestHiddenFiles:
    def test_hidden_folders_and_files_are_ignored(self, image_subdir_tree):
        root, classes = image_subdir_tree
        (root / ".ipynb_checkpoints").mkdir()
        (root / ".ipynb_checkpoints" / "img_0.png").write_bytes(b"not an image")
        (root / "cat" / "._img_0.png").write_bytes(b"macOS metadata")
        (root / "cat" / ".DS_Store").write_bytes(b"")
        ds = ImageSubdirDataset(root)
        assert sorted(ds.class_to_idx) == sorted(classes)
        assert len(ds) == sum(classes.values())
        for i in range(len(ds)):
            ds[i]  # every kept file is a real image
