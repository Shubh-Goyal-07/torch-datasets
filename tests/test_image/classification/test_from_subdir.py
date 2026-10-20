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


class TestMultiLabel:
    """label_sep: a folder name lists several labels (cat,dog/ holds images of both)."""

    @pytest.fixture
    def tagged_tree(self, tmp_path, tmp_image_factory):
        for name in ["cat/0.png", "cat,dog/0.png", "cat,dog/1.png", "dog, bird/0.png"]:
            tmp_image_factory(name)
        return tmp_path

    def test_classes_are_the_individual_labels(self, tagged_tree):
        ds = ImageSubdirDataset(tagged_tree, label_sep=",")
        assert ds.class_to_idx == {"bird": 0, "cat": 1, "dog": 2}

    def test_each_label_is_a_list_of_class_indices(self, tagged_tree):
        ds = ImageSubdirDataset(tagged_tree, label_sep=",")
        assert ds.targets == [[1], [1, 2], [1, 2], [2, 0]]  # folders sorted, labels in folder-name order
        image, label = ds[1]
        assert label == [1, 2]
        assert image.shape == (3, 8, 8)

    def test_class_count_counts_every_label(self, tagged_tree):
        ds = ImageSubdirDataset(tagged_tree, label_sep=",")
        assert ds.class_count == {0: 1, 1: 3, 2: 3}

    def test_return_path(self, tagged_tree):
        _, label, path = ImageSubdirDataset(tagged_tree, label_sep=",", return_path=True)[0]
        assert label == [1]
        assert path.endswith("0.png")

    def test_spaces_and_repeated_labels(self, tmp_path, tmp_image_factory):
        tmp_image_factory("cat , cat,dog/0.png")
        ds = ImageSubdirDataset(tmp_path, label_sep=",")
        assert ds.class_to_idx == {"cat": 0, "dog": 1}
        assert ds.targets == [[0, 1]]

    def test_other_separator(self, tmp_path, tmp_image_factory):
        tmp_image_factory("cat+dog/0.png")
        assert ImageSubdirDataset(tmp_path, label_sep="+").targets == [[0, 1]]

    def test_without_label_sep_a_folder_name_is_one_class(self, tagged_tree):
        ds = ImageSubdirDataset(tagged_tree)
        assert ds.class_to_idx == {"cat": 0, "cat,dog": 1, "dog, bird": 2}
        assert ds.targets == [0, 1, 1, 2]

    def test_folder_name_without_a_label_raises(self, tmp_path, tmp_image_factory):
        tmp_image_factory("cat/0.png")
        tmp_image_factory(",/0.png")
        with pytest.raises(ValueError, match="no class name"):
            ImageSubdirDataset(tmp_path, label_sep=",")

    def test_empty_label_sep_raises(self, tagged_tree):
        with pytest.raises(ValueError, match="label_sep"):
            ImageSubdirDataset(tagged_tree, label_sep="")

    def test_labels_are_separate_lists(self, tagged_tree):
        """Changing one sample's label list must not change another's."""
        ds = ImageSubdirDataset(tagged_tree, label_sep=",")
        ds.samples[1][1].append(99)
        assert ds.samples[2][1] == [1, 2]
