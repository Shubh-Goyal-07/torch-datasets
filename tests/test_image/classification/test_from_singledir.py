"""Tests for ImageSingleDirDataset — one folder, class read from the file name prefix."""

import pytest
import torch
from torchvision import tv_tensors

from torchdatasets.image.classification.from_singledir import ImageSingleDirDataset


@pytest.fixture
def prefixed_images(tmp_path, tmp_image_factory):
    """cat x2, dog x1, bird x1 in one folder. Returns the folder."""
    for name in ["dog_0.png", "cat_1.png", "bird_0.jpg", "cat_0.png"]:
        tmp_image_factory(f"imgs/{name}", size=(6, 4))
    return tmp_path / "imgs"


class TestBasicLoading:
    def test_length(self, prefixed_images):
        assert len(ImageSingleDirDataset(prefixed_images)) == 4

    def test_class_to_idx_follows_sorted_class_names(self, prefixed_images):
        ds = ImageSingleDirDataset(prefixed_images)
        assert ds.class_to_idx == {"bird": 0, "cat": 1, "dog": 2}
        assert ds.idx_to_class == {0: "bird", 1: "cat", 2: "dog"}

    def test_samples_sorted_by_path_with_labels(self, prefixed_images):
        ds = ImageSingleDirDataset(prefixed_images)
        assert [(p.name, label) for p, label in ds.samples] == [
            ("bird_0.jpg", 0),
            ("cat_0.png", 1),
            ("cat_1.png", 1),
            ("dog_0.png", 2),
        ]

    def test_getitem_returns_uint8_image_and_int_label(self, prefixed_images):
        image, label = ImageSingleDirDataset(prefixed_images)[1]
        assert isinstance(image, tv_tensors.Image)
        assert image.dtype == torch.uint8
        assert image.shape == (3, 4, 6)
        assert label == 1

    def test_class_count(self, prefixed_images):
        ds = ImageSingleDirDataset(prefixed_images)
        assert ds.class_count == {0: 1, 1: 2, 2: 1}

    def test_targets(self, prefixed_images):
        assert ImageSingleDirDataset(prefixed_images).targets == [0, 1, 1, 2]

    def test_return_path(self, prefixed_images):
        image, label, path = ImageSingleDirDataset(prefixed_images, return_path=True)[0]
        assert path.endswith("bird_0.jpg")
        assert isinstance(path, str)

    def test_transform_applied(self, prefixed_images):
        ds = ImageSingleDirDataset(prefixed_images, transform=lambda img: img.float() / 255)
        image, _ = ds[0]
        assert image.dtype == torch.float32


class TestFileNames:
    def test_custom_delimiter(self, tmp_path, tmp_image_factory):
        tmp_image_factory("imgs/golden_retriever-001.png")
        tmp_image_factory("imgs/pug-001.png")
        ds = ImageSingleDirDataset(tmp_path / "imgs", delimiter="-")
        assert ds.class_to_idx == {"golden_retriever": 0, "pug": 1}

    def test_dot_delimiter(self, tmp_path, tmp_image_factory):
        for i in range(3):
            tmp_image_factory(f"imgs/class{i}.img{i}.jpg")
        ds = ImageSingleDirDataset(tmp_path / "imgs", delimiter=".")
        assert sorted(ds.class_to_idx) == ["class0", "class1", "class2"]

    def test_name_without_delimiter_is_used_whole(self, tmp_path, tmp_image_factory):
        tmp_image_factory("imgs/cat.png")
        tmp_image_factory("imgs/cat_2.png")
        ds = ImageSingleDirDataset(tmp_path / "imgs")
        assert ds.class_to_idx == {"cat": 0}
        assert ds.targets == [0, 0]

    def test_only_top_level_files_are_read(self, tmp_path, tmp_image_factory):
        tmp_image_factory("imgs/cat_0.png")
        tmp_image_factory("imgs/nested/dog_0.png")
        ds = ImageSingleDirDataset(tmp_path / "imgs")
        assert len(ds) == 1

    def test_extension_filter(self, tmp_path, tmp_image_factory):
        tmp_image_factory("imgs/cat_0.png")
        tmp_image_factory("imgs/dog_0.jpg")
        (tmp_path / "imgs" / "notes_0.txt").write_text("not an image")
        assert len(ImageSingleDirDataset(tmp_path / "imgs")) == 2
        assert len(ImageSingleDirDataset(tmp_path / "imgs", extensions=[".PNG"])) == 1


class TestLabelMap:
    def test_label_map_sets_indices(self, prefixed_images):
        label_map = {"cat": 10, "dog": 20, "bird": 30}
        ds = ImageSingleDirDataset(prefixed_images, label_map=label_map)
        assert ds.class_to_idx == label_map
        assert ds.targets == [30, 10, 10, 20]
        assert ds.idx_to_class[20] == "dog"

    def test_label_map_may_list_extra_classes(self, prefixed_images):
        ds = ImageSingleDirDataset(prefixed_images, label_map={"bird": 0, "cat": 1, "dog": 2, "fish": 3})
        assert 3 not in ds.class_count
        assert len(ds) == 4

    def test_label_map_missing_class_raises(self, prefixed_images):
        with pytest.raises(ValueError, match=r"no entry for class\(es\) \['bird'\]"):
            ImageSingleDirDataset(prefixed_images, label_map={"cat": 0, "dog": 1})

    def test_label_map_is_copied(self, prefixed_images):
        label_map = {"bird": 0, "cat": 1, "dog": 2}
        ds = ImageSingleDirDataset(prefixed_images, label_map=label_map)
        ds.class_to_idx["new"] = 3
        assert "new" not in label_map


class TestErrors:
    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Directory not found"):
            ImageSingleDirDataset(tmp_path / "nope")

    def test_no_images_raises(self, tmp_path):
        (tmp_path / "imgs").mkdir()
        (tmp_path / "imgs" / "readme.txt").write_text("hi")
        with pytest.raises(FileNotFoundError, match="No files with extensions"):
            ImageSingleDirDataset(tmp_path / "imgs")

    def test_empty_delimiter_raises(self, prefixed_images):
        with pytest.raises(ValueError, match="delimiter"):
            ImageSingleDirDataset(prefixed_images, delimiter="")

    def test_file_without_class_name_raises(self, tmp_path, tmp_image_factory):
        tmp_image_factory("imgs/cat_0.png")
        tmp_image_factory("imgs/_0.png")
        with pytest.raises(ValueError, match=r"no class name before '_'.*_0\.png"):
            ImageSingleDirDataset(tmp_path / "imgs")
