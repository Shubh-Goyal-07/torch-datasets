"""Tests for BaseImageClassificationDataset contract."""

import numpy as np
import pytest

from torchdatasets.image.classification.base import BaseImageClassificationDataset


class ConcreteImageClassificationDataset(BaseImageClassificationDataset):
    """Minimal concrete subclass for testing the base class behaviour."""

    def __init__(self, samples, class_to_idx, **kwargs):
        super().__init__(**kwargs)
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.create_metadata()


class TestLen:
    def test_length_matches_samples(self, tmp_image_factory):
        paths = [tmp_image_factory(f"img_{i}.png") for i in range(4)]
        samples = [(str(p), 0) for p in paths]
        ds = ConcreteImageClassificationDataset(samples, {"cls": 0})
        assert len(ds) == 4

    def test_empty_samples(self):
        ds = ConcreteImageClassificationDataset([], {})
        assert len(ds) == 0


class TestGetItem:
    def test_returns_image_and_label(self, tmp_image_factory):
        path = tmp_image_factory("img.png", size=(8, 8))
        ds = ConcreteImageClassificationDataset([(str(path), 0)], {"a": 0})
        img, label = ds[0]
        assert isinstance(img, np.ndarray)
        assert label == 0

    def test_return_path_gives_3_tuple(self, tmp_image_factory):
        path = tmp_image_factory("img.png")
        ds = ConcreteImageClassificationDataset(
            [(str(path), 1)], {"b": 1}, return_path=True
        )
        result = ds[0]
        assert len(result) == 3
        assert result[2] == str(path)

    def test_transform_is_applied(self, tmp_image_factory):
        path = tmp_image_factory("img.png", size=(8, 8))
        transform = lambda x: x[:4, :4]  # noqa: E731 — crop
        ds = ConcreteImageClassificationDataset(
            [(str(path), 0)], {"a": 0}, transform=transform
        )
        img, _ = ds[0]
        assert img.shape[:2] == (4, 4)


class TestMetadata:
    def test_idx_to_class_is_inverse(self, tmp_image_factory):
        path = tmp_image_factory("img.png")
        class_to_idx = {"cat": 0, "dog": 1}
        ds = ConcreteImageClassificationDataset(
            [(str(path), 0), (str(path), 1)], class_to_idx
        )
        for cls, idx in class_to_idx.items():
            assert ds.idx_to_class[idx] == cls

    def test_class_count(self, tmp_image_factory):
        path = tmp_image_factory("img.png")
        samples = [(str(path), 0), (str(path), 0), (str(path), 1)]
        ds = ConcreteImageClassificationDataset(
            samples, {"a": 0, "b": 1}
        )
        assert ds.class_count[0] == 2
        assert ds.class_count[1] == 1


class TestExtensions:
    def test_default_extensions_set(self):
        ds = ConcreteImageClassificationDataset([], {})
        assert ".png" in ds.extensions
        assert ".jpg" in ds.extensions

    def test_custom_extensions(self):
        ds = ConcreteImageClassificationDataset(
            [], {}, extensions=[".TIFF", ".BMP"]
        )
        assert ds.extensions == {".tiff", ".bmp"}
