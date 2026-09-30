"""Tests for BaseImageClassificationDataset contract."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.transforms import v2

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
        assert isinstance(img, tv_tensors.Image)
        assert img.dtype == torch.uint8
        assert img.shape == (3, 8, 8)  # C, H, W
        assert label == 0

    def test_channels_are_rgb(self, tmp_image_factory):
        path = tmp_image_factory("red.png", size=(4, 4), color=(255, 0, 0))
        ds = ConcreteImageClassificationDataset([(str(path), 0)], {"a": 0})
        img, _ = ds[0]
        assert img[:, 0, 0].tolist() == [255, 0, 0]

    def test_return_path_is_str_for_path_samples(self, tmp_image_factory):
        path = tmp_image_factory("img.png")
        ds = ConcreteImageClassificationDataset([(path, 0)], {"a": 0}, return_path=True)
        assert ds[0][2] == str(path)

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
        transform = lambda x: x[:, :4, :4]  # noqa: E731 — crop
        ds = ConcreteImageClassificationDataset(
            [(str(path), 0)], {"a": 0}, transform=transform
        )
        img, _ = ds[0]
        assert img.shape[1:] == (4, 4)

    def test_torchvision_v2_pipeline(self, tmp_image_factory):
        """Regression: images used to be numpy HWC arrays, which Resize rejects."""
        path = tmp_image_factory("img.png", size=(10, 6))
        transform = v2.Compose([v2.Resize((4, 4)), v2.ToDtype(torch.float32, scale=True)])
        ds = ConcreteImageClassificationDataset([(str(path), 0)], {"a": 0}, transform=transform)
        img, _ = ds[0]
        assert img.shape == (3, 4, 4)
        assert img.dtype == torch.float32
        assert float(img.max()) <= 1.0

    def test_dataloader_batches_are_nchw(self, tmp_image_factory):
        paths = [tmp_image_factory(f"img_{i}.png", size=(8, 6)) for i in range(2)]
        ds = ConcreteImageClassificationDataset([(str(p), 0) for p in paths], {"a": 0})
        images, labels = next(iter(DataLoader(ds, batch_size=2)))
        assert images.shape == (2, 3, 6, 8)
        assert labels.tolist() == [0, 0]


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
