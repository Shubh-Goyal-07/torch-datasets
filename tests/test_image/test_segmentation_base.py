"""Tests for BaseImageSegmentationDataset contract."""

import torch
import numpy as np
import pytest

from torchdatasets.image.segmentation.base import BaseImageSegmentationDataset


class ConcreteSegmentationDataset(BaseImageSegmentationDataset):
    """Minimal concrete subclass for testing the base class behaviour."""

    def __init__(self, samples, **kwargs):
        super().__init__(**kwargs)
        self.samples = samples


class TestLen:
    def test_length_matches_samples(self, tmp_image_factory, tmp_mask_factory):
        img = tmp_image_factory("img.png", size=(8, 8))
        mask = tmp_mask_factory("mask.png", size=(8, 8))
        ds = ConcreteSegmentationDataset([(img, mask)])
        assert len(ds) == 1

    def test_empty_samples(self):
        ds = ConcreteSegmentationDataset([])
        assert len(ds) == 0


class TestGetItem:
    def test_returns_image_and_mask_tensors(self, tmp_image_factory, tmp_mask_factory):
        img = tmp_image_factory("img.png", size=(8, 8))
        mask = tmp_mask_factory("mask.png", size=(8, 8))
        ds = ConcreteSegmentationDataset([(img, mask)])
        image_out, mask_out = ds[0]
        assert isinstance(image_out, torch.Tensor)
        assert isinstance(mask_out, torch.Tensor)

    def test_image_tensor_shape_is_chw(self, tmp_image_factory, tmp_mask_factory):
        w, h = 16, 12
        img = tmp_image_factory("img.png", size=(w, h))
        mask = tmp_mask_factory("mask.png", size=(w, h))
        ds = ConcreteSegmentationDataset([(img, mask)])
        image_out, _ = ds[0]
        assert image_out.shape == (3, h, w)  # C, H, W

    def test_mask_tensor_has_channel_dim(self, tmp_image_factory, tmp_mask_factory):
        w, h = 8, 8
        img = tmp_image_factory("img.png", size=(w, h))
        mask = tmp_mask_factory("mask.png", size=(w, h))
        ds = ConcreteSegmentationDataset([(img, mask)])
        _, mask_out = ds[0]
        assert mask_out.shape[0] == 1  # unsqueeze(0) adds channel dim

    def test_return_path_gives_3_tuple(self, tmp_image_factory, tmp_mask_factory):
        img = tmp_image_factory("img.png", size=(8, 8))
        mask = tmp_mask_factory("mask.png", size=(8, 8))
        ds = ConcreteSegmentationDataset([(img, mask)], return_path=True)
        result = ds[0]
        assert len(result) == 3
        assert str(img) in result[2]

    def test_binary_mask_mode(self, tmp_image_factory, tmp_mask_factory):
        img = tmp_image_factory("img.png", size=(8, 8))
        mask = tmp_mask_factory("mask.png", size=(8, 8), value=200)
        ds = ConcreteSegmentationDataset([(img, mask)], is_binary=True)
        _, mask_out = ds[0]
        unique = torch.unique(mask_out)
        assert all(v in [0, 1] for v in unique.tolist())


class TestExtensions:
    def test_default_extensions_include_common_formats(self):
        ds = ConcreteSegmentationDataset([])
        assert ".png" in ds.extensions
        assert ".jpg" in ds.extensions

    def test_custom_extensions(self):
        ds = ConcreteSegmentationDataset([], extensions=[".TIFF"])
        assert ds.extensions == {".tiff"}
