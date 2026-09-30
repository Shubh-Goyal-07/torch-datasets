"""Tests for BaseImageSegmentationDataset contract."""

import torch
import numpy as np
import pytest
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.transforms import v2

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


def _write_pattern_pair(tmp_path, h=8, w=10):
    """Write an image and mask sharing the same asymmetric 0/255 pattern, so
    channel 0 of the image must equal the mask after any geometric transform."""
    pattern = np.zeros((h, w), dtype=np.uint8)
    pattern[: h // 2, : w // 3] = 255
    pattern[h - 1, w - 1] = 255
    img_path, mask_path = tmp_path / "img.png", tmp_path / "mask.png"
    Image.fromarray(np.stack([pattern] * 3, axis=-1)).save(img_path)
    Image.fromarray(pattern).save(mask_path)
    return img_path, mask_path


class TestTvTensorsAndTransforms:
    def test_outputs_are_tv_tensors(self, tmp_path):
        ds = ConcreteSegmentationDataset([_write_pattern_pair(tmp_path)])
        image, mask = ds[0]
        assert isinstance(image, tv_tensors.Image)
        assert isinstance(mask, tv_tensors.Mask)

    @pytest.mark.parametrize(
        "transform",
        [
            v2.RandomHorizontalFlip(p=1.0),
            v2.RandomVerticalFlip(p=1.0),
            v2.RandomCrop(size=(5, 6)),
            v2.RandomRotation(degrees=(90, 90)),
        ],
        ids=["hflip", "vflip", "crop", "rot90"],
    )
    def test_geometric_transform_keeps_mask_aligned(self, tmp_path, transform):
        """Regression: v2 transforms used to transform the image but not the mask."""
        torch.manual_seed(0)
        ds = ConcreteSegmentationDataset([_write_pattern_pair(tmp_path)], transform=transform)
        image, mask = ds[0]
        assert image.shape[1:] == mask.shape[1:]
        assert torch.equal(image[0], mask[0])

    def test_dataloader_batches(self, tmp_path):
        pair = _write_pattern_pair(tmp_path)
        ds = ConcreteSegmentationDataset([pair, pair])
        images, masks = next(iter(DataLoader(ds, batch_size=2)))
        assert images.shape == (2, 3, 8, 10)
        assert masks.shape == (2, 1, 8, 10)
