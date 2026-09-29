"""Tests for ImageSegMultidirDataset — paired image/mask directories."""

import torch
import pytest

from torchdatasets.image.segmentation.from_multidir import ImageSegMultidirDataset


class TestBasicLoading:
    def test_length(self, image_seg_paired_dirs):
        img_dir, mask_dir, count = image_seg_paired_dirs
        ds = ImageSegMultidirDataset(image_dir=img_dir, mask_dir=mask_dir)
        assert len(ds) == count

    def test_getitem_returns_tensors(self, image_seg_paired_dirs):
        img_dir, mask_dir, _ = image_seg_paired_dirs
        ds = ImageSegMultidirDataset(image_dir=img_dir, mask_dir=mask_dir)
        image_out, mask_out = ds[0]
        assert isinstance(image_out, torch.Tensor)
        assert isinstance(mask_out, torch.Tensor)


class TestSuffix:
    def test_suffix_matching(self, tmp_path, tmp_image_factory, tmp_mask_factory):
        tmp_image_factory("imgs/photo.png", size=(8, 8))
        tmp_mask_factory("msks/photo_mask.png", size=(8, 8))
        ds = ImageSegMultidirDataset(
            image_dir=tmp_path / "imgs",
            mask_dir=tmp_path / "msks",
            suffix="_mask",
        )
        assert len(ds) == 1


class TestReturnPath:
    def test_return_path(self, image_seg_paired_dirs):
        img_dir, mask_dir, _ = image_seg_paired_dirs
        ds = ImageSegMultidirDataset(
            image_dir=img_dir, mask_dir=mask_dir, return_path=True
        )
        result = ds[0]
        assert len(result) == 3


class TestErrors:
    def test_no_matching_masks_raises(self, tmp_path, tmp_image_factory):
        tmp_image_factory("imgs/a.png")
        (tmp_path / "msks").mkdir()
        with pytest.raises(AssertionError, match="No valid samples"):
            ImageSegMultidirDataset(
                image_dir=tmp_path / "imgs",
                mask_dir=tmp_path / "msks",
            )
