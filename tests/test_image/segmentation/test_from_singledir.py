"""Tests for ImageSegSingleDirDataset — image/mask pairs in one directory via suffix."""

import torch
import pytest

from torchdatasets.image.segmentation.from_singledir import ImageSegSingleDirDataset


class TestBasicLoading:
    def test_length(self, image_seg_single_dir):
        src_dir, suffix, count = image_seg_single_dir
        ds = ImageSegSingleDirDataset(src_dir=src_dir, suffix=suffix)
        assert len(ds) == count

    def test_getitem_returns_tensors(self, image_seg_single_dir):
        src_dir, suffix, _ = image_seg_single_dir
        ds = ImageSegSingleDirDataset(src_dir=src_dir, suffix=suffix)
        image_out, mask_out = ds[0]
        assert isinstance(image_out, torch.Tensor)
        assert isinstance(mask_out, torch.Tensor)


class TestSuffixFiltering:
    def test_masks_not_treated_as_images(self, image_seg_single_dir):
        """Mask files (containing the suffix in their stem) should be excluded
        from the image list, so the length equals the number of *image* files."""
        src_dir, suffix, count = image_seg_single_dir
        ds = ImageSegSingleDirDataset(src_dir=src_dir, suffix=suffix)
        assert len(ds) == count  # Only images, not masks


class TestReturnPath:
    def test_return_path(self, image_seg_single_dir):
        src_dir, suffix, _ = image_seg_single_dir
        ds = ImageSegSingleDirDataset(
            src_dir=src_dir, suffix=suffix, return_path=True
        )
        result = ds[0]
        assert len(result) == 3


class TestErrors:
    def test_no_masks_found_raises(self, tmp_path, tmp_image_factory):
        tmp_image_factory("dir/a.png")
        with pytest.raises(AssertionError, match="No valid samples"):
            ImageSegSingleDirDataset(
                src_dir=tmp_path / "dir", suffix="_mask"
            )


class TestSuffixMatching:
    def test_image_name_containing_suffix_is_kept(
        self, tmp_path, tmp_image_factory, tmp_mask_factory
    ):
        """Regression: any image whose name *contained* the suffix was dropped."""
        for name in ["img_mask_scan", "b"]:
            tmp_image_factory(f"seg/{name}.png")
            tmp_mask_factory(f"seg/{name}_mask.png")
        ds = ImageSegSingleDirDataset(src_dir=tmp_path / "seg", suffix="_mask")
        assert sorted(img.name for img, _ in ds.samples) == ["b.png", "img_mask_scan.png"]
        assert all(mask.stem.endswith("_mask") for _, mask in ds.samples)
