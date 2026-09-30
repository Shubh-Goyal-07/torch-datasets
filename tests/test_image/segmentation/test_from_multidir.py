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


class TestMaskExtensions:
    def test_jpg_image_with_png_mask(self, tmp_path, tmp_image_factory, tmp_mask_factory):
        """Regression: masks had to share the image's extension, so x.jpg + x.png never paired."""
        tmp_image_factory("imgs/a.jpg")
        tmp_image_factory("imgs/b.jpg")
        tmp_mask_factory("msks/a.png")
        tmp_mask_factory("msks/b_mask.png")
        ds = ImageSegMultidirDataset(tmp_path / "imgs", tmp_path / "msks")
        assert [(img.name, mask.name) for img, mask in ds.samples] == [("a.jpg", "a.png")]

        ds = ImageSegMultidirDataset(tmp_path / "imgs", tmp_path / "msks", suffix="_mask")
        assert [(img.name, mask.name) for img, mask in ds.samples] == [("b.jpg", "b_mask.png")]

    def test_prefers_mask_with_same_extension(self, tmp_path, tmp_image_factory, tmp_mask_factory):
        tmp_image_factory("imgs/a.png")
        tmp_mask_factory("msks/a.bmp")
        tmp_mask_factory("msks/a.png")
        ds = ImageSegMultidirDataset(tmp_path / "imgs", tmp_path / "msks")
        assert ds.samples[0][1].name == "a.png"

    def test_mask_extensions_order_and_filter(self, tmp_path, tmp_image_factory, tmp_mask_factory):
        tmp_image_factory("imgs/a.jpg")
        tmp_mask_factory("msks/a.png")
        tmp_mask_factory("msks/a.bmp")
        ds = ImageSegMultidirDataset(tmp_path / "imgs", tmp_path / "msks", mask_extensions=[".BMP", ".png"])
        assert ds.samples[0][1].name == "a.bmp"

        with pytest.raises(AssertionError, match="No valid samples"):
            ImageSegMultidirDataset(tmp_path / "imgs", tmp_path / "msks", mask_extensions=[".tiff"])

    def test_uppercase_mask_extension(self, tmp_path, tmp_image_factory, tmp_mask_factory):
        tmp_image_factory("imgs/a.jpg")
        mask = tmp_mask_factory("msks/a.png")
        mask.rename(mask.with_suffix(".PNG"))
        ds = ImageSegMultidirDataset(tmp_path / "imgs", tmp_path / "msks")
        assert ds.samples[0][1].name == "a.PNG"

    def test_mixed_extension_pair_loads(self, tmp_path, tmp_image_factory, tmp_mask_factory):
        tmp_image_factory("imgs/a.jpg", size=(8, 6))
        tmp_mask_factory("msks/a.png", size=(6, 8))  # masks take (H, W)
        image, mask = ImageSegMultidirDataset(tmp_path / "imgs", tmp_path / "msks")[0]
        assert image.shape == (3, 6, 8)
        assert mask.shape == (1, 6, 8)
