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


class TestMaskExtensions:
    def test_jpg_image_with_png_mask(self, tmp_path, tmp_image_factory, tmp_mask_factory):
        """Regression: masks had to share the image's extension, so a.jpg + a_mask.png never paired."""
        tmp_image_factory("seg/a.jpg")
        tmp_mask_factory("seg/a_mask.png")
        tmp_image_factory("seg/b.png")
        tmp_mask_factory("seg/b_mask.png")
        ds = ImageSegSingleDirDataset(src_dir=tmp_path / "seg", suffix="_mask")
        assert [(img.name, mask.name) for img, mask in ds.samples] == [
            ("a.jpg", "a_mask.png"),
            ("b.png", "b_mask.png"),
        ]

    def test_prefers_mask_with_same_extension(self, tmp_path, tmp_image_factory, tmp_mask_factory):
        tmp_image_factory("seg/a.png")
        tmp_mask_factory("seg/a_mask.bmp")
        tmp_mask_factory("seg/a_mask.png")
        ds = ImageSegSingleDirDataset(src_dir=tmp_path / "seg", suffix="_mask")
        assert ds.samples[0][1].name == "a_mask.png"

    def test_mask_extensions_order_and_filter(self, tmp_path, tmp_image_factory, tmp_mask_factory):
        tmp_image_factory("seg/a.jpg")
        tmp_mask_factory("seg/a_mask.png")
        tmp_mask_factory("seg/a_mask.bmp")
        ds = ImageSegSingleDirDataset(src_dir=tmp_path / "seg", suffix="_mask", mask_extensions=[".BMP", ".png"])
        assert ds.samples[0][1].name == "a_mask.bmp"

        with pytest.raises(AssertionError, match="No valid samples"):
            ImageSegSingleDirDataset(src_dir=tmp_path / "seg", suffix="_mask", mask_extensions=[".tiff"])

    def test_mixed_extension_pair_loads(self, tmp_path, tmp_image_factory, tmp_mask_factory):
        tmp_image_factory("seg/a.jpg", size=(8, 6))
        tmp_mask_factory("seg/a_mask.png", size=(6, 8))  # masks take (H, W)
        image, mask = ImageSegSingleDirDataset(src_dir=tmp_path / "seg", suffix="_mask")[0]
        assert image.shape == (3, 6, 8)
        assert mask.shape == (1, 6, 8)
