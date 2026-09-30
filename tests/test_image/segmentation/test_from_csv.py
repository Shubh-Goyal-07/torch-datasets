"""Tests for ImageSegCSVXLSXDataset — loads segmentation pairs from CSV."""

import torch
import pytest

from torchdatasets.image.segmentation.from_csv import ImageSegCSVXLSXDataset


class TestBasicLoading:
    def test_length(self, tmp_path, tmp_image_factory, tmp_mask_factory, tmp_csv_factory):
        img = tmp_image_factory("images/img_0.png", size=(8, 8))
        mask = tmp_mask_factory("masks/mask_0.png", size=(8, 8))
        csv_path = tmp_csv_factory(
            "data.csv",
            [{"path": str(img.relative_to(tmp_path)), "label": str(mask.relative_to(tmp_path))}],
        )
        ds = ImageSegCSVXLSXDataset(file_path=csv_path)
        assert len(ds) == 1

    def test_getitem_returns_tensors(self, tmp_path, tmp_image_factory, tmp_mask_factory, tmp_csv_factory):
        img = tmp_image_factory("images/img_0.png", size=(8, 8))
        mask = tmp_mask_factory("masks/mask_0.png", size=(8, 8))
        csv_path = tmp_csv_factory(
            "data.csv",
            [{"path": str(img.relative_to(tmp_path)), "label": str(mask.relative_to(tmp_path))}],
        )
        ds = ImageSegCSVXLSXDataset(file_path=csv_path)
        image_out, mask_out = ds[0]
        assert isinstance(image_out, torch.Tensor)
        assert isinstance(mask_out, torch.Tensor)


class TestCustomColumns:
    def test_custom_col_names(self, tmp_path, tmp_image_factory, tmp_mask_factory, tmp_csv_factory):
        img = tmp_image_factory("images/img.png", size=(8, 8))
        mask = tmp_mask_factory("masks/mask.png", size=(8, 8))
        csv_path = tmp_csv_factory(
            "data.csv",
            [{"img": str(img.relative_to(tmp_path)), "msk": str(mask.relative_to(tmp_path))}],
        )
        ds = ImageSegCSVXLSXDataset(file_path=csv_path, image_col="img", mask_col="msk")
        assert len(ds) == 1


class TestReturnPath:
    def test_return_path_flag(self, tmp_path, tmp_image_factory, tmp_mask_factory, tmp_csv_factory):
        img = tmp_image_factory("images/img.png", size=(8, 8))
        mask = tmp_mask_factory("masks/mask.png", size=(8, 8))
        csv_path = tmp_csv_factory(
            "data.csv",
            [{"path": str(img.relative_to(tmp_path)), "label": str(mask.relative_to(tmp_path))}],
        )
        ds = ImageSegCSVXLSXDataset(file_path=csv_path, return_path=True)
        result = ds[0]
        assert len(result) == 3


class TestErrors:
    def test_no_valid_entries_raises(self, tmp_path, tmp_csv_factory):
        csv_path = tmp_csv_factory(
            "data.csv",
            [{"path": "nonexistent.png", "label": "nonexistent_mask.png"}],
        )
        with pytest.raises(AssertionError, match="No valid entries"):
            ImageSegCSVXLSXDataset(file_path=csv_path)

    def test_missing_columns_raises(self, tmp_path, tmp_csv_factory):
        csv_path = tmp_csv_factory("data.csv", [{"x": 1, "y": 2}])
        with pytest.raises(AssertionError, match="path.*label"):
            ImageSegCSVXLSXDataset(file_path=csv_path)


class TestMaskValidation:
    def test_rows_with_missing_mask_are_skipped(
        self, tmp_path, tmp_image_factory, tmp_mask_factory, tmp_csv_factory
    ):
        """Regression: the mask path was never checked, so broken rows were kept."""
        tmp_image_factory("images/ok.png")
        mask_ok = tmp_mask_factory("masks/ok.png")
        tmp_image_factory("images/orphan.png")
        csv_path = tmp_csv_factory(
            "data.csv",
            [
                {"path": "images/ok.png", "label": "masks/ok.png"},
                {"path": "images/orphan.png", "label": "masks/missing.png"},
            ],
        )
        ds = ImageSegCSVXLSXDataset(file_path=csv_path)
        assert len(ds) == 1
        assert ds.samples[0][1] == mask_ok.resolve()

    def test_mask_with_invalid_extension_is_skipped(
        self, tmp_path, tmp_image_factory, tmp_mask_factory, tmp_csv_factory
    ):
        tmp_image_factory("images/a.png")
        tmp_mask_factory("masks/a.png")
        tmp_image_factory("images/b.png")
        (tmp_path / "masks" / "b.txt").write_text("not a mask")
        csv_path = tmp_csv_factory(
            "data.csv",
            [
                {"path": "images/a.png", "label": "masks/a.png"},
                {"path": "images/b.png", "label": "masks/b.txt"},
            ],
        )
        ds = ImageSegCSVXLSXDataset(file_path=csv_path)
        assert len(ds) == 1

    def test_all_masks_missing_raises(self, tmp_path, tmp_image_factory, tmp_csv_factory):
        tmp_image_factory("images/a.png")
        csv_path = tmp_csv_factory(
            "data.csv", [{"path": "images/a.png", "label": "masks/missing.png"}]
        )
        with pytest.raises(AssertionError, match="No valid entries"):
            ImageSegCSVXLSXDataset(file_path=csv_path)
