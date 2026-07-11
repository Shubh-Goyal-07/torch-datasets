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
