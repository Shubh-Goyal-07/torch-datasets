"""Tests for ImageSegCSVDataset — loads segmentation pairs from CSV."""

import torch
import pytest

from torchdatasets.image.segmentation.from_csv import ImageSegCSVDataset


class TestBasicLoading:
    def test_length(self, tmp_path, tmp_image_factory, tmp_mask_factory, tmp_csv_factory):
        img = tmp_image_factory("images/img_0.png", size=(8, 8))
        mask = tmp_mask_factory("masks/mask_0.png", size=(8, 8))
        csv_path = tmp_csv_factory(
            "data.csv",
            [{"image": str(img.relative_to(tmp_path)), "mask": str(mask.relative_to(tmp_path))}],
        )
        ds = ImageSegCSVDataset(file_path=csv_path)
        assert len(ds) == 1

    def test_getitem_returns_tensors(self, tmp_path, tmp_image_factory, tmp_mask_factory, tmp_csv_factory):
        img = tmp_image_factory("images/img_0.png", size=(8, 8))
        mask = tmp_mask_factory("masks/mask_0.png", size=(8, 8))
        csv_path = tmp_csv_factory(
            "data.csv",
            [{"image": str(img.relative_to(tmp_path)), "mask": str(mask.relative_to(tmp_path))}],
        )
        ds = ImageSegCSVDataset(file_path=csv_path)
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
        ds = ImageSegCSVDataset(file_path=csv_path, image_col="img", mask_col="msk")
        assert len(ds) == 1

    def test_default_columns_are_image_and_mask(self):
        import inspect

        params = inspect.signature(ImageSegCSVDataset.__init__).parameters
        assert params["image_col"].default == "image"
        assert params["mask_col"].default == "mask"


class TestReturnPath:
    def test_return_path_flag(self, tmp_path, tmp_image_factory, tmp_mask_factory, tmp_csv_factory):
        img = tmp_image_factory("images/img.png", size=(8, 8))
        mask = tmp_mask_factory("masks/mask.png", size=(8, 8))
        csv_path = tmp_csv_factory(
            "data.csv",
            [{"image": str(img.relative_to(tmp_path)), "mask": str(mask.relative_to(tmp_path))}],
        )
        ds = ImageSegCSVDataset(file_path=csv_path, return_path=True)
        result = ds[0]
        assert len(result) == 3


class TestErrors:
    def test_no_valid_entries_raises(self, tmp_path, tmp_csv_factory):
        csv_path = tmp_csv_factory(
            "data.csv",
            [{"image": "nonexistent.png", "mask": "nonexistent_mask.png"}],
        )
        with pytest.raises(FileNotFoundError, match="No valid entries"):
            ImageSegCSVDataset(file_path=csv_path)

    def test_missing_columns_raises(self, tmp_path, tmp_csv_factory):
        csv_path = tmp_csv_factory("data.csv", [{"x": 1, "y": 2}])
        with pytest.raises(ValueError, match="image.*mask"):
            ImageSegCSVDataset(file_path=csv_path)

    def test_old_default_column_names_raise_clear_error(self, tmp_path, tmp_csv_factory):
        """A file written for the old defaults (path/label) names the columns it needs."""
        csv_path = tmp_csv_factory("data.csv", [{"path": "a.png", "label": "a_mask.png"}])
        with pytest.raises(ValueError, match=r"\['image', 'mask'\]"):
            ImageSegCSVDataset(file_path=csv_path)


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
                {"image": "images/ok.png", "mask": "masks/ok.png"},
                {"image": "images/orphan.png", "mask": "masks/missing.png"},
            ],
        )
        with pytest.warns(UserWarning, match=r"Skipped 1 of 2 row\(s\).*images/orphan\.png"):
            ds = ImageSegCSVDataset(file_path=csv_path)
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
                {"image": "images/a.png", "mask": "masks/a.png"},
                {"image": "images/b.png", "mask": "masks/b.txt"},
            ],
        )
        with pytest.warns(UserWarning, match=r"Skipped 1 of 2 row\(s\)"):
            ds = ImageSegCSVDataset(file_path=csv_path)
        assert len(ds) == 1

    def test_all_masks_missing_raises(self, tmp_path, tmp_image_factory, tmp_csv_factory):
        tmp_image_factory("images/a.png")
        csv_path = tmp_csv_factory(
            "data.csv", [{"image": "images/a.png", "mask": "masks/missing.png"}]
        )
        with pytest.raises(FileNotFoundError, match="No valid entries"):
            ImageSegCSVDataset(file_path=csv_path)


class TestExcelFile:
    def test_loads_xlsx(self, tmp_path, tmp_image_factory, tmp_mask_factory):
        pd = pytest.importorskip("pandas")
        tmp_image_factory("images/a.png")
        tmp_mask_factory("masks/a.png")
        xlsx_path = tmp_path / "data.xlsx"
        pd.DataFrame({"image": ["images/a.png"], "mask": ["masks/a.png"]}).to_excel(xlsx_path, index=False)
        ds = ImageSegCSVDataset(file_path=xlsx_path)
        assert len(ds) == 1


class TestSkippedRows:
    def test_blank_cells_are_skipped_with_a_warning(self, tmp_image_factory, tmp_mask_factory, tmp_csv_factory):
        tmp_image_factory("images/a.png")
        tmp_mask_factory("masks/a.png")
        tmp_image_factory("images/b.png")
        csv_path = tmp_csv_factory(
            "data.csv",
            [
                {"image": "images/a.png", "mask": "masks/a.png"},
                {"image": "images/b.png", "mask": None},
                {"image": None, "mask": "masks/a.png"},
            ],
        )
        with pytest.warns(UserWarning, match=r"Skipped 2 of 3 row\(s\)") as record:
            ds = ImageSegCSVDataset(csv_path)
        assert record[0].filename == __file__  # the warning points at the caller
        assert len(ds) == 1

    def test_paths_with_spaces_around_them_are_found(self, tmp_image_factory, tmp_mask_factory, tmp_csv_factory):
        tmp_image_factory("images/a.png")
        tmp_mask_factory("masks/a.png")
        csv_path = tmp_csv_factory("data.csv", [{"image": " images/a.png", "mask": "masks/a.png "}])
        ds = ImageSegCSVDataset(csv_path)
        assert len(ds) == 1
