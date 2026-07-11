"""Tests for ImageCSVXLSXDataset — loads images from CSV metadata files."""

import pytest

from torchdatasets.image.classification.from_csv import ImageCSVXLSXDataset


class TestSingleLabel:
    def test_length(self, tmp_path, tmp_image_factory, tmp_csv_factory):
        p1 = tmp_image_factory("img_0.png")
        p2 = tmp_image_factory("img_1.png")
        csv_path = tmp_csv_factory(
            "data.csv",
            [
                {"path": str(p1), "label": "cat"},
                {"path": str(p2), "label": "dog"},
            ],
        )
        ds = ImageCSVXLSXDataset(file_path=csv_path)
        assert len(ds) == 2

    def test_getitem(self, tmp_path, tmp_image_factory, tmp_csv_factory):
        p1 = tmp_image_factory("img.png")
        csv_path = tmp_csv_factory(
            "data.csv",
            [{"path": str(p1), "label": "cat"}],
        )
        ds = ImageCSVXLSXDataset(file_path=csv_path)
        img, label = ds[0]
        assert isinstance(label, int)

    def test_class_mapping(self, tmp_path, tmp_image_factory, tmp_csv_factory):
        p1 = tmp_image_factory("a.png")
        p2 = tmp_image_factory("b.png")
        csv_path = tmp_csv_factory(
            "data.csv",
            [
                {"path": str(p1), "label": "cat"},
                {"path": str(p2), "label": "dog"},
            ],
        )
        ds = ImageCSVXLSXDataset(file_path=csv_path)
        assert "cat" in ds.class_to_idx
        assert "dog" in ds.class_to_idx


class TestMultiLabel:
    def test_multi_label_with_separator(self, tmp_path, tmp_image_factory, tmp_csv_factory):
        p1 = tmp_image_factory("img.png")
        csv_path = tmp_csv_factory(
            "data.csv",
            [{"path": str(p1), "label": "cat,dog"}],
        )
        ds = ImageCSVXLSXDataset(file_path=csv_path, label_sep=",")
        _, label = ds[0]
        assert isinstance(label, list)
        assert len(label) == 2


class TestReturnPath:
    def test_return_path_flag(self, tmp_path, tmp_image_factory, tmp_csv_factory):
        p1 = tmp_image_factory("img.png")
        csv_path = tmp_csv_factory(
            "data.csv",
            [{"path": str(p1), "label": "cat"}],
        )
        ds = ImageCSVXLSXDataset(file_path=csv_path, return_path=True)
        result = ds[0]
        assert len(result) == 3


class TestCustomColumns:
    def test_custom_column_names(self, tmp_path, tmp_image_factory, tmp_csv_factory):
        p1 = tmp_image_factory("img.png")
        csv_path = tmp_csv_factory(
            "data.csv",
            [{"image_path": str(p1), "class": "cat"}],
        )
        ds = ImageCSVXLSXDataset(
            file_path=csv_path, path_col="image_path", label_col="class"
        )
        assert len(ds) == 1


class TestFilteringAndErrors:
    def test_missing_images_skipped(self, tmp_path, tmp_image_factory, tmp_csv_factory):
        p1 = tmp_image_factory("exists.png")
        csv_path = tmp_csv_factory(
            "data.csv",
            [
                {"path": str(p1), "label": "cat"},
                {"path": str(tmp_path / "missing.png"), "label": "dog"},
            ],
        )
        ds = ImageCSVXLSXDataset(file_path=csv_path)
        assert len(ds) == 1

    def test_extension_filter(self, tmp_path, tmp_image_factory, tmp_csv_factory):
        p_png = tmp_image_factory("a.png")
        p_jpg = tmp_image_factory("b.jpg")
        csv_path = tmp_csv_factory(
            "data.csv",
            [
                {"path": str(p_png), "label": "cat"},
                {"path": str(p_jpg), "label": "dog"},
            ],
        )
        ds = ImageCSVXLSXDataset(file_path=csv_path, extensions=[".png"])
        assert len(ds) == 1

    def test_no_valid_entries_raises(self, tmp_path, tmp_csv_factory):
        csv_path = tmp_csv_factory(
            "data.csv",
            [{"path": str(tmp_path / "missing.png"), "label": "cat"}],
        )
        with pytest.raises(AssertionError, match="No valid image entries"):
            ImageCSVXLSXDataset(file_path=csv_path)

    def test_missing_columns_raises(self, tmp_path, tmp_csv_factory):
        csv_path = tmp_csv_factory("data.csv", [{"x": 1, "y": 2}])
        with pytest.raises(AssertionError, match="path.*label"):
            ImageCSVXLSXDataset(file_path=csv_path)
