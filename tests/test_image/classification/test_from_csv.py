"""Tests for ImageCSVDataset — loads images from CSV metadata files."""

import os
import warnings
from pathlib import Path

import pytest

from torchdatasets.image.classification.from_csv import ImageCSVDataset


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
        ds = ImageCSVDataset(file_path=csv_path)
        assert len(ds) == 2

    def test_getitem(self, tmp_path, tmp_image_factory, tmp_csv_factory):
        p1 = tmp_image_factory("img.png")
        csv_path = tmp_csv_factory(
            "data.csv",
            [{"path": str(p1), "label": "cat"}],
        )
        ds = ImageCSVDataset(file_path=csv_path)
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
        ds = ImageCSVDataset(file_path=csv_path)
        assert "cat" in ds.class_to_idx
        assert "dog" in ds.class_to_idx


class TestMultiLabel:
    def test_multi_label_with_separator(self, tmp_path, tmp_image_factory, tmp_csv_factory):
        p1 = tmp_image_factory("img.png")
        csv_path = tmp_csv_factory(
            "data.csv",
            [{"path": str(p1), "label": "cat,dog"}],
        )
        ds = ImageCSVDataset(file_path=csv_path, label_sep=",")
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
        ds = ImageCSVDataset(file_path=csv_path, return_path=True)
        result = ds[0]
        assert len(result) == 3


class TestCustomColumns:
    def test_custom_column_names(self, tmp_path, tmp_image_factory, tmp_csv_factory):
        p1 = tmp_image_factory("img.png")
        csv_path = tmp_csv_factory(
            "data.csv",
            [{"image_path": str(p1), "class": "cat"}],
        )
        ds = ImageCSVDataset(
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
        with pytest.warns(UserWarning, match=r"Skipped 1 of 2 row\(s\).*missing\.png"):
            ds = ImageCSVDataset(file_path=csv_path)
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
        with pytest.warns(UserWarning, match=r"Skipped 1 of 2 row\(s\).*\['\.png'\].*b\.jpg"):
            ds = ImageCSVDataset(file_path=csv_path, extensions=[".png"])
        assert len(ds) == 1

    def test_no_valid_entries_raises(self, tmp_path, tmp_csv_factory):
        csv_path = tmp_csv_factory(
            "data.csv",
            [{"path": str(tmp_path / "missing.png"), "label": "cat"}],
        )
        with pytest.raises(FileNotFoundError, match="No valid entries"):
            ImageCSVDataset(file_path=csv_path)

    def test_missing_columns_raises(self, tmp_path, tmp_csv_factory):
        csv_path = tmp_csv_factory("data.csv", [{"x": 1, "y": 2}])
        with pytest.raises(ValueError, match="path.*label"):
            ImageCSVDataset(file_path=csv_path)

    def test_missing_columns_error_lists_found_columns(self, tmp_path, tmp_csv_factory):
        csv_path = tmp_csv_factory("data.csv", [{"path": "a.png", "class": "cat"}])
        with pytest.raises(ValueError, match=r"missing: \['label'\].*found: \['path', 'class'\]"):
            ImageCSVDataset(file_path=csv_path)


class TestSampleAlignment:
    def test_each_row_keeps_its_own_path_and_label(self, tmp_image_factory, tmp_csv_factory):
        """Regression: every sample used to get the *last* row's image path."""
        rows = [
            ("red.png", (255, 0, 0), "a"),
            ("green.png", (0, 255, 0), "b"),
            ("blue.png", (0, 0, 255), "c"),
        ]
        paths = [tmp_image_factory(name, color=color) for name, color, _ in rows]
        csv_path = tmp_csv_factory(
            "data.csv",
            [{"path": str(p), "label": lbl} for p, (_, _, lbl) in zip(paths, rows)],
        )
        ds = ImageCSVDataset(file_path=csv_path, return_path=True)

        assert [Path(path) for path, _ in ds.samples] == paths
        for i, (_, color, lbl) in enumerate(rows):
            img, label, path = ds[i]
            assert Path(path) == paths[i]
            assert label == ds.class_to_idx[lbl]
            assert img[:, 0, 0].tolist() == list(color)


class TestExcelFile:
    def test_loads_xlsx(self, tmp_path, tmp_image_factory):
        pd = pytest.importorskip("pandas")
        p1, p2 = tmp_image_factory("a.png"), tmp_image_factory("b.png")
        xlsx_path = tmp_path / "data.xlsx"
        pd.DataFrame({"path": [str(p1), str(p2)], "label": ["cat", "dog"]}).to_excel(xlsx_path, index=False)
        ds = ImageCSVDataset(file_path=xlsx_path)
        assert len(ds) == 2
        assert ds.class_to_idx == {"cat": 0, "dog": 1}


class TestSkippedRows:
    def test_blank_cells_are_skipped_with_a_warning(self, tmp_image_factory, tmp_csv_factory):
        p1 = tmp_image_factory("a.png")
        p2 = tmp_image_factory("b.png")
        csv_path = tmp_csv_factory(
            "data.csv",
            [
                {"path": str(p1), "label": "cat"},
                {"path": str(p2), "label": None},  # no label
                {"path": None, "label": "dog"},  # no path
                {"path": "   ", "label": "dog"},  # only spaces
            ],
        )
        with pytest.warns(UserWarning, match=r"Skipped 3 of 4 row\(s\)"):
            ds = ImageCSVDataset(csv_path)
        assert [path for path, _ in ds.samples] == [p1]
        assert ds.class_to_idx == {"cat": 0}

    def test_warning_points_at_the_callers_line(self, tmp_path, tmp_image_factory, tmp_csv_factory):
        p1 = tmp_image_factory("a.png")
        csv_path = tmp_csv_factory(
            "data.csv",
            [{"path": str(p1), "label": "cat"}, {"path": str(tmp_path / "gone.png"), "label": "dog"}],
        )
        with pytest.warns(UserWarning) as record:
            ImageCSVDataset(csv_path)
        assert record[0].filename == __file__

    def test_no_warning_when_every_row_is_used(self, tmp_image_factory, tmp_csv_factory):
        p1 = tmp_image_factory("a.png")
        csv_path = tmp_csv_factory("data.csv", [{"path": str(p1), "label": "cat"}])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ImageCSVDataset(csv_path)

    def test_whole_number_labels_keep_their_text(self, tmp_image_factory, tmp_csv_factory):
        """A blank cell makes pandas read a number column as floats; label 3 must stay "3", not "3.0"."""
        p1 = tmp_image_factory("a.png")
        p2 = tmp_image_factory("b.png")
        p3 = tmp_image_factory("c.png")
        csv_path = tmp_csv_factory(
            "data.csv",
            [{"path": str(p1), "label": 3}, {"path": str(p2), "label": 10}, {"path": str(p3), "label": None}],
        )
        with pytest.warns(UserWarning, match="Skipped 1 of 3"):
            ds = ImageCSVDataset(csv_path)
        assert ds.class_to_idx == {"10": 0, "3": 1}

    def test_labels_are_stripped(self, tmp_image_factory, tmp_csv_factory):
        p1 = tmp_image_factory("a.png")
        p2 = tmp_image_factory("b.png")
        csv_path = tmp_csv_factory(
            "data.csv", [{"path": str(p1), "label": " cat "}, {"path": str(p2), "label": "cat"}]
        )
        ds = ImageCSVDataset(csv_path)
        assert ds.class_to_idx == {"cat": 0}
        assert ds.targets == [0, 0]


class TestMultiLabelCells:
    def test_empty_parts_are_dropped(self, tmp_image_factory, tmp_csv_factory):
        p1 = tmp_image_factory("a.png")
        p2 = tmp_image_factory("b.png")
        csv_path = tmp_csv_factory(
            "data.csv", [{"path": str(p1), "label": "cat;"}, {"path": str(p2), "label": "cat;;dog"}]
        )
        ds = ImageCSVDataset(csv_path, label_sep=";")
        assert ds.class_to_idx == {"cat": 0, "dog": 1}

    def test_row_with_only_separators_is_skipped(self, tmp_image_factory, tmp_csv_factory):
        p1 = tmp_image_factory("a.png")
        p2 = tmp_image_factory("b.png")
        csv_path = tmp_csv_factory(
            "data.csv", [{"path": str(p1), "label": "cat"}, {"path": str(p2), "label": " ; "}]
        )
        with pytest.warns(UserWarning, match="Skipped 1 of 2"):
            ds = ImageCSVDataset(csv_path, label_sep=";")
        assert len(ds) == 1


class TestLabelsAsText:
    LABELS = ["None", "NA", "007", "7", "1.50", "nan"]

    @pytest.mark.parametrize("suffix", [".csv", ".xlsx"])
    def test_labels_are_kept_exactly_as_written(self, tmp_path, tmp_image_factory, suffix):
        """No type guessing ("007" != "7") and no missing-value words ("NA" is a label)."""
        import pandas as pd

        paths = [str(tmp_image_factory(f"{i}.png")) for i in range(len(self.LABELS))]
        file_path = tmp_path / f"labels{suffix}"
        frame = pd.DataFrame({"path": paths, "label": self.LABELS})
        if suffix == ".csv":
            frame.to_csv(file_path, index=False)
        else:
            frame.to_excel(file_path, index=False)
        ds = ImageCSVDataset(file_path)
        assert sorted(ds.class_to_idx) == sorted(self.LABELS)
        assert len(ds) == len(self.LABELS)

    def test_whole_numbers_in_excel_stay_whole(self, tmp_path, tmp_image_factory):
        import pandas as pd

        paths = [str(tmp_image_factory(f"{i}.png")) for i in range(3)]
        pd.DataFrame({"path": paths, "label": [3, 10, None]}).to_excel(tmp_path / "n.xlsx", index=False)
        with pytest.warns(UserWarning, match="Skipped 1 of 3"):
            ds = ImageCSVDataset(tmp_path / "n.xlsx")
        assert ds.class_to_idx == {"10": 0, "3": 1}

    def test_error_when_no_row_is_usable_names_the_rows(self, tmp_path, tmp_csv_factory):
        csv_path = tmp_csv_factory("data.csv", [{"path": "gone.png", "label": "cat"}, {"path": "", "label": "dog"}])
        with pytest.raises(FileNotFoundError, match=r"none of its 2 row\(s\) can be used.*gone\.png"):
            ImageCSVDataset(csv_path)


class TestPaths:
    """Paths are absolute, or relative to the CSV file's folder (never the working directory)."""

    def test_relative_paths_start_from_the_csv_folder(self, tmp_path, tmp_image_factory, monkeypatch):
        img = tmp_image_factory("data/imgs/a.png")
        (tmp_path / "data" / "labels.csv").write_text("path,label\nimgs/a.png,cat\n")
        (tmp_path / "elsewhere").mkdir()
        monkeypatch.chdir(tmp_path / "elsewhere")  # the working directory must not matter
        ds = ImageCSVDataset(tmp_path / "data" / "labels.csv", return_path=True)
        assert ds.samples[0][0] == img
        assert ds[0][2] == str(img)

    def test_csv_given_by_a_relative_path(self, tmp_path, tmp_image_factory, monkeypatch):
        img = tmp_image_factory("data/imgs/a.png")
        (tmp_path / "data" / "labels.csv").write_text("path,label\nimgs/a.png,cat\n")
        monkeypatch.chdir(tmp_path)
        ds = ImageCSVDataset(Path("data") / "labels.csv")
        assert ds.samples[0][0].is_absolute()
        assert ds.samples[0][0] == img

    def test_parent_folder_segments_are_normalised(self, tmp_path, tmp_image_factory):
        img = tmp_image_factory("imgs/a.png")
        (tmp_path / "csvs").mkdir()
        (tmp_path / "csvs" / "labels.csv").write_text("path,label\n../imgs/a.png,cat\n")
        ds = ImageCSVDataset(tmp_path / "csvs" / "labels.csv", return_path=True)
        assert ds[0][2] == str(img)  # no ".." left

    def test_absolute_paths_are_used_as_written(self, tmp_image_factory, tmp_csv_factory):
        img = tmp_image_factory("somewhere/else/a.png")
        csv_path = tmp_csv_factory("csvs/labels.csv", [{"path": str(img), "label": "cat"}])
        ds = ImageCSVDataset(csv_path, return_path=True)
        assert ds[0][2] == str(img)

    def test_symlinks_are_not_followed(self, tmp_path, tmp_image_factory):
        target = tmp_image_factory("pool/a.png")
        link = tmp_path / "train" / "a.png"
        link.parent.mkdir()
        try:
            os.symlink(target, link)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks are not available here")
        (tmp_path / "labels.csv").write_text("path,label\ntrain/a.png,cat\n")
        ds = ImageCSVDataset(tmp_path / "labels.csv", return_path=True)
        assert ds[0][2] == str(link)

    def test_warning_names_each_row_and_where_it_looked(self, tmp_path, tmp_image_factory):
        tmp_image_factory("data/a.png")
        (tmp_path / "data" / "labels.csv").write_text("path,label\na.png,cat\ngone.png,dog\n,bird\n")
        with pytest.warns(UserWarning) as record:
            ImageCSVDataset(tmp_path / "data" / "labels.csv")
        message = str(record[0].message)
        assert "Skipped 2 of 3 row(s)" in message
        assert f"row 3: {tmp_path / 'data' / 'gone.png'}" in message
        assert "row 4: empty cell" in message
        assert f"relative paths start from the file's folder '{tmp_path / 'data'}'" in message
