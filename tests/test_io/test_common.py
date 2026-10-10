"""Tests for torchdatasets._internal.io.common — CSV/Excel loading utility."""

import pytest
import pandas as pd
from pathlib import Path

import numpy as np

from torchdatasets._internal.io.common import cell_text, folder_of, is_blank, load_csv_or_excel, resolve_path


class TestLoadCSV:
    """Tests for loading CSV files."""

    def test_loads_csv_returns_dataframe(self, tmp_csv_factory):
        path = tmp_csv_factory("data.csv", [{"a": 1, "b": 2}, {"a": 3, "b": 4}])
        df = load_csv_or_excel(path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["a", "b"]

    def test_csv_preserves_values(self, tmp_csv_factory):
        rows = [{"x": 10, "y": "hello"}, {"x": 20, "y": "world"}]
        path = tmp_csv_factory("data.csv", rows)
        df = load_csv_or_excel(path)
        assert df["x"].tolist() == [10, 20]
        assert df["y"].tolist() == ["hello", "world"]


class TestLoadExcel:
    """Tests for loading Excel files."""

    def test_loads_xlsx_returns_dataframe(self, tmp_path):
        path = tmp_path / "data.xlsx"
        df_expected = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        df_expected.to_excel(path, index=False)

        df = load_csv_or_excel(path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["col1", "col2"]

    def test_xlsx_preserves_values(self, tmp_path):
        path = tmp_path / "data.xlsx"
        df_expected = pd.DataFrame({"a": [10, 20], "b": ["x", "y"]})
        df_expected.to_excel(path, index=False)

        df = load_csv_or_excel(path)
        assert df["a"].tolist() == [10, 20]
        assert df["b"].tolist() == ["x", "y"]


class TestLoadErrors:
    """Tests for error handling."""

    def test_unsupported_extension_raises_value_error(self, tmp_path):
        path = tmp_path / "data.json"
        path.write_text("{}")
        with pytest.raises(ValueError, match="Only .csv and .xlsx/.xls"):
            load_csv_or_excel(path)

    def test_missing_file_raises(self, tmp_path):
        path = tmp_path / "nonexistent.csv"
        with pytest.raises(Exception):
            load_csv_or_excel(path)


class TestCellHelpers:
    """Tests for is_blank and cell_text (reading path and label cells)."""

    @pytest.mark.parametrize("cell", [None, float("nan"), np.nan, pd.NA, pd.NaT, "", "   "])
    def test_blank_cells(self, cell):
        assert is_blank(cell)

    @pytest.mark.parametrize("cell", ["a", "0", 0, 0.0, False])
    def test_cells_with_a_value(self, cell):
        assert not is_blank(cell)

    @pytest.mark.parametrize(
        "cell, text",
        [(3.0, "3"), (np.float64(10.0), "10"), (2.5, "2.5"), (np.int64(4), "4"), (" cat ", "cat")],
    )
    def test_cell_text(self, cell, text):
        assert cell_text(cell) == text


class TestResolvePath:
    """CSV path cells: absolute as written, relative from the CSV file's folder."""

    def test_relative_path_starts_from_the_base_folder(self, tmp_path):
        assert resolve_path("imgs/a.png", tmp_path) == tmp_path / "imgs" / "a.png"

    def test_parent_segments_are_normalised(self, tmp_path):
        resolved = resolve_path("../imgs/a.png", tmp_path / "csvs")
        assert resolved == tmp_path / "imgs" / "a.png"
        assert ".." not in str(resolved)

    def test_absolute_path_is_kept(self, tmp_path):
        absolute = tmp_path / "x" / "a.png"
        assert str(resolve_path(str(absolute), tmp_path / "other")) == str(absolute)

    def test_folder_of_a_relative_file_is_absolute(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert folder_of(Path("data") / "labels.csv") == tmp_path / "data"
        assert folder_of(Path("labels.csv")) == tmp_path
