"""Tests for torchdatasets._internal.io.common — CSV/Excel loading utility."""

import pytest
import pandas as pd
from pathlib import Path

from torchdatasets._internal.io.common import load_csv_or_excel


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
