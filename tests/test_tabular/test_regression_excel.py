"""Tests for TabularRegressionFromExcel — end-to-end Excel loading for regression."""

import pandas as pd
import torch
import pytest

from torchdatasets.tabular.regression.from_excel import TabularRegressionFromExcel


def _make_excel(tmp_path, rows=None, filename="data.xlsx"):
    if rows is None:
        rows = [
            {"f1": 1.0, "f2": 2.0, "target": 10.5},
            {"f1": 3.0, "f2": 4.0, "target": 20.1},
            {"f1": 5.0, "f2": 6.0, "target": 30.7},
        ]
    path = tmp_path / filename
    pd.DataFrame(rows).to_excel(path, index=False)
    return path


class TestBasicLoading:
    def test_length(self, tmp_path):
        path = _make_excel(tmp_path)
        ds = TabularRegressionFromExcel(path, target_column="target")
        assert len(ds) == 3

    def test_getitem(self, tmp_path):
        path = _make_excel(tmp_path)
        ds = TabularRegressionFromExcel(path, target_column="target")
        x, y = ds[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)

    def test_target_name(self, tmp_path):
        path = _make_excel(tmp_path)
        ds = TabularRegressionFromExcel(path, target_column="target")
        assert ds.target_name == "target"


class TestFileValidation:
    def test_non_excel_extension_raises(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("f1,target\n1,10\n")
        with pytest.raises(ValueError, match=".xls or .xlsx"):
            TabularRegressionFromExcel(path, target_column="target")


class TestFeatures:
    def test_normalize(self, tmp_path):
        path = _make_excel(tmp_path)
        ds = TabularRegressionFromExcel(
            path, target_column="target", normalize=True
        )
        mean = ds.features.mean(dim=0)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
