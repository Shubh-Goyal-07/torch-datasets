"""Tests for TabularClassificationFromExcel — end-to-end Excel loading."""

import pandas as pd
import torch
import pytest

from torchdatasets.tabular.classification.from_excel import TabularClassificationFromExcel


def _make_excel(tmp_path, rows=None, filename="data.xlsx"):
    """Create a minimal classification Excel file and return its path."""
    if rows is None:
        rows = [
            {"f1": 1.0, "f2": 2.0, "target": "A"},
            {"f1": 3.0, "f2": 4.0, "target": "B"},
            {"f1": 5.0, "f2": 6.0, "target": "A"},
        ]
    path = tmp_path / filename
    pd.DataFrame(rows).to_excel(path, index=False)
    return path


class TestBasicLoading:
    def test_length(self, tmp_path):
        path = _make_excel(tmp_path)
        ds = TabularClassificationFromExcel(path, target_column="target")
        assert len(ds) == 3

    def test_getitem(self, tmp_path):
        path = _make_excel(tmp_path)
        ds = TabularClassificationFromExcel(path, target_column="target")
        x, y = ds[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)


class TestFileValidation:
    def test_non_excel_extension_raises(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("f1,target\n1,A\n")
        with pytest.raises(ValueError, match=".xls or .xlsx"):
            TabularClassificationFromExcel(path, target_column="target")


class TestFeatures:
    def test_normalize(self, tmp_path):
        path = _make_excel(tmp_path)
        ds = TabularClassificationFromExcel(
            path, target_column="target", normalize=True
        )
        mean = ds.features.mean(dim=0)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)

    def test_feature_columns_selection(self, tmp_path):
        path = _make_excel(tmp_path)
        ds = TabularClassificationFromExcel(
            path, target_column="target", feature_columns=["f1"]
        )
        x, _ = ds[0]
        assert x.shape == (1,)
