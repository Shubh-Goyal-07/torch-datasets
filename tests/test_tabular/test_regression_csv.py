"""Tests for TabularRegressionFromCSV — end-to-end CSV loading for regression."""

import torch
import pytest

from torchdatasets.tabular.regression.from_csv import TabularRegressionFromCSV


def _make_csv(tmp_csv_factory, rows=None):
    if rows is None:
        rows = [
            {"f1": 1.0, "f2": 2.0, "target": 10.5},
            {"f1": 3.0, "f2": 4.0, "target": 20.1},
            {"f1": 5.0, "f2": 6.0, "target": 30.7},
            {"f1": 7.0, "f2": 8.0, "target": 40.3},
        ]
    return tmp_csv_factory("data.csv", rows)


class TestBasicLoading:
    def test_length(self, tmp_csv_factory):
        path = _make_csv(tmp_csv_factory)
        ds = TabularRegressionFromCSV(path, target_column="target")
        assert len(ds) == 4

    def test_getitem_returns_tensors(self, tmp_csv_factory):
        path = _make_csv(tmp_csv_factory)
        ds = TabularRegressionFromCSV(path, target_column="target")
        x, y = ds[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)

    def test_targets_are_float(self, tmp_csv_factory):
        path = _make_csv(tmp_csv_factory)
        ds = TabularRegressionFromCSV(path, target_column="target")
        assert ds.targets.dtype == torch.float32

    def test_target_name_is_set(self, tmp_csv_factory):
        path = _make_csv(tmp_csv_factory)
        ds = TabularRegressionFromCSV(path, target_column="target")
        assert ds.target_name == "target"


class TestFeatureSelection:
    def test_feature_columns(self, tmp_csv_factory):
        path = _make_csv(tmp_csv_factory)
        ds = TabularRegressionFromCSV(
            path, target_column="target", feature_columns=["f1"]
        )
        x, _ = ds[0]
        assert x.shape == (1,)

    def test_drop_columns(self, tmp_csv_factory):
        path = _make_csv(tmp_csv_factory)
        ds = TabularRegressionFromCSV(
            path, target_column="target", drop_columns=["f2"]
        )
        x, _ = ds[0]
        assert x.shape == (1,)


class TestNormalization:
    def test_normalize_flag(self, tmp_csv_factory):
        path = _make_csv(tmp_csv_factory)
        ds = TabularRegressionFromCSV(
            path, target_column="target", normalize=True
        )
        mean = ds.features.mean(dim=0)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)


class TestSampleWeights:
    def test_weight_column(self, tmp_csv_factory):
        rows = [
            {"f1": 1, "target": 10.0, "w": 0.5},
            {"f1": 2, "target": 20.0, "w": 1.5},
        ]
        path = _make_csv(tmp_csv_factory, rows)
        ds = TabularRegressionFromCSV(
            path, target_column="target", sample_weight_column="w"
        )
        assert ds.sample_weights is not None
        assert ds.sample_weights.shape == (2,)


class TestErrors:
    def test_missing_target_column(self, tmp_csv_factory):
        path = _make_csv(tmp_csv_factory)
        with pytest.raises(ValueError, match="target_column"):
            TabularRegressionFromCSV(path, target_column="nonexistent")

    def test_empty_csv_raises(self, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text("f1,target\n")
        with pytest.raises(ValueError, match="empty"):
            TabularRegressionFromCSV(path, target_column="target")
