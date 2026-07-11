"""Tests for TabularClassificationFromCSV — end-to-end CSV loading."""

import torch
import pytest

from torchdatasets.tabular.classification.from_csv import TabularClassificationFromCSV


def _make_csv(tmp_csv_factory, rows=None):
    """Create a minimal classification CSV and return its path."""
    if rows is None:
        rows = [
            {"f1": 1.0, "f2": 2.0, "target": "A"},
            {"f1": 3.0, "f2": 4.0, "target": "B"},
            {"f1": 5.0, "f2": 6.0, "target": "A"},
            {"f1": 7.0, "f2": 8.0, "target": "B"},
        ]
    return tmp_csv_factory("data.csv", rows)


class TestBasicLoading:
    def test_length(self, tmp_csv_factory):
        path = _make_csv(tmp_csv_factory)
        ds = TabularClassificationFromCSV(path, target_column="target")
        assert len(ds) == 4

    def test_getitem_returns_tensors(self, tmp_csv_factory):
        path = _make_csv(tmp_csv_factory)
        ds = TabularClassificationFromCSV(path, target_column="target")
        x, y = ds[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)

    def test_feature_shape(self, tmp_csv_factory):
        path = _make_csv(tmp_csv_factory)
        ds = TabularClassificationFromCSV(path, target_column="target")
        x, _ = ds[0]
        assert x.shape == (2,)  # f1, f2

    def test_targets_are_encoded(self, tmp_csv_factory):
        path = _make_csv(tmp_csv_factory)
        ds = TabularClassificationFromCSV(path, target_column="target")
        unique = set(ds.targets.tolist())
        assert unique == {0, 1}


class TestFeatureSelection:
    def test_feature_columns(self, tmp_csv_factory):
        path = _make_csv(tmp_csv_factory)
        ds = TabularClassificationFromCSV(
            path, target_column="target", feature_columns=["f1"]
        )
        x, _ = ds[0]
        assert x.shape == (1,)

    def test_drop_columns(self, tmp_csv_factory):
        path = _make_csv(tmp_csv_factory)
        ds = TabularClassificationFromCSV(
            path, target_column="target", drop_columns=["f2"]
        )
        x, _ = ds[0]
        assert x.shape == (1,)  # f1 only


class TestNormalization:
    def test_normalize_flag(self, tmp_csv_factory):
        path = _make_csv(tmp_csv_factory)
        ds = TabularClassificationFromCSV(
            path, target_column="target", normalize=True
        )
        # After z-score, mean should be ~0
        mean = ds.features.mean(dim=0)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)


class TestCategoricalEncoding:
    def test_non_numeric_features_are_encoded(self, tmp_csv_factory):
        rows = [
            {"color": "red", "target": "A"},
            {"color": "blue", "target": "B"},
            {"color": "red", "target": "A"},
        ]
        path = _make_csv(tmp_csv_factory, rows)
        ds = TabularClassificationFromCSV(path, target_column="target")
        assert ds.features.dtype == torch.float32
        assert len(ds) == 3


class TestSampleWeights:
    def test_weight_column(self, tmp_csv_factory):
        rows = [
            {"f1": 1, "target": "A", "w": 0.5},
            {"f1": 2, "target": "B", "w": 1.5},
        ]
        path = _make_csv(tmp_csv_factory, rows)
        ds = TabularClassificationFromCSV(
            path, target_column="target", sample_weight_column="w"
        )
        assert ds.sample_weights is not None
        assert ds.sample_weights.shape == (2,)


class TestErrors:
    def test_missing_target_column(self, tmp_csv_factory):
        path = _make_csv(tmp_csv_factory)
        with pytest.raises(ValueError, match="target_column"):
            TabularClassificationFromCSV(path, target_column="nonexistent")

    def test_empty_csv_raises(self, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text("f1,target\n")
        with pytest.raises(ValueError, match="empty"):
            TabularClassificationFromCSV(path, target_column="target")

    def test_missing_weight_column_raises(self, tmp_csv_factory):
        path = _make_csv(tmp_csv_factory)
        with pytest.raises(ValueError, match="sample_weight_column"):
            TabularClassificationFromCSV(
                path, target_column="target", sample_weight_column="missing"
            )
