"""Tests for BaseTabularRegressionDataset contract."""

import torch
import pytest

from torchdatasets.tabular.regression.base import BaseTabularRegressionDataset


def _make_ds(**overrides):
    """Helper to create a regression dataset with sensible defaults."""
    defaults = dict(
        features=torch.randn(10, 4),
        targets=torch.randn(10),
    )
    defaults.update(overrides)
    return BaseTabularRegressionDataset(**defaults)


class TestConstruction:
    def test_basic(self):
        ds = _make_ds()
        assert len(ds) == 10

    def test_with_target_name(self):
        ds = _make_ds(target_name="price")
        assert ds.target_name == "price"

    def test_with_feature_names(self):
        ds = _make_ds(feature_names=["a", "b", "c", "d"])
        assert ds.feature_names == ["a", "b", "c", "d"]


class TestGetItem:
    def test_returns_tuple_by_default(self):
        ds = _make_ds()
        result = ds[0]
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_feature_shape(self):
        ds = _make_ds()
        x, _ = ds[0]
        assert x.shape == (4,)

    def test_target_is_scalar(self):
        ds = _make_ds()
        _, y = ds[0]
        assert y.ndim == 0  # scalar tensor

    def test_return_dict_mode(self):
        ds = _make_ds(return_dict=True)
        result = ds[0]
        assert isinstance(result, dict)
        assert "features" in result
        assert "target" in result
        assert "index" in result

    def test_dict_mode_includes_target_name(self):
        ds = _make_ds(return_dict=True, target_name="price")
        result = ds[0]
        assert result["target_name"] == "price"

    def test_transform_is_applied(self):
        transform = lambda x: x * 0  # noqa: E731 — zero out
        ds = _make_ds(transform=transform)
        x, _ = ds[0]
        assert torch.all(x == 0)

    def test_target_transform_is_applied(self):
        target_transform = lambda y: y + 999  # noqa: E731
        ds = _make_ds(target_transform=target_transform)
        _, y = ds[0]
        expected = ds.targets[0] + 999
        torch.testing.assert_close(y, expected)

    def test_sample_weights_returned_in_tuple(self):
        weights = torch.ones(10)
        ds = _make_ds(sample_weights=weights)
        result = ds[0]
        assert len(result) == 3

    def test_sample_weights_in_dict(self):
        weights = torch.ones(10)
        ds = _make_ds(sample_weights=weights, return_dict=True)
        result = ds[0]
        assert "sample_weight" in result


class TestMultiTargetRegression:
    """Regression targets can be 2D (multi-output)."""

    def test_2d_targets_accepted(self):
        ds = _make_ds(targets=torch.randn(10, 3))
        assert len(ds) == 10

    def test_2d_target_shape(self):
        ds = _make_ds(targets=torch.randn(10, 3))
        _, y = ds[0]
        assert y.shape == (3,)


class TestValidationErrors:
    def test_1d_features_raises(self):
        with pytest.raises(ValueError, match="2D tensor"):
            _make_ds(features=torch.randn(10))

    def test_3d_targets_raises(self):
        with pytest.raises(ValueError, match="1D or 2D"):
            _make_ds(targets=torch.randn(10, 2, 3))

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same number"):
            _make_ds(
                features=torch.randn(10, 4),
                targets=torch.randn(5),
            )

    def test_mismatched_weights_raises(self):
        with pytest.raises(ValueError, match="sample_weights"):
            _make_ds(sample_weights=torch.ones(5))
