"""Tests for BaseTabularClassificationDataset contract."""

import torch
import pytest

from torchdatasets.tabular.classification.base import BaseTabularClassificationDataset


def _make_ds(**overrides):
    """Helper to create a dataset with sensible defaults."""
    defaults = dict(
        features=torch.randn(10, 4),
        targets=torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0]),
    )
    defaults.update(overrides)
    return BaseTabularClassificationDataset(**defaults)


class TestConstruction:
    def test_basic(self):
        ds = _make_ds()
        assert len(ds) == 10

    def test_with_class_names(self):
        ds = _make_ds(
            targets=torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
            class_names=["cat", "dog"],
        )
        assert ds.class_to_idx == {"cat": 0, "dog": 1}
        assert ds.idx_to_class == {0: "cat", 1: "dog"}

    def test_auto_class_names(self):
        ds = _make_ds(targets=torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0]))
        assert len(ds.class_names) == 3


class TestGetItem:
    def test_returns_tuple_by_default(self):
        ds = _make_ds()
        result = ds[0]
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_return_dict_mode(self):
        ds = _make_ds(return_dict=True)
        result = ds[0]
        assert isinstance(result, dict)
        assert "features" in result
        assert "target" in result
        assert "index" in result

    def test_transform_is_applied(self):
        transform = lambda x: x * 2  # noqa: E731
        ds = _make_ds(transform=transform)
        original = ds.features[0]
        x, _ = ds[0]
        torch.testing.assert_close(x, original * 2)

    def test_target_transform_is_applied(self):
        target_transform = lambda y: y + 100  # noqa: E731
        ds = _make_ds(target_transform=target_transform)
        _, y = ds[0]
        assert y == ds.targets[0] + 100

    def test_sample_weights_returned_in_tuple(self):
        weights = torch.ones(10)
        ds = _make_ds(sample_weights=weights)
        result = ds[0]
        assert len(result) == 3  # (features, target, weight)

    def test_sample_weights_in_dict_mode(self):
        weights = torch.ones(10)
        ds = _make_ds(sample_weights=weights, return_dict=True)
        result = ds[0]
        assert "sample_weight" in result

    def test_feature_names_in_dict_mode(self):
        ds = _make_ds(
            return_dict=True,
            feature_names=["a", "b", "c", "d"],
        )
        result = ds[0]
        assert result["feature_names"] == ["a", "b", "c", "d"]


class TestClassDistribution:
    def test_distribution_sums_to_one(self):
        ds = _make_ds()
        dist = ds.get_class_distribution()
        assert pytest.approx(sum(dist.values()), abs=1e-6) == 1.0

    def test_distribution_values(self):
        targets = torch.tensor([0, 0, 0, 1, 1])
        ds = _make_ds(
            features=torch.randn(5, 2),
            targets=targets,
        )
        dist = ds.get_class_distribution()
        assert dist[0] == pytest.approx(0.6)
        assert dist[1] == pytest.approx(0.4)


class TestClassCount:
    def test_counts_match_targets(self):
        targets = torch.tensor([0, 0, 1, 1, 1, 2])
        ds = _make_ds(features=torch.randn(6, 3), targets=targets)
        assert ds.class_count[0] == 2
        assert ds.class_count[1] == 3
        assert ds.class_count[2] == 1


class TestInverseFrequencyWeights:
    def test_output_shape(self):
        targets = [0, 0, 1, 1, 1]
        weights = BaseTabularClassificationDataset.build_inverse_frequency_weights(targets)
        assert weights.shape == (5,)

    def test_minority_class_gets_higher_weight(self):
        targets = [0, 0, 0, 1]
        weights = BaseTabularClassificationDataset.build_inverse_frequency_weights(targets)
        assert weights[3] > weights[0]  # class 1 (minority) > class 0


class TestValidationErrors:
    def test_1d_features_raises(self):
        with pytest.raises(ValueError, match="2D tensor"):
            _make_ds(features=torch.randn(10))

    def test_2d_targets_raises(self):
        with pytest.raises(ValueError, match="1D tensor"):
            _make_ds(targets=torch.randn(10, 2))

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same number"):
            _make_ds(
                features=torch.randn(10, 4),
                targets=torch.tensor([0, 1, 2]),
            )

    def test_mismatched_weights_raises(self):
        with pytest.raises(ValueError, match="sample_weights"):
            _make_ds(sample_weights=torch.ones(5))  # 10 samples, 5 weights
