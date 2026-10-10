"""Tests for ContrastiveWrapper — ((x1, x2), 1 if same class else 0)."""

import pytest
import torch
from torch.utils.data import DataLoader

from torchdatasets.image import ImageSubdirDataset
from torchdatasets.metric_learning import ContrastiveWrapper


def pair_ids(wrapper, idx):
    """(first sample id, second sample id, pair label) — toy samples hold their own index."""
    (first, second), pair_label = wrapper[idx]
    return int(first.item()), int(second.item()), pair_label


class TestPairs:
    def test_labels_match_classes(self, toy_dataset):
        ds = toy_dataset([0, 1, 2] * 20)
        wrapper = ContrastiveWrapper(ds, random_state=0)
        for idx in range(len(ds)):
            first, second, pair_label = pair_ids(wrapper, idx)
            assert first == idx
            assert pair_label == int(ds.targets[first] == ds.targets[second])

    def test_positive_is_never_the_anchor(self, toy_dataset):
        wrapper = ContrastiveWrapper(toy_dataset([0, 0, 1, 1]), same_class_prob=1.0)
        for _ in range(50):
            for idx in range(4):
                first, second, pair_label = pair_ids(wrapper, idx)
                assert pair_label == 1 and second != first

    def test_all_negative_with_zero_probability(self, toy_dataset):
        ds = toy_dataset([0, 1, 2] * 4)
        wrapper = ContrastiveWrapper(ds, same_class_prob=0.0)
        for idx in range(len(ds)):
            first, second, pair_label = pair_ids(wrapper, idx)
            assert pair_label == 0 and ds.targets[first] != ds.targets[second]

    def test_probability_sets_the_positive_share(self, toy_dataset):
        wrapper = ContrastiveWrapper(toy_dataset([0, 1] * 500), same_class_prob=0.3, random_state=1)
        share = sum(wrapper[i][1] for i in range(len(wrapper))) / len(wrapper)
        assert 0.25 < share < 0.35

    def test_negatives_spread_over_the_other_classes(self, toy_dataset):
        ds = toy_dataset([0, 1, 2, 3] * 25)
        wrapper = ContrastiveWrapper(ds, same_class_prob=0.0, random_state=2)
        classes = {ds.targets[pair_ids(wrapper, i)[1]] for i in range(0, len(ds), 4)}  # anchors of class 0
        assert classes == {1, 2, 3}


class TestValidation:
    def test_one_class_needs_probability_one(self, toy_dataset):
        with pytest.raises(ValueError, match="at least 2 classes"):
            ContrastiveWrapper(toy_dataset([0, 0, 0]))
        wrapper = ContrastiveWrapper(toy_dataset([0, 0, 0]), same_class_prob=1.0)
        assert pair_ids(wrapper, 0)[2] == 1

    @pytest.mark.parametrize("prob", [-0.1, 1.5])
    def test_probability_range(self, toy_dataset, prob):
        with pytest.raises(ValueError, match="same_class_prob"):
            ContrastiveWrapper(toy_dataset([0, 1]), same_class_prob=prob)

    def test_single_sample_class_warns_and_pairs_with_itself(self, toy_dataset):
        with pytest.warns(UserWarning, match="single sample"):
            wrapper = ContrastiveWrapper(toy_dataset([0, 0, 1]), same_class_prob=1.0)
        assert pair_ids(wrapper, 2)[:2] == (2, 2)


class TestOutput:
    def test_transform_pair_only_touches_the_second_sample(self, toy_dataset):
        wrapper = ContrastiveWrapper(toy_dataset([0, 1] * 3), transform_pair=lambda x: x + 100)
        (first, second), _ = wrapper[0]
        assert first.item() == 0 and second.item() >= 100

    def test_batches_with_the_default_dataloader(self, toy_dataset):
        (first, second), pair_labels = next(iter(DataLoader(ContrastiveWrapper(toy_dataset([0, 1] * 4)), batch_size=4)))
        assert first.shape == (4, 1) and second.shape == (4, 1)
        assert pair_labels.dtype == torch.int64

    def test_image_dataset(self, image_subdir_tree):
        root, _ = image_subdir_tree
        (first, second), pair_label = ContrastiveWrapper(ImageSubdirDataset(root), random_state=0)[0]
        assert first.shape == second.shape == (3, 8, 8)
        assert pair_label in (0, 1)
