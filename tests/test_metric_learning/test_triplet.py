"""Tests for TripletWrapper — (anchor, positive, negative)."""

import pandas as pd
import pytest
from torch.utils.data import DataLoader

from torchdatasets.metric_learning import TripletWrapper
from torchdatasets.tabular import TabularDatasetFromDataFrame


def triplet_ids(wrapper, idx):
    return tuple(int(x.item()) for x in wrapper[idx])


class TestTriplets:
    def test_classes_of_anchor_positive_negative(self, toy_dataset):
        ds = toy_dataset([0, 1, 2] * 10)
        wrapper = TripletWrapper(ds)
        for _ in range(5):
            for idx in range(len(ds)):
                anchor, positive, negative = triplet_ids(wrapper, idx)
                assert anchor == idx
                assert ds.targets[positive] == ds.targets[anchor] and positive != anchor
                assert ds.targets[negative] != ds.targets[anchor]

    def test_random_state_fixes_triplets(self, toy_dataset):
        ds = toy_dataset([0, 1, 2] * 10)
        a, b = TripletWrapper(ds, random_state=4), TripletWrapper(ds, random_state=4)
        assert [triplet_ids(a, i) for i in range(30)] == [triplet_ids(b, i) for i in range(30)]

    def test_tabular_rows(self):
        df = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "label": ["a", "b", "a", "b"]})
        anchor, positive, negative = TripletWrapper(TabularDatasetFromDataFrame(df, target_cols="label", scaling_type=None))[0]
        assert (anchor.item(), positive.item(), negative.item())[:2] == (0.0, 2.0)
        assert negative.item() in (1.0, 3.0)


class TestValidation:
    def test_needs_two_classes(self, toy_dataset):
        with pytest.raises(ValueError, match="at least 2 classes"):
            TripletWrapper(toy_dataset([1, 1, 1]))

    def test_single_sample_class_warns(self, toy_dataset):
        with pytest.warns(UserWarning, match="single sample"):
            wrapper = TripletWrapper(toy_dataset([0, 0, 1]))
        assert triplet_ids(wrapper, 2)[:2] == (2, 2)


class TestOutput:
    def test_transforms_touch_positive_and_negative_only(self, toy_dataset):
        wrapper = TripletWrapper(toy_dataset([0, 1] * 3), transform_pos=lambda x: x + 100, transform_neg=lambda x: x + 1000)
        anchor, positive, negative = triplet_ids(wrapper, 0)
        assert anchor == 0 and 100 <= positive < 1000 and negative >= 1000

    def test_batches_with_the_default_dataloader(self, toy_dataset):
        anchors, positives, negatives = next(iter(DataLoader(TripletWrapper(toy_dataset([0, 1] * 4)), batch_size=4)))
        assert anchors.shape == positives.shape == negatives.shape == (4, 1)
