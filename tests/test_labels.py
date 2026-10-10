"""Tests for the label helpers shared by the split helper and the metric-learning wrappers."""

import numpy as np
import pytest
import torch
from torch.utils.data import ConcatDataset, Dataset, Subset

from torchdatasets._internal.labels import get_targets, indices_by_class, require_class_labels
from torchdatasets.image import ImageSegMultiDirDataset


class TestGetTargets:
    def test_dataset_with_targets(self, toy_dataset):
        assert get_targets(toy_dataset([3, 1, 2])) == [3, 1, 2]

    def test_tensor_targets_become_ints(self):
        class TensorTargets(Dataset):  # torchvision MNIST style
            targets = torch.tensor([1, 0, 1])

            def __len__(self):
                return 3

        targets = get_targets(TensorTargets())
        assert targets == [1, 0, 1]
        assert all(type(t) is int for t in targets)

    def test_subset_and_nested_subset(self, toy_dataset):
        ds = toy_dataset([10, 11, 12, 13, 14])
        assert get_targets(Subset(ds, [4, 0])) == [14, 10]
        assert get_targets(Subset(Subset(ds, [4, 3, 2]), [2, 0])) == [12, 14]

    def test_concat_dataset(self, toy_dataset):
        assert get_targets(ConcatDataset([toy_dataset([0, 1]), toy_dataset([2])])) == [0, 1, 2]

    def test_dataset_without_targets_raises(self, image_seg_paired_dirs):
        img_dir, mask_dir, _ = image_seg_paired_dirs
        with pytest.raises(TypeError, match="no `targets`"):
            get_targets(ImageSegMultiDirDataset(img_dir, mask_dir))

    def test_length_mismatch_raises(self):
        class Broken(Dataset):
            targets = [0, 1]

            def __len__(self):
                return 3

        with pytest.raises(ValueError, match="2 entries for 3 samples"):
            get_targets(Broken())


class TestRequireClassLabels:
    def test_ints_pass_and_numpy_ints_convert(self):
        labels = require_class_labels([1, np.int64(2)], "test")
        assert labels == [1, 2] and all(type(t) is int for t in labels)

    @pytest.mark.parametrize("bad", [[0, 1], 0.5, "cat", True])
    def test_non_class_labels_raise(self, bad):
        with pytest.raises(ValueError, match="stratify needs one integer class label"):
            require_class_labels([0, bad], "stratify")


def test_indices_by_class_sorted_by_label():
    groups = indices_by_class([2, 0, 2, 1])
    assert groups == {0: [1], 1: [3], 2: [0, 2]}
    assert list(groups) == [0, 1, 2]
