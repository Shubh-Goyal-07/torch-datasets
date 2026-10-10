"""Tests for BaseMetricLearningWrapper — label reading, index handling and randomness."""

import pandas as pd
import pytest
from torch.utils.data import ConcatDataset, Subset

from torchdatasets.image import ImageSegMultiDirDataset
from torchdatasets.metric_learning import BaseMetricLearningWrapper, ContrastiveWrapper
from torchdatasets.tabular import TabularDataFrameDataset
from torchdatasets.utils import train_val_split


class TestLabels:
    def test_groups_samples_by_class(self, toy_dataset):
        wrapper = BaseMetricLearningWrapper(toy_dataset([2, 0, 2, 1, 0]))
        assert wrapper.labels == [0, 1, 2]
        assert wrapper.class_to_indices == {0: [1, 4], 1: [3], 2: [0, 2]}
        assert len(wrapper) == 5

    def test_works_on_splits_subsets_and_concats(self, toy_dataset):
        ds = toy_dataset([0, 1] * 5)
        train, _ = train_val_split(ds, val_ratio=0.2)
        assert BaseMetricLearningWrapper(train).labels == [0, 1]
        assert BaseMetricLearningWrapper(Subset(ds, [1, 3])).class_to_indices == {1: [0, 1]}
        assert len(BaseMetricLearningWrapper(ConcatDataset([ds, ds]))) == 20

    def test_works_on_tabular_classification(self):
        df = pd.DataFrame({"x": [0.1, 0.2, 0.3, 0.4], "label": ["cat", "dog", "cat", "dog"]})
        wrapper = BaseMetricLearningWrapper(TabularDataFrameDataset(df, target_cols="label"))
        assert wrapper.class_to_indices == {0: [0, 2], 1: [1, 3]}

    def test_needs_targets(self, image_seg_paired_dirs):
        img_dir, mask_dir, _ = image_seg_paired_dirs
        with pytest.raises(TypeError, match="no `targets`"):
            BaseMetricLearningWrapper(ImageSegMultiDirDataset(img_dir, mask_dir))

    @pytest.mark.parametrize("labels", [[[0], [0, 1]], [0.5, 1.5]])
    def test_rejects_multi_label_and_regression(self, toy_dataset, labels):
        with pytest.raises(ValueError, match="BaseMetricLearningWrapper needs one integer class label"):
            BaseMetricLearningWrapper(toy_dataset(labels))

    def test_rejects_empty_dataset(self, toy_dataset):
        with pytest.raises(ValueError, match="non-empty"):
            BaseMetricLearningWrapper(toy_dataset([]))


class TestIndexing:
    def test_getitem_is_abstract(self, toy_dataset):
        with pytest.raises(NotImplementedError):
            BaseMetricLearningWrapper(toy_dataset([0, 1]))[0]

    def test_negative_index_and_out_of_range(self, toy_dataset):
        wrapper = ContrastiveWrapper(toy_dataset([0, 1, 0, 1]), random_state=0)
        (first, _), _ = wrapper[-1]
        assert first.item() == 3
        with pytest.raises(IndexError):
            wrapper[4]
        with pytest.raises(IndexError):
            wrapper[-5]

    def test_iteration_stops_at_the_end(self, toy_dataset):
        assert len(list(ContrastiveWrapper(toy_dataset([0, 1] * 3)))) == 6

    def test_return_path_datasets_give_samples_only(self, toy_dataset):
        (first, second), _ = ContrastiveWrapper(toy_dataset([0, 1, 0, 1], return_path=True))[0]
        assert first.shape == (1,) and second.shape == (1,)


class TestPickling:
    def test_wrappers_pickle_for_dataloader_workers(self, tmp_image_factory, tmp_path):
        """DataLoader workers on Windows/macOS (spawn) pickle the dataset they get."""
        import pickle

        from torchvision.transforms import v2

        from torchdatasets.image import ImageSubdirDataset
        from torchdatasets.metric_learning import FewShotWrapper, TripletWrapper

        for cls in ("a", "b"):
            for i in range(4):
                tmp_image_factory(f"imgs/{cls}/{i}.png")
        train, _ = train_val_split(ImageSubdirDataset(tmp_path / "imgs"), val_ratio=0.25, stratify=True,
                                   train_transform=v2.Resize((4, 4)))
        for wrapper in (ContrastiveWrapper(train, random_state=0), TripletWrapper(train, random_state=0),
                        FewShotWrapper(train, n_way=2, k_shot=1, q_query=1, random_state=0)):
            restored = pickle.loads(pickle.dumps(wrapper))
            first = restored[0][0]
            first = first[0] if isinstance(first, (list, tuple)) else first
            first = first[0] if isinstance(first, (list, tuple)) else first
            assert first.shape == (3, 4, 4)


class TestRandomness:
    def test_random_state_fixes_partners_per_index(self, toy_dataset):
        ds = toy_dataset([0, 1, 2, 3] * 25)
        a, b = ContrastiveWrapper(ds, random_state=7), ContrastiveWrapper(ds, random_state=7)
        pairs_a = [(a[i][0][1].item(), a[i][1]) for i in range(len(ds))]
        assert pairs_a == [(b[i][0][1].item(), b[i][1]) for i in range(len(ds))]
        assert pairs_a == [(a[i][0][1].item(), a[i][1]) for i in range(len(ds))]  # same again on a re-read
        other = ContrastiveWrapper(ds, random_state=8)
        assert pairs_a != [(other[i][0][1].item(), other[i][1]) for i in range(len(ds))]

    def test_no_random_state_draws_new_partners(self, toy_dataset):
        wrapper = ContrastiveWrapper(toy_dataset([0] * 50 + [1] * 50), same_class_prob=1.0)
        partners = {wrapper[0][0][1].item() for _ in range(30)}
        assert len(partners) > 1
