"""Tests for torchdatasets.utils — train/val(/test) splitting and TransformSubset."""

import pickle

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from torchdatasets.image import ImageSegMultiDirDataset, ImageSubdirDataset
from torchdatasets.tabular import TabularCSVDataset, TabularDataFrameDataset
from torchdatasets.utils import TransformSubset, train_val_split, train_val_test_split


def sample_ids(part):
    """Toy samples hold their own index, so this tells which samples a part returns."""
    return [int(part[i][0].item()) for i in range(len(part))]


def rare_class_outside_training_seed(targets, rare_class, ratios=(0.3,)):
    """A random_state for which an unstratified split puts every sample of rare_class outside training."""
    from torchdatasets.utils import _split_indices

    for seed in range(200):
        train_rows = _split_indices(len(targets), list(ratios), None, seed)[0]
        if all(targets[i] != rare_class for i in train_rows):
            return seed
    raise AssertionError("no seed puts the rare class outside training")


class TestTrainValSplit:
    def test_sizes_and_disjoint_cover(self, toy_dataset):
        train, val = train_val_split(toy_dataset([0, 1] * 10), val_ratio=0.25)
        assert (len(train), len(val)) == (15, 5)
        assert sorted(train.indices + val.indices) == list(range(20))
        assert train.indices == sorted(train.indices)

    def test_parts_return_their_own_samples(self, toy_dataset):
        train, val = train_val_split(toy_dataset([0, 1] * 5))
        assert sample_ids(val) == val.indices
        assert sample_ids(train) == train.indices

    def test_val_size_rounds_up(self, toy_dataset):
        _, val = train_val_split(toy_dataset([0] * 11), val_ratio=0.2)
        assert len(val) == 3  # ceil(2.2)

    def test_reproducible_and_seed_dependent(self, toy_dataset):
        ds = toy_dataset([0, 1] * 25)
        assert train_val_split(ds)[1].indices == train_val_split(ds)[1].indices
        assert train_val_split(ds, random_state=1)[1].indices != train_val_split(ds, random_state=2)[1].indices

    def test_stratify_keeps_class_proportions(self, toy_dataset):
        train, val = train_val_split(toy_dataset([0] * 80 + [1] * 20), val_ratio=0.2, stratify=True)
        assert val.targets.count(1) == 4 and len(val) == 20
        assert train.targets.count(1) == 16

    @pytest.mark.parametrize("ratio", [0, 1, 1.5, -0.1])
    def test_invalid_ratio_raises(self, toy_dataset, ratio):
        with pytest.raises(ValueError, match="between 0 and 1"):
            train_val_split(toy_dataset([0, 1] * 5), val_ratio=ratio)

    def test_dataset_too_small_raises(self, toy_dataset):
        with pytest.raises(ValueError, match="too small"):
            train_val_split(toy_dataset([0]), val_ratio=0.5)

    def test_stratify_needs_targets(self, image_seg_paired_dirs):
        img_dir, mask_dir, _ = image_seg_paired_dirs
        with pytest.raises(TypeError, match="no `targets`"):
            train_val_split(ImageSegMultiDirDataset(img_dir, mask_dir), stratify=True)

    @pytest.mark.parametrize("labels", [[[0], [0, 1]] * 5, [0.5, 1.5] * 5])
    def test_stratify_rejects_multi_label_and_regression(self, toy_dataset, labels):
        with pytest.raises(ValueError, match="integer class label"):
            train_val_split(toy_dataset(labels), stratify=True)

    def test_unstratified_split_needs_no_targets(self, image_seg_paired_dirs):
        img_dir, mask_dir, count = image_seg_paired_dirs
        train, val = train_val_split(ImageSegMultiDirDataset(img_dir, mask_dir), val_ratio=0.34)
        assert len(train) + len(val) == count


class TestMissingClassWarning:
    def test_warns_when_a_class_has_no_training_samples(self, toy_dataset):
        targets = [0] * 9 + [1]
        seed = rare_class_outside_training_seed(targets, 1)
        with pytest.warns(UserWarning, match=r"1 class\(es\) have no samples in the training part \(e.g. \[1\]\)") as record:
            train_val_split(toy_dataset(targets), val_ratio=0.3, random_state=seed)
        assert record[0].filename == __file__  # points at the caller

    def test_names_the_class_when_the_dataset_knows_it(self, tmp_image_factory, tmp_path):
        for i in range(9):
            tmp_image_factory(f"imgs/common/{i}.png")
        tmp_image_factory("imgs/rare/0.png")
        ds = ImageSubdirDataset(tmp_path / "imgs")
        seed = rare_class_outside_training_seed(ds.targets, ds.class_to_idx["rare"])
        with pytest.warns(UserWarning, match=r"\['rare'\]"):
            train_val_split(ds, val_ratio=0.3, random_state=seed)

    def test_three_way_split_warns_too(self, toy_dataset):
        targets = [0] * 9 + [1]
        seed = rare_class_outside_training_seed(targets, 1, ratios=(0.2, 0.2))
        with pytest.warns(UserWarning, match="no samples in the training part"):
            train_val_test_split(toy_dataset(targets), val_ratio=0.2, test_ratio=0.2, random_state=seed)

    def test_no_warning_when_every_class_is_in_training(self, toy_dataset, recwarn):
        train_val_split(toy_dataset([0, 1] * 10), stratify=True)
        train_val_split(toy_dataset([[0], [1]] * 5))  # multi-label targets: nothing to check
        assert not [w for w in recwarn if "training part" in str(w.message)]


class TestPerSplitTransforms:
    def test_each_part_gets_its_transform(self, toy_dataset):
        ds = toy_dataset([0, 1] * 5)
        train, val = train_val_split(ds, train_transform=lambda x: x + 100, val_transform=lambda x: -x)
        assert [int(train[i][0].item()) for i in range(len(train))] == [i + 100 for i in train.indices]
        assert [int(val[i][0].item()) for i in range(len(val))] == [-i for i in val.indices]
        # the original dataset is untouched
        assert ds.transform is None and ds[3][0].item() == 3

    def test_none_keeps_the_dataset_transform(self, toy_dataset):
        ds = toy_dataset([0, 1] * 5)
        ds.transform = lambda x: x + 0.5
        train, val = train_val_split(ds, val_transform=lambda x: x * 0)
        assert train[0][0].item() == train.indices[0] + 0.5
        assert val[0][0].item() == 0

    def test_part_transform_replaces_the_dataset_transform(self, toy_dataset):
        ds = toy_dataset([0, 1] * 5)
        ds.transform = lambda x: x + 0.5
        train, _ = train_val_split(ds, train_transform=lambda x: x * 10)
        assert train[0][0].item() == train.indices[0] * 10  # not (i + 0.5) * 10

    def test_setting_a_part_transform_to_none_removes_it(self, toy_dataset):
        ds = toy_dataset([0, 1] * 5)
        ds.transform = lambda x: x + 0.5
        train, val = train_val_split(ds)
        val.transform = None
        assert val[0][0].item() == val.indices[0]
        assert train[0][0].item() == train.indices[0] + 0.5 and ds.transform is not None

    def test_image_parts_resize_differently(self, image_subdir_tree):
        root, _ = image_subdir_tree
        ds = ImageSubdirDataset(root)
        train, val = train_val_split(ds, val_ratio=0.4, train_transform=v2.Resize((4, 4)), val_transform=v2.Resize((2, 2)))
        assert train[0][0].shape == (3, 4, 4)
        assert val[0][0].shape == (3, 2, 2)
        assert ds[0][0].shape == (3, 8, 8)

    def test_segmentation_transform_moves_mask_too(self, image_seg_paired_dirs):
        img_dir, mask_dir, _ = image_seg_paired_dirs
        train, val = train_val_split(ImageSegMultiDirDataset(img_dir, mask_dir), val_ratio=0.34,
                                     train_transform=v2.Resize((4, 4)))
        image, mask = train[0]
        assert image.shape == (3, 4, 4) and mask.shape == (1, 4, 4)
        assert val[0][0].shape == (3, 8, 8)

    def test_parts_pickle_for_dataloader_workers(self, image_subdir_tree):
        root, _ = image_subdir_tree
        train, _ = train_val_split(ImageSubdirDataset(root), train_transform=v2.Resize((4, 4)))
        restored = pickle.loads(pickle.dumps(train))
        assert restored[0][0].shape == (3, 4, 4)
        assert restored.indices == train.indices


class TestTransformSubset:
    def test_setting_transform_later(self, toy_dataset):
        ds = toy_dataset([0, 1, 0, 1])
        sub = TransformSubset(ds, [1, 3])
        assert sub.transform is None
        sub.transform = lambda x: x * 10
        assert sub[1][0].item() == 30
        assert ds[3][0].item() == 3

    def test_targets_follow_subset_order(self, toy_dataset):
        assert TransformSubset(toy_dataset([5, 6, 7, 8]), [3, 0]).targets == [8, 5]

    def test_nested_subset_transform_replaces_outer_without_touching_it(self, toy_dataset):
        ds = toy_dataset(list(range(10)))
        outer = TransformSubset(ds, range(10), transform=lambda x: x + 100)
        inner = TransformSubset(outer, [2, 4], transform=lambda x: x + 1000)
        assert inner[0][0].item() == 1002
        assert outer[2][0].item() == 102
        assert ds[2][0].item() == 2
        assert inner.targets == [2, 4]

    def test_dataset_without_transform_attribute(self):
        class Bare(Dataset):
            def __len__(self):
                return 3

            def __getitem__(self, idx):
                return idx, 0

        assert TransformSubset(Bare(), [2])[0] == (2, 0)
        with pytest.raises(TypeError, match="no `transform` attribute"):
            TransformSubset(Bare(), [0], transform=lambda x: x)


class TestTrainValTestSplit:
    def test_sizes_disjoint_and_complete(self, toy_dataset):
        train, val, test = train_val_test_split(toy_dataset([0, 1] * 50), val_ratio=0.1, test_ratio=0.2)
        assert (len(train), len(val), len(test)) == (70, 10, 20)
        assert sorted(train.indices + val.indices + test.indices) == list(range(100))

    def test_stratified(self, toy_dataset):
        train, val, test = train_val_test_split(toy_dataset([0] * 60 + [1] * 40), stratify=True)
        assert (val.targets.count(1), test.targets.count(1), train.targets.count(1)) == (4, 4, 32)

    def test_each_part_has_its_transform(self, toy_dataset):
        train, val, test = train_val_test_split(
            toy_dataset([0, 1] * 10), train_transform=lambda x: x + 100,
            val_transform=lambda x: x + 200, test_transform=lambda x: x + 300,
        )
        assert train[0][0].item() == train.indices[0] + 100
        assert val[0][0].item() == val.indices[0] + 200
        assert test[0][0].item() == test.indices[0] + 300

    def test_ratios_must_leave_training_data(self, toy_dataset):
        with pytest.raises(ValueError, match="add up to 1"):
            train_val_test_split(toy_dataset([0, 1] * 10), val_ratio=0.5, test_ratio=0.5)


@pytest.fixture
def tabular_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "x": np.arange(40, dtype=float),
        "color": rng.choice(["red", "green", "blue"], 40),
        "label": ["a", "b", "b", "c"] * 10,
    })


class TestTabularSplit:
    def test_preprocessing_is_fitted_on_training_rows_only(self, tabular_df):
        ds = TabularDataFrameDataset(tabular_df, target_cols="label")
        train, val = train_val_split(ds, val_ratio=0.25)
        assert isinstance(train, TabularDataFrameDataset) and isinstance(val, TabularDataFrameDataset)
        train_x = train.dataframe["x"].to_numpy()
        assert train.scaler.mean_[0] == pytest.approx(train_x.mean())
        assert train.scaler.mean_[0] != pytest.approx(tabular_df["x"].mean())
        # validation rows are scaled with the training statistics
        assert val.scaler is train.scaler
        expected = (val.dataframe["x"].to_numpy() - train_x.mean()) / train_x.std()
        assert torch.allclose(val.X[:, 0], torch.tensor(expected, dtype=torch.float32), atol=1e-5)

    def test_rows_and_labels_are_kept(self, tabular_df):
        ds = TabularDataFrameDataset(tabular_df, target_cols="label")
        train, val = train_val_split(ds, val_ratio=0.25, stratify=True)
        assert sorted(list(train.dataframe.index) + list(val.dataframe.index)) == list(range(40))
        assert train.class_to_idx == val.class_to_idx == ds.class_to_idx
        assert sorted(train.targets + val.targets) == sorted(ds.targets)
        assert val.targets.count(ds.class_to_idx["b"]) == 5  # stratified: half of the 10 rows are "b"

    def test_transforms_carry_over(self, tabular_df):
        ds = TabularDataFrameDataset(tabular_df, target_cols="label", target_transform=lambda y: y + 10)
        train, val = train_val_split(ds, val_transform=lambda x: x * 0)
        assert int(train[0][1]) >= 10
        assert torch.all(val[0][0] == 0)
        assert train.transform is None
        val.transform = None
        assert not torch.all(val[0][0] == 0)

    def test_class_only_in_validation_keeps_its_index_and_warns(self):
        df = pd.DataFrame({"x": np.arange(10, dtype=float), "label": ["a"] * 9 + ["rare"]})
        ds = TabularDataFrameDataset(df, target_cols="label")
        seed = rare_class_outside_training_seed(ds.targets, 1)
        with pytest.warns(UserWarning, match=r"no samples in the training part \(e.g. \['rare'\]\)"):
            train, val = train_val_split(ds, val_ratio=0.3, random_state=seed)
        assert train.class_to_idx == val.class_to_idx == {"a": 0, "rare": 1}
        assert 1 in val.targets and 1 not in train.targets

    def test_label_map_of_the_dataset_is_kept(self, tabular_df):
        ds = TabularDataFrameDataset(tabular_df, target_cols="label", label_map={"c": 0, "b": 1, "a": 2})
        train, val = train_val_split(ds, stratify=True)
        assert train.class_to_idx == val.class_to_idx == {"c": 0, "b": 1, "a": 2}
        assert sorted(train.targets + val.targets) == sorted(ds.targets)

    def test_binned_targets_keep_training_bins(self):
        df = pd.DataFrame({"x": np.arange(20, dtype=float), "y": np.arange(20, dtype=float)})
        ds = TabularDataFrameDataset(df, target_cols="y", n_bins=2)
        train, val = train_val_split(ds, val_ratio=0.25, stratify=True)
        assert val.bin_edges is train.bin_edges and len(train.class_to_idx) == 2

    def test_dropped_duplicates_are_respected(self):
        df = pd.DataFrame({"x": [1.0, 1.0, 2.0, 3.0, 4.0, 5.0], "label": [0, 0, 1, 0, 1, 1]})
        ds = TabularDataFrameDataset(df, target_cols="label", drop_duplicates=True)
        train, val = train_val_split(ds, val_ratio=0.4)
        assert len(ds) == 5 and len(train) + len(val) == 5

    def test_csv_dataset_splits_into_dataframe_datasets(self, tmp_path, tabular_df):
        tabular_df.to_csv(tmp_path / "data.csv", index=False)
        ds = TabularCSVDataset(tmp_path / "data.csv", target_cols="label", scaling_type=None)
        train, val = train_val_split(ds)
        assert type(train) is TabularDataFrameDataset and train.scaling_type == "none"
        assert len(train) + len(val) == 40

    def test_three_way_parts_share_the_training_fit(self, tabular_df):
        ds = TabularDataFrameDataset(tabular_df, target_cols="label")
        train, val, test = train_val_test_split(ds, stratify=True)
        assert val.scaler is train.scaler and test.scaler is train.scaler
        assert val.label_encoders["color"] is train.label_encoders["color"]
        assert (len(train), len(val), len(test)) == (32, 4, 4)

    def test_regression_split(self):
        df = pd.DataFrame({"x": np.arange(10, dtype=float), "y": np.linspace(0, 1, 10)})
        ds = TabularDataFrameDataset(df, target_cols="y", task="regression")
        train, val = train_val_split(ds)
        assert (len(train), len(val)) == (8, 2) and train.task == "regression"
        with pytest.raises(ValueError, match="integer class label"):
            train_val_split(ds, stratify=True)
