import pytest
import pandas as pd
import numpy as np
import torch
from torchdatasets.tabular.from_dataframe import TabularDatasetFromDataFrame


def create_sample_dataframe():
    np.random.seed(42)
    return pd.DataFrame({
        'num_feature1': np.random.rand(10),
        'num_feature2': np.random.rand(10) * 100,
        'cat_feature1': ['A', 'B', 'A', 'C', 'B', 'A', 'A', 'C', 'B', 'B'],
        'missing_feature': [1.0, np.nan, 3.0, 4.0, np.nan, 6.0, 7.0, np.nan, 9.0, 10.0],
        'target_class': np.random.randint(0, 2, 10),
        'target_reg': np.random.rand(10) * 50
    })


class TestTabularDatasetFromDataFrame:

    def test_basic_classification(self):
        df = create_sample_dataframe()
        dataset = TabularDatasetFromDataFrame(
            dataframe=df,
            target_cols='target_class',
            feature_cols=['num_feature1', 'num_feature2', 'cat_feature1'],
            task='classification'
        )
        assert len(dataset) == 10
        assert dataset.X.shape == (10, 3)
        assert dataset.y.shape == (10,)
        assert dataset.y.dtype == torch.long
        info = dataset.get_dataset_info()
        assert info['task'] == 'classification'
        assert info['num_classes'] <= 2

    def test_basic_regression(self):
        df = create_sample_dataframe()
        dataset = TabularDatasetFromDataFrame(
            dataframe=df,
            target_cols='target_reg',
            feature_cols=['num_feature1', 'num_feature2'],
            task='regression'
        )
        assert dataset.y.dtype == torch.float32
        assert dataset.y.dim() == 1
        info = dataset.get_dataset_info()
        assert info['task'] == 'regression'

    def test_missing_values_handling(self):
        df = create_sample_dataframe()
        dataset = TabularDatasetFromDataFrame(
            dataframe=df,
            target_cols='target_class',
            feature_cols=['missing_feature'],
            handle_missing=True,
            fill_missing='mean'
        )
        # Check no NaN in tensors
        assert not torch.isnan(dataset.X).any()

    def test_scaling_standard(self):
        df = create_sample_dataframe()
        dataset = TabularDatasetFromDataFrame(
            dataframe=df,
            target_cols='target_class',
            feature_cols=['num_feature2'],
            scaling_type='standard'
        )
        # Mean should be close to 0, std close to 1
        mean = dataset.X[:, 0].mean().item()
        std = dataset.X[:, 0].std().item()
        assert abs(mean) < 1e-4
        assert abs(std - 1.0) < 0.2

    def test_scaling_minmax(self):
        df = create_sample_dataframe()
        dataset = TabularDatasetFromDataFrame(
            dataframe=df,
            target_cols='target_class',
            feature_cols=['num_feature2'],
            scaling_type='minmax'
        )
        min_val = dataset.X[:, 0].min().item()
        max_val = dataset.X[:, 0].max().item()
        assert abs(min_val - 0.0) < 1e-5
        assert abs(max_val - 1.0) < 1e-5

    def test_categorical_encoding(self):
        df = create_sample_dataframe()
        dataset = TabularDatasetFromDataFrame(
            dataframe=df,
            target_cols='target_class',
            feature_cols=['cat_feature1'],
            encode_categorical=True
        )
        # Should be converted to float representation of label encoding
        assert dataset.X.shape == (10, 1)
        unique_vals = torch.unique(dataset.X)
        assert len(unique_vals) == 3

    def test_n_bins_classification(self):
        df = create_sample_dataframe()
        dataset = TabularDatasetFromDataFrame(
            dataframe=df,
            target_cols='target_reg',  # continuous target
            task='classification',
            n_bins=3
        )
        assert dataset.y.dtype == torch.long
        assert len(torch.unique(dataset.y)) <= 3

    def test_auto_detect_features(self):
        df = create_sample_dataframe()
        dataset = TabularDatasetFromDataFrame(
            dataframe=df,
            target_cols='target_class'
        )
        assert dataset.X.shape[1] == 5  # 6 columns total - 1 target

    def test_invalid_task(self):
        df = create_sample_dataframe()
        with pytest.raises(ValueError):
            TabularDatasetFromDataFrame(
                dataframe=df,
                target_cols='target_class',
                task='clustering'
            )

    def test_missing_target(self):
        df = create_sample_dataframe()
        with pytest.raises(ValueError):
            TabularDatasetFromDataFrame(
                dataframe=df,
                target_cols='nonexistent_target'
            )


# ---------------------------------------------------------------------------
# Fill-missing strategies
# ---------------------------------------------------------------------------

class TestFillMissingStrategies:

    def test_fill_missing_median(self):
        df = create_sample_dataframe()
        dataset = TabularDatasetFromDataFrame(
            dataframe=df,
            target_cols='target_class',
            feature_cols=['missing_feature'],
            handle_missing=True,
            fill_missing='median',
            scaling_type='none',
        )
        assert not torch.isnan(dataset.X).any()

    def test_fill_missing_mode(self):
        df = create_sample_dataframe()
        dataset = TabularDatasetFromDataFrame(
            dataframe=df,
            target_cols='target_class',
            feature_cols=['missing_feature'],
            handle_missing=True,
            fill_missing='mode',
            scaling_type='none',
        )
        assert not torch.isnan(dataset.X).any()

    def test_fill_missing_constant_numeric_fills_zero(self):
        df = create_sample_dataframe()
        dataset = TabularDatasetFromDataFrame(
            dataframe=df,
            target_cols='target_class',
            feature_cols=['missing_feature'],
            handle_missing=True,
            fill_missing='constant',
            scaling_type='none',
        )
        assert not torch.isnan(dataset.X).any()
        # NaN positions should have become 0.0
        assert (dataset.X == 0.0).any()

    def test_fill_missing_constant_non_numeric_fills_unknown(self):
        df = pd.DataFrame({
            'cat_with_nan': ['A', None, 'B', None, 'C', 'A', 'B', 'C', 'A', 'B'],
            'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        })
        # Non-numeric NaN filled with 'unknown', then label-encoded
        dataset = TabularDatasetFromDataFrame(
            dataframe=df,
            target_cols='target',
            feature_cols=['cat_with_nan'],
            handle_missing=True,
            fill_missing='constant',
            encode_categorical=True,
        )
        assert not torch.isnan(dataset.X).any()

    def test_handle_missing_false_preserves_nan(self):
        df = create_sample_dataframe()
        dataset = TabularDatasetFromDataFrame(
            dataframe=df,
            target_cols='target_class',
            feature_cols=['missing_feature'],
            handle_missing=False,
            scaling_type='none',   # avoid sklearn NaN error
        )
        assert torch.isnan(dataset.X).any()


# ---------------------------------------------------------------------------
# Scaling options
# ---------------------------------------------------------------------------

class TestScalingOptions:

    def test_scaling_none_preserves_raw_values(self):
        df = create_sample_dataframe()
        dataset = TabularDatasetFromDataFrame(
            dataframe=df,
            target_cols='target_class',
            feature_cols=['num_feature2'],
            scaling_type='none',
        )
        raw = torch.tensor(df['num_feature2'].values, dtype=torch.float32)
        assert torch.allclose(dataset.X[:, 0], raw, atol=1e-5)

    def test_scaling_partial_features(self):
        df = create_sample_dataframe()
        # Scale only num_feature1; num_feature2 (mean ≈ 50) should be unchanged
        dataset = TabularDatasetFromDataFrame(
            dataframe=df,
            target_cols='target_class',
            feature_cols=['num_feature1', 'num_feature2'],
            scaling_type='standard',
            scaling_features=['num_feature1'],
        )
        # col 0: standardized → mean ≈ 0
        assert abs(dataset.X[:, 0].mean().item()) < 1e-4
        # col 1: not scaled → mean ≈ 50
        assert dataset.X[:, 1].mean().item() > 1.0


# ---------------------------------------------------------------------------
# Drop duplicates
# ---------------------------------------------------------------------------

class TestDropDuplicates:

    def test_drop_duplicates_reduces_row_count(self):
        df = pd.DataFrame({
            'f1': [1.0, 1.0, 2.0, 3.0, 3.0],
            'target': [0, 0, 1, 1, 1],
        })
        dataset = TabularDatasetFromDataFrame(
            dataframe=df,
            target_cols='target',
            feature_cols=['f1'],
            task='classification',
            drop_duplicates=True,
        )
        assert len(dataset) == 3  # 5 rows − 2 exact duplicates

    def test_no_drop_duplicates_keeps_all_rows(self):
        df = pd.DataFrame({
            'f1': [1.0, 1.0, 2.0],
            'target': [0, 0, 1],
        })
        dataset = TabularDatasetFromDataFrame(
            dataframe=df,
            target_cols='target',
            feature_cols=['f1'],
            task='classification',
            drop_duplicates=False,
        )
        assert len(dataset) == 3


# ---------------------------------------------------------------------------
# Multi-column targets
# ---------------------------------------------------------------------------

class TestMultiColumnTargets:

    def test_multi_target_regression_shape(self):
        np.random.seed(1)
        df = pd.DataFrame({
            'f1': np.random.rand(10),
            'f2': np.random.rand(10),
            't1': np.random.rand(10),
            't2': np.random.rand(10),
        })
        dataset = TabularDatasetFromDataFrame(
            dataframe=df,
            target_cols=['t1', 't2'],
            task='regression',
        )
        assert dataset.y.shape == (10, 2)
        assert dataset.y.dtype == torch.float32

    def test_multi_target_regression_info_num_targets(self):
        np.random.seed(1)
        df = pd.DataFrame({
            'f1': np.random.rand(10),
            't1': np.random.rand(10),
            't2': np.random.rand(10),
        })
        dataset = TabularDatasetFromDataFrame(
            dataframe=df,
            target_cols=['t1', 't2'],
            task='regression',
        )
        info = dataset.get_dataset_info()
        assert info['num_targets'] == 2

    def test_single_target_regression_info_num_targets(self):
        df = create_sample_dataframe()
        dataset = TabularDatasetFromDataFrame(
            dataframe=df,
            target_cols='target_reg',
            task='regression',
        )
        info = dataset.get_dataset_info()
        assert info['num_targets'] == 1  # 1-D tensor — must not IndexError

    def test_multi_label_classification_shape(self):
        np.random.seed(2)
        df = pd.DataFrame({
            'f1': np.random.rand(10),
            'label1': np.random.randint(0, 2, 10).astype(float),
            'label2': np.random.randint(0, 2, 10).astype(float),
        })
        dataset = TabularDatasetFromDataFrame(
            dataframe=df,
            target_cols=['label1', 'label2'],
            task='classification',
        )
        assert dataset.y.shape == (10, 2)
        assert dataset.y.dtype == torch.float32

    def test_multi_label_classification_info_num_labels(self):
        np.random.seed(2)
        df = pd.DataFrame({
            'f1': np.random.rand(10),
            'label1': np.random.randint(0, 2, 10).astype(float),
            'label2': np.random.randint(0, 2, 10).astype(float),
        })
        dataset = TabularDatasetFromDataFrame(
            dataframe=df,
            target_cols=['label1', 'label2'],
            task='classification',
        )
        info = dataset.get_dataset_info()
        assert 'num_labels' in info
        assert info['num_labels'] == 2


# ---------------------------------------------------------------------------
# Encoding options
# ---------------------------------------------------------------------------

class TestEncodingOptions:

    def test_encode_categorical_false_with_numeric_only(self):
        df = create_sample_dataframe()
        dataset = TabularDatasetFromDataFrame(
            dataframe=df,
            target_cols='target_class',
            feature_cols=['num_feature1', 'num_feature2'],
            encode_categorical=False,
        )
        assert dataset.X.shape == (10, 2)

    def test_encode_categorical_false_with_cat_col_raises(self):
        """String columns must raise when encode_categorical=False (can't cast to float)."""
        df = create_sample_dataframe()
        with pytest.raises(Exception):
            TabularDatasetFromDataFrame(
                dataframe=df,
                target_cols='target_class',
                feature_cols=['cat_feature1'],
                encode_categorical=False,
            )


# ---------------------------------------------------------------------------
# Validation & edge cases
# ---------------------------------------------------------------------------

class TestValidationAndEdgeCases:

    def test_invalid_dataframe_type_raises_type_error(self):
        with pytest.raises(TypeError):
            TabularDatasetFromDataFrame(
                dataframe={'a': [1, 2]},   # not a DataFrame
                target_cols='a',
            )

    def test_invalid_fill_missing_raises(self):
        df = create_sample_dataframe()
        with pytest.raises(ValueError):
            TabularDatasetFromDataFrame(
                dataframe=df,
                target_cols='target_class',
                fill_missing='interpolate',
            )

    def test_invalid_scaling_type_raises(self):
        df = create_sample_dataframe()
        with pytest.raises(ValueError):
            TabularDatasetFromDataFrame(
                dataframe=df,
                target_cols='target_class',
                scaling_type='robust',
            )

    def test_missing_feature_col_raises(self):
        df = create_sample_dataframe()
        with pytest.raises(ValueError):
            TabularDatasetFromDataFrame(
                dataframe=df,
                target_cols='target_class',
                feature_cols=['nonexistent_col'],
            )

    def test_n_bins_ignored_for_regression(self):
        """n_bins must be silently ignored when task='regression'."""
        df = create_sample_dataframe()
        dataset = TabularDatasetFromDataFrame(
            dataframe=df,
            target_cols='target_reg',
            task='regression',
            n_bins=5,
        )
        assert dataset.y.dtype == torch.float32

    def test_empty_dataframe_raises(self):
        df = pd.DataFrame({'f1': pd.Series([], dtype=float), 'target': pd.Series([], dtype=int)})
        with pytest.raises(Exception):
            TabularDatasetFromDataFrame(
                dataframe=df,
                target_cols='target',
                feature_cols=['f1'],
            )


# ---------------------------------------------------------------------------
# Internal attributes & helper methods
# ---------------------------------------------------------------------------

class TestAttributesAndHelpers:

    def test_n_classes_attribute_set(self):
        df = create_sample_dataframe()
        dataset = TabularDatasetFromDataFrame(
            dataframe=df,
            target_cols='target_class',
            task='classification',
        )
        assert hasattr(dataset, 'n_classes')
        assert dataset.n_classes <= 2

    def test_class_to_idx_keys_are_plain_int(self):
        df = create_sample_dataframe()
        dataset = TabularDatasetFromDataFrame(
            dataframe=df,
            target_cols='target_class',
            task='classification',
        )
        assert hasattr(dataset, 'class_to_idx')
        for key in dataset.class_to_idx:
            assert isinstance(key, int), f"Expected int key, got {type(key)}"

    def test_get_preprocessing_info_structure(self):
        df = create_sample_dataframe()
        dataset = TabularDatasetFromDataFrame(
            dataframe=df,
            target_cols='target_class',
            feature_cols=['num_feature1'],
            scaling_type='standard',
        )
        info = dataset.get_preprocessing_info()
        for key in ('handle_missing', 'fill_missing', 'scaling_type',
                    'encode_categorical', 'drop_duplicates', 'is_fitted',
                    'scaler_type', 'encoded_features', 'dataset_shape'):
            assert key in info, f"Missing key: {key}"
        assert info['is_fitted'] is True
        assert info['dataset_shape'] == (10, 1)

    def test_getitem_returns_tensor_tuple(self):
        df = create_sample_dataframe()
        dataset = TabularDatasetFromDataFrame(
            dataframe=df,
            target_cols='target_class',
            feature_cols=['num_feature1'],
        )
        result = dataset[0]
        assert isinstance(result, tuple) and len(result) == 2
        x, y = result
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)

    def test_getitem_index_out_of_range(self):
        df = create_sample_dataframe()
        dataset = TabularDatasetFromDataFrame(
            dataframe=df,
            target_cols='target_class',
            feature_cols=['num_feature1'],
        )
        with pytest.raises(IndexError):
            _ = dataset[999]


# ---------------------------------------------------------------------------
# Class-label encoding and column arguments
# ---------------------------------------------------------------------------

class TestClassLabelEncoding:

    def test_string_labels_are_encoded(self):
        df = pd.DataFrame({'f1': [1.0, 2.0, 3.0, 4.0], 'y': ['cat', 'dog', 'cat', 'bird']})
        dataset = TabularDatasetFromDataFrame(df, target_cols='y', drop_duplicates=False)
        assert dataset.class_to_idx == {'bird': 0, 'cat': 1, 'dog': 2}
        assert dataset.idx_to_class == {0: 'bird', 1: 'cat', 2: 'dog'}
        assert dataset.y.tolist() == [1, 2, 1, 0]
        assert dataset.n_classes == 3
        assert dataset.get_dataset_info()['classes'] == ['bird', 'cat', 'dog']

    def test_non_contiguous_int_labels_are_remapped(self):
        df = pd.DataFrame({'f1': [1.0, 2.0, 3.0, 4.0, 5.0], 'y': [3, 7, 3, 7, 10]})
        dataset = TabularDatasetFromDataFrame(df, target_cols='y')
        assert dataset.class_to_idx == {3: 0, 7: 1, 10: 2}
        assert all(isinstance(k, int) for k in dataset.class_to_idx)
        assert dataset.y.tolist() == [0, 1, 0, 1, 2]
        # Labels must be valid for CrossEntropyLoss with n_classes outputs
        logits = torch.zeros(len(dataset), dataset.n_classes)
        torch.nn.functional.cross_entropy(logits, dataset.y)

    def test_single_feature_col_as_string(self):
        df = create_sample_dataframe()
        dataset = TabularDatasetFromDataFrame(df, target_cols='target_class', feature_cols='num_feature1')
        assert dataset.feature_cols == ['num_feature1']
        assert dataset.X.shape == (10, 1)

    def test_missing_target_values_raise(self):
        df = pd.DataFrame({'f1': [1.0, 2.0, 3.0], 'y': ['a', None, 'b']})
        with pytest.raises(ValueError, match="missing values"):
            TabularDatasetFromDataFrame(df, target_cols='y')

    def test_scaling_type_none_is_accepted(self):
        df = create_sample_dataframe()
        dataset = TabularDatasetFromDataFrame(
            df, target_cols='target_class', feature_cols=['num_feature2'], scaling_type=None
        )
        raw = torch.tensor(df['num_feature2'].values, dtype=torch.float32)
        assert torch.allclose(dataset.X[:, 0], raw, atol=1e-5)


# ---------------------------------------------------------------------------
# Reusing fitted preprocessing on validation / test data
# ---------------------------------------------------------------------------

class TestFitFrom:

    @staticmethod
    def _train_df():
        return pd.DataFrame({
            'num': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            'cat': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A'],
            'y': ['cat', 'dog'] * 5,
        })

    def test_scaler_is_reused(self):
        train = TabularDatasetFromDataFrame(self._train_df(), target_cols='y', feature_cols=['num'])
        val_df = pd.DataFrame({'num': [100.0, 200.0], 'y': ['cat', 'dog']})
        val = TabularDatasetFromDataFrame(val_df, fit_from=train)

        assert val.scaler is train.scaler
        mean, std = 4.5, np.std(np.arange(10.0))  # StandardScaler uses the population std
        expected = torch.tensor([(100.0 - mean) / std, (200.0 - mean) / std], dtype=torch.float32)
        assert torch.allclose(val.X[:, 0], expected, atol=1e-4)

    def test_fill_values_are_reused(self):
        train = TabularDatasetFromDataFrame(
            self._train_df(), target_cols='y', feature_cols=['num'], scaling_type='none'
        )
        val_df = pd.DataFrame({'num': [np.nan, 50.0], 'y': ['cat', 'dog']})
        val = TabularDatasetFromDataFrame(val_df, fit_from=train)
        assert val.X[0, 0].item() == pytest.approx(4.5)  # train mean, not val mean (50)

    def test_category_encoding_is_reused(self):
        train = TabularDatasetFromDataFrame(
            self._train_df(), target_cols='y', feature_cols=['cat'], scaling_type='none'
        )
        val_df = pd.DataFrame({'cat': ['C', 'A'], 'y': ['cat', 'dog']})
        val = TabularDatasetFromDataFrame(val_df, fit_from=train)
        assert val.X[:, 0].tolist() == [2.0, 0.0]  # refitting would give [1, 0]

    def test_unseen_category_is_encoded_as_minus_one(self):
        train = TabularDatasetFromDataFrame(
            self._train_df(), target_cols='y', feature_cols=['cat'], scaling_type='none'
        )
        val_df = pd.DataFrame({'cat': ['Z', 'A'], 'y': ['cat', 'dog']})
        with pytest.warns(UserWarning, match="not seen during fitting"):
            val = TabularDatasetFromDataFrame(val_df, fit_from=train)
        assert val.X[:, 0].tolist() == [-1.0, 0.0]

    def test_label_encoding_is_reused(self):
        train = TabularDatasetFromDataFrame(self._train_df(), target_cols='y')
        val_df = pd.DataFrame({'num': [1.0], 'cat': ['A'], 'y': ['dog']})
        val = TabularDatasetFromDataFrame(val_df, fit_from=train)
        assert val.y.tolist() == [1]  # 'dog' keeps index 1 even though it is the only class here
        assert val.n_classes == 2
        assert val.class_to_idx == train.class_to_idx

    def test_unseen_target_label_raises(self):
        train = TabularDatasetFromDataFrame(self._train_df(), target_cols='y')
        val_df = pd.DataFrame({'num': [1.0], 'cat': ['A'], 'y': ['horse']})
        with pytest.raises(ValueError, match="not seen"):
            TabularDatasetFromDataFrame(val_df, fit_from=train)

    def test_bin_edges_are_reused(self):
        df = pd.DataFrame({'f1': np.arange(11.0), 't': np.arange(11.0)})
        train = TabularDatasetFromDataFrame(df, target_cols='t', task='classification', n_bins=2)
        val_df = pd.DataFrame({'f1': [0.0, 0.0, 0.0], 't': [-5.0, 2.0, 100.0]})
        val = TabularDatasetFromDataFrame(val_df, fit_from=train)
        assert val.y.tolist() == [0, 0, 1]  # out-of-range values fall into the outer bins
        assert np.array_equal(val.bin_edges, train.bin_edges)

    def test_settings_and_columns_are_inherited(self):
        train = TabularDatasetFromDataFrame(
            self._train_df(), target_cols='y', feature_cols=['num'],
            task='classification', scaling_type='minmax', fill_missing='median',
        )
        val = TabularDatasetFromDataFrame(self._train_df().head(3), fit_from=train)
        assert val.task == 'classification'
        assert val.scaling_type == 'minmax'
        assert val.fill_missing == 'median'
        assert val.feature_cols == ['num'] and val.target_cols == ['y']

    def test_mismatched_columns_raise(self):
        train = TabularDatasetFromDataFrame(self._train_df(), target_cols='y', feature_cols=['num'])
        with pytest.raises(ValueError, match="must match"):
            TabularDatasetFromDataFrame(self._train_df(), feature_cols=['num', 'cat'], fit_from=train)

    def test_fit_from_must_be_a_dataset(self):
        with pytest.raises(TypeError):
            TabularDatasetFromDataFrame(self._train_df(), target_cols='y', fit_from="train")
