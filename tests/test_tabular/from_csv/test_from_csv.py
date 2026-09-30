import pytest
import pandas as pd
import numpy as np
import torch
import tempfile
import os
from torchdatasets.tabular.from_csv import TabularDatasetFromCSVXLSX


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def create_classification_csv(path):
    np.random.seed(42)
    df = pd.DataFrame({
        'num_feature1': np.random.rand(10),
        'cat_feature1': ['A', 'B', 'A', 'C', 'B', 'A', 'A', 'C', 'B', 'B'],
        'target_class': np.random.randint(0, 2, 10),
    })
    df.to_csv(path, index=False)
    return df


def create_regression_csv(path):
    np.random.seed(0)
    df = pd.DataFrame({
        'num_feature1': np.random.rand(20),
        'num_feature2': np.random.rand(20) * 100,
        'target_reg': np.random.rand(20) * 50,
    })
    df.to_csv(path, index=False)
    return df


def create_duplicates_csv(path):
    df = pd.DataFrame({
        'f1': [1.0, 1.0, 2.0, 3.0, 3.0],
        'target': [0, 0, 1, 1, 1],
    })
    df.to_csv(path, index=False)
    return df


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestTabularDatasetFromCSVXLSX:

    # ------------------------------------------------------------------ #
    # Fixtures
    # ------------------------------------------------------------------ #

    @pytest.fixture
    def cls_csv(self):
        fd, path = tempfile.mkstemp(suffix='.csv')
        os.close(fd)
        create_classification_csv(path)
        yield path
        os.remove(path)

    @pytest.fixture
    def reg_csv(self):
        fd, path = tempfile.mkstemp(suffix='.csv')
        os.close(fd)
        create_regression_csv(path)
        yield path
        os.remove(path)

    @pytest.fixture
    def dup_csv(self):
        fd, path = tempfile.mkstemp(suffix='.csv')
        os.close(fd)
        create_duplicates_csv(path)
        yield path
        os.remove(path)

    # ------------------------------------------------------------------ #
    # Basic construction
    # ------------------------------------------------------------------ #

    def test_basic_classification_from_csv(self, cls_csv):
        dataset = TabularDatasetFromCSVXLSX(
            file_path=cls_csv,
            target_cols='target_class',
            task='classification',
        )
        assert len(dataset) == 10
        assert dataset.X.shape == (10, 2)
        assert dataset.y.shape == (10,)
        assert dataset.y.dtype == torch.long

    def test_basic_regression_from_csv(self, reg_csv):
        dataset = TabularDatasetFromCSVXLSX(
            file_path=reg_csv,
            target_cols='target_reg',
            task='regression',
        )
        assert len(dataset) == 20
        assert dataset.y.dtype == torch.float32
        assert dataset.y.dim() == 1

    def test_explicit_feature_cols(self, reg_csv):
        dataset = TabularDatasetFromCSVXLSX(
            file_path=reg_csv,
            target_cols='target_reg',
            feature_cols=['num_feature1'],
            task='regression',
        )
        assert dataset.X.shape == (20, 1)

    # ------------------------------------------------------------------ #
    # File-loading errors
    # ------------------------------------------------------------------ #

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            TabularDatasetFromCSVXLSX(
                file_path='nonexistent_file.csv',
                target_cols='target_class',
            )

    # ------------------------------------------------------------------ #
    # get_dataset_info
    # ------------------------------------------------------------------ #

    def test_get_dataset_info_classification(self, cls_csv):
        dataset = TabularDatasetFromCSVXLSX(
            file_path=cls_csv,
            target_cols='target_class',
            feature_cols=['num_feature1'],
            task='classification',
        )
        info = dataset.get_dataset_info()
        assert info['file_path'] == cls_csv
        assert info['source'] == 'CSV/Excel'
        assert info['task'] == 'classification'
        assert info['num_samples'] == 10
        assert info['num_features'] == 1
        assert info['target_columns'] == ['target_class']
        assert info['features_shape'] == (10, 1)
        assert info['targets_shape'] == (10,)
        assert info['features_dtype'] == 'torch.float32'
        assert 'preprocessing' in info
        assert 'num_classes' in info

    def test_get_dataset_info_regression(self, reg_csv):
        dataset = TabularDatasetFromCSVXLSX(
            file_path=reg_csv,
            target_cols='target_reg',
            task='regression',
        )
        info = dataset.get_dataset_info()
        assert info['source'] == 'CSV/Excel'
        assert info['task'] == 'regression'
        assert info['num_targets'] == 1
        assert 'preprocessing' in info

    # ------------------------------------------------------------------ #
    # n_bins discretization
    # ------------------------------------------------------------------ #

    def test_n_bins_classification(self, reg_csv):
        """Continuous target discretized into bins for classification."""
        dataset = TabularDatasetFromCSVXLSX(
            file_path=reg_csv,
            target_cols='target_reg',
            task='classification',
            n_bins=3,
        )
        assert dataset.y.dtype == torch.long
        assert len(torch.unique(dataset.y)) <= 3

    # ------------------------------------------------------------------ #
    # Scaling
    # ------------------------------------------------------------------ #

    def test_scaling_standard(self, reg_csv):
        dataset = TabularDatasetFromCSVXLSX(
            file_path=reg_csv,
            target_cols='target_reg',
            feature_cols=['num_feature2'],
            task='regression',
            scaling_type='standard',
        )
        assert abs(dataset.X[:, 0].mean().item()) < 1e-4

    def test_scaling_minmax(self, reg_csv):
        dataset = TabularDatasetFromCSVXLSX(
            file_path=reg_csv,
            target_cols='target_reg',
            feature_cols=['num_feature1'],
            task='regression',
            scaling_type='minmax',
        )
        assert dataset.X[:, 0].min().item() >= -1e-5
        assert dataset.X[:, 0].max().item() <= 1.0 + 1e-5

    def test_scaling_none(self, reg_csv):
        """With scaling_type='none', raw values should be preserved."""
        raw_df = pd.read_csv(reg_csv)
        dataset = TabularDatasetFromCSVXLSX(
            file_path=reg_csv,
            target_cols='target_reg',
            feature_cols=['num_feature2'],
            task='regression',
            scaling_type='none',
        )
        raw_vals = torch.tensor(raw_df['num_feature2'].values, dtype=torch.float32)
        assert torch.allclose(dataset.X[:, 0], raw_vals, atol=1e-5)

    # ------------------------------------------------------------------ #
    # drop_duplicates
    # ------------------------------------------------------------------ #

    def test_drop_duplicates(self, dup_csv):
        dataset = TabularDatasetFromCSVXLSX(
            file_path=dup_csv,
            target_cols='target',
            task='classification',
            drop_duplicates=True,
        )
        assert len(dataset) == 3  # 5 rows − 2 exact duplicates

    # ------------------------------------------------------------------ #
    # Transforms
    # ------------------------------------------------------------------ #

    def test_transform_applied(self, cls_csv):
        transform = lambda x: x * 0  # noqa: E731
        dataset = TabularDatasetFromCSVXLSX(
            file_path=cls_csv,
            target_cols='target_class',
            feature_cols=['num_feature1'],
            task='classification',
            transform=transform,
        )
        x, _ = dataset[0]
        assert torch.all(x == 0)

    def test_target_transform_applied(self, cls_csv):
        target_transform = lambda y: y + 100  # noqa: E731
        dataset = TabularDatasetFromCSVXLSX(
            file_path=cls_csv,
            target_cols='target_class',
            feature_cols=['num_feature1'],
            task='classification',
            target_transform=target_transform,
        )
        _, y = dataset[0]
        assert y >= 100

    # ------------------------------------------------------------------ #
    # Invalid parameter validation
    # ------------------------------------------------------------------ #

    def test_invalid_fill_missing(self, cls_csv):
        with pytest.raises(ValueError):
            TabularDatasetFromCSVXLSX(
                file_path=cls_csv,
                target_cols='target_class',
                fill_missing='interpolate',
            )

    def test_invalid_scaling_type(self, cls_csv):
        with pytest.raises(ValueError):
            TabularDatasetFromCSVXLSX(
                file_path=cls_csv,
                target_cols='target_class',
                scaling_type='robust',
            )

    def test_invalid_task(self, cls_csv):
        with pytest.raises(ValueError):
            TabularDatasetFromCSVXLSX(
                file_path=cls_csv,
                target_cols='target_class',
                task='clustering',
            )

    def test_missing_target_col(self, cls_csv):
        with pytest.raises(ValueError):
            TabularDatasetFromCSVXLSX(
                file_path=cls_csv,
                target_cols='nonexistent_col',
            )


class TestFitFromCSV:

    def test_val_csv_reuses_train_preprocessing(self, tmp_path):
        train_path, val_path = tmp_path / 'train.csv', tmp_path / 'val.csv'
        pd.DataFrame({
            'num': [0.0, 1.0, 2.0, 3.0],
            'cat': ['A', 'B', 'C', 'A'],
            'y': ['no', 'yes', 'no', 'yes'],
        }).to_csv(train_path, index=False)
        pd.DataFrame({'num': [10.0], 'cat': ['C'], 'y': ['yes']}).to_csv(val_path, index=False)

        train = TabularDatasetFromCSVXLSX(train_path, target_cols='y')
        val = TabularDatasetFromCSVXLSX(val_path, fit_from=train)

        assert val.scaler is train.scaler
        assert val.y.tolist() == [train.class_to_idx['yes']]
        assert val.feature_cols == train.feature_cols


class TestExcelFiles:

    def test_excel_file_matches_csv(self, tmp_path):
        """The same data read from .xlsx and .csv must give identical tensors."""
        df = pd.DataFrame({
            'num': [0.5, 1.5, 2.5, 3.5],
            'cat': ['A', 'B', 'A', 'C'],
            'y': ['no', 'yes', 'no', 'yes'],
        })
        csv_path, xlsx_path = tmp_path / 'data.csv', tmp_path / 'data.xlsx'
        df.to_csv(csv_path, index=False)
        df.to_excel(xlsx_path, index=False)

        from_csv = TabularDatasetFromCSVXLSX(csv_path, target_cols='y')
        from_xlsx = TabularDatasetFromCSVXLSX(xlsx_path, target_cols='y')

        assert torch.equal(from_xlsx.X, from_csv.X)
        assert torch.equal(from_xlsx.y, from_csv.y)
        assert from_xlsx.class_to_idx == from_csv.class_to_idx
        assert from_xlsx.get_dataset_info()['file_path'] == str(xlsx_path)
