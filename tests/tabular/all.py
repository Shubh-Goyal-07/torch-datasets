import unittest
import pandas as pd
import numpy as np
from torchdatasets.tabular.base import BaseTabularDataset


class TestBaseTabularDataset(unittest.TestCase):
    """Unit tests for BaseTabularDataset"""

    def setUp(self):
        # Classification dataset
        self.df_classification = pd.DataFrame({
            "age": [25, 30, np.nan, 40, 35],
            "salary": [50000, np.nan, 60000, 80000, 75000],
            "gender": ["M", "F", "M", np.nan, "F"],
            "label": [0, 1, 0, 1, 1],   # categorical target
        })

        # Regression dataset
        self.df_regression = pd.DataFrame({
            "feature1": [1.2, 2.5, np.nan, 4.0, 5.5],
            "feature2": [10, 20, 30, np.nan, 50],
            "target": [2.3, 3.5, 4.0, 5.7, 6.1],  # continuous target
        })

        # Multi-label dataset
        self.df_multilabel = pd.DataFrame({
            "f1": [0.1, 0.5, 0.9, 0.7, 0.3],
            "f2": [1.0, 2.0, 3.0, 4.0, 5.0],
            "label_a": [1, 0, 1, 0, 1],
            "label_b": [0, 1, 0, 1, 0],
            "label_c": [1, 1, 0, 0, 1],
        })

    def test_invalid_init_params(self):
        with self.assertRaises(ValueError):
            BaseTabularDataset(fill_missing="wrong")

        with self.assertRaises(ValueError):
            BaseTabularDataset(scaling_type="unknown")

    def test_len_and_getitem_classification(self):
        dataset = BaseTabularDataset()
        features, labels = dataset.preprocess_dataframe(
            self.df_classification, ["age", "salary"], ["label"]
        )
        dataset.features, dataset.labels = features.values, labels.values

        self.assertEqual(len(dataset), 5)
        x, y = dataset[0]
        self.assertTrue(isinstance(x, (np.ndarray, list, np.generic)))
        self.assertIsInstance(y, (np.integer, np.generic, np.int_))

    def test_len_and_getitem_regression(self):
        dataset = BaseTabularDataset()
        features, labels = dataset.preprocess_dataframe(
            self.df_regression, ["feature1", "feature2"], ["target"]
        )
        dataset.features, dataset.labels = features.values, labels.values

        self.assertEqual(len(dataset), 5)
        x, y = dataset[0]
        self.assertTrue(isinstance(x, (np.ndarray, list, np.generic)))
        self.assertTrue(isinstance(float(y), float))  # regression target must be float

    def test_len_and_getitem_multilabel(self):
        dataset = BaseTabularDataset()
        features, labels = dataset.preprocess_dataframe(
            self.df_multilabel, ["f1", "f2"], ["label_a", "label_b", "label_c"]
        )
        dataset.features, dataset.labels = features.values, labels.values

        self.assertEqual(len(dataset), 5)
        x, y = dataset[0]
        self.assertTrue(isinstance(x, (np.ndarray, list, np.generic)))
        self.assertEqual(y.shape, (3,))  # multilabel target has 3 labels
        self.assertTrue(all(val in [0, 1] for val in y))  # must be binary

    def test_handle_missing_mean(self):
        dataset = BaseTabularDataset(fill_missing="mean")
        processed = dataset._handle_missing_values(self.df_classification, ["age", "salary"])
        self.assertEqual(processed.isnull().sum().sum(), 0)

    def test_handle_missing_median(self):
        dataset = BaseTabularDataset(fill_missing="median")
        processed = dataset._handle_missing_values(self.df_classification, ["age", "salary"])
        self.assertEqual(processed.isnull().sum().sum(), 0)

    def test_handle_missing_mode(self):
        dataset = BaseTabularDataset(fill_missing="mode")
        processed = dataset._handle_missing_values(self.df_classification, ["gender"])
        self.assertEqual(processed.isnull().sum().sum(), 0)

    def test_handle_missing_constant(self):
        dataset = BaseTabularDataset(fill_missing="constant")
        processed = dataset._handle_missing_values(self.df_classification, ["gender"])
        self.assertEqual(processed.isnull().sum().sum(), 0)

    def test_encode_categorical(self):
        dataset = BaseTabularDataset()
        encoded = dataset._encode_categorical_features(self.df_classification, ["gender"])
        self.assertTrue(pd.api.types.is_integer_dtype(encoded["gender"]))

    def test_scaling_standard(self):
        dataset = BaseTabularDataset(scaling_type="standard")
        features, labels = dataset.preprocess_dataframe(
            self.df_classification, ["age", "salary"], ["label"]
        )
        # After StandardScaler, mean ~ 0
        self.assertAlmostEqual(features.mean().mean(), 0, delta=1)

    def test_scaling_minmax(self):
        dataset = BaseTabularDataset(scaling_type="minmax")
        features, labels = dataset.preprocess_dataframe(
            self.df_classification, ["age", "salary"], ["label"]
        )
        self.assertGreaterEqual(features.min().min(), 0)
        self.assertLessEqual(features.max().max(), 1)

    def test_get_preprocessing_info(self):
        dataset = BaseTabularDataset()
        features, labels = dataset.preprocess_dataframe(
            self.df_classification, ["age", "salary"], ["label"]
        )
        dataset.features, dataset.labels = features.values, labels.values
        info = dataset.get_preprocessing_info()

        self.assertIn("handle_missing", info)
        self.assertIn("scaler_type", info)
        self.assertIn("dataset_shape", info)


if __name__ == "__main__":
    # Run tests with timing
    print("Running Tabular Dataset tests...")
    unittest.main()
