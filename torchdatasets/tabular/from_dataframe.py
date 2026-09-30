import numpy as np
import pandas as pd
import torch
from typing import Any, List, Dict, Optional, Callable, Union
from .base import BaseTabularDataset

class TabularDatasetFromDataFrame(BaseTabularDataset):
    """Tabular dataset from a pandas DataFrame.

    For classification with a single target column, labels (strings or numbers) are
    encoded to contiguous indices ``0..n_classes-1``; see ``class_to_idx``/``idx_to_class``.

    To build validation/test datasets that are preprocessed exactly like the training
    data, pass the training dataset as ``fit_from``::

        train_ds = TabularDatasetFromDataFrame(train_df, target_cols="label")
        val_ds = TabularDatasetFromDataFrame(val_df, fit_from=train_ds)
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        target_cols: Optional[Union[str, List[str]]] = None,
        feature_cols: Optional[Union[str, List[str]]] = None,
        task: Optional[str] = 'classification',
        # Preprocessing options (passed to base class)
        handle_missing: bool = True,
        fill_missing: Optional[str] = 'mean',
        scaling_type: Optional[str] = 'standard',
        scaling_features: Optional[List[str]] = None,
        encode_categorical: Optional[bool] = True,
        drop_duplicates: Optional[bool] = True,
        n_bins: Optional[int] = 0,
        # Transform options
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        # Reuse fitted preprocessing from another dataset (e.g. train -> val/test)
        fit_from: Optional["TabularDatasetFromDataFrame"] = None
    ) -> None:
        """Initialize DataFrame tabular dataset.

        Args:
            dataframe: Input pandas DataFrame
            target_cols: Target column name or list of names (required unless fit_from is given)
            feature_cols: Feature column name or list of names (None = all non-target columns)
            task: Task type ('classification' or 'regression')
            handle_missing: Whether to handle missing values
            fill_missing: Strategy for missing values ('mean', 'median', 'mode', 'constant')
            scaling_type: Type of scaling ('standard', 'minmax', 'none'; None is treated as 'none')
            scaling_features: List of specific features to scale (None = all numeric)
            encode_categorical: Whether to encode categorical variables
            drop_duplicates: Whether to remove duplicate rows
            n_bins: Number of bins to use for discretization. Only applicable when task is 'classification'
            transform: Optional transform for features
            target_transform: Optional transform for targets
            fit_from: An already-built dataset (e.g. the training set) whose fitted preprocessing, label
                encoding, columns, task and preprocessing options are reused. transform, target_transform
                and drop_duplicates still apply per dataset.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame")

        if fit_from is not None:
            if not isinstance(fit_from, TabularDatasetFromDataFrame):
                raise TypeError("fit_from must be a TabularDatasetFromDataFrame (or TabularDatasetFromCSVXLSX)")
            target_cols = fit_from.target_cols if target_cols is None else target_cols
            feature_cols = fit_from.feature_cols if feature_cols is None else feature_cols
            task = fit_from.task

        self.dataframe = dataframe.copy()
        self.feature_cols = self._as_column_list(feature_cols)
        self.target_cols = self._as_column_list(target_cols)
        self.task = task

        if not self.target_cols:
            raise ValueError("target_cols must be specified")

        if fit_from is not None and (self.feature_cols != fit_from.feature_cols or self.target_cols != fit_from.target_cols):
            raise ValueError(
                "feature_cols/target_cols must match the fit_from dataset "
                f"(features={fit_from.feature_cols}, targets={fit_from.target_cols})"
            )

        # Target encoding state (fitted on the first dataframe, or copied from fit_from)
        self.class_to_idx: Dict[Any, int] = {}
        self.idx_to_class: Dict[int, Any] = {}
        self.bin_edges: Optional[np.ndarray] = None
        self._target_fitted = False

        super().__init__(
            handle_missing=handle_missing,
            fill_missing=fill_missing,
            scaling_type=scaling_type,
            scaling_features=scaling_features,
            encode_categorical=encode_categorical,
            drop_duplicates=drop_duplicates,
            n_bins=n_bins,
            transform=transform,
            target_transform=target_transform,
            fit_from=fit_from
        )

        if fit_from is not None:
            self.class_to_idx = dict(fit_from.class_to_idx)
            self.idx_to_class = dict(fit_from.idx_to_class)
            self.bin_edges = fit_from.bin_edges
            self._target_fitted = fit_from._target_fitted

        self.make_dataset()

    @staticmethod
    def _as_column_list(cols: Optional[Union[str, List[str]]]) -> Optional[List[str]]:
        """Normalize a column argument to a list (a single name becomes a one-item list).

        Args:
            cols (Optional[Union[str, List[str]]]): Column name(s) or None.

        Returns:
            Optional[List[str]]: List of column names, or None.
        """
        if cols is None:
            return None
        if isinstance(cols, str):
            return [cols]
        return list(cols)

    @staticmethod
    def _to_python(value: Any) -> Any:
        """Convert numpy scalars to plain Python values (e.g. np.int64 -> int)."""
        return value.item() if isinstance(value, np.generic) else value

    def _encode_class_labels(self, values: pd.Series) -> np.ndarray:
        """Encode single-column class labels to contiguous indices 0..n_classes-1.

        Args:
            values (pd.Series): Raw target values (strings or numbers).

        Raises:
            ValueError: If a label was not seen when the encoding was fitted.

        Returns:
            np.ndarray: Encoded labels.
        """
        if not self._target_fitted:
            uniques = [self._to_python(v) for v in pd.unique(values)]
            try:
                classes = sorted(uniques)
            except TypeError:
                classes = sorted(uniques, key=str)
            self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
            self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
            self._target_fitted = True

        encoded = values.map(self.class_to_idx)
        unseen = encoded.isna()
        if unseen.any():
            examples = sorted({str(v) for v in values[unseen]})[:5]
            raise ValueError(f"Target labels not seen when the label encoding was fitted: {examples}")
        return encoded.to_numpy(dtype=np.int64)

    def _bin_targets(self, values: pd.Series) -> np.ndarray:
        """Discretize a continuous target into ``n_bins`` equal-width bins.

        Args:
            values (pd.Series): Continuous target values.

        Returns:
            np.ndarray: Bin index of each value.
        """
        if not self._target_fitted:
            _, edges = pd.cut(values, bins=self.n_bins, retbins=True, include_lowest=True)
            self.bin_edges = edges
            self.class_to_idx = {idx: idx for idx in range(len(edges) - 1)}
            self.idx_to_class = dict(self.class_to_idx)
            self._target_fitted = True

        # Open-ended outer bins so values outside the fitted range still get a bin
        edges = np.array(self.bin_edges, dtype=float)
        edges[0], edges[-1] = -np.inf, np.inf
        return pd.cut(values, bins=edges, labels=False).to_numpy(dtype=np.int64)

    def make_dataset(self) -> None:
        """Make the dataset from the DataFrame."""
        if self.feature_cols is None:
            self.feature_cols = [col for col in self.dataframe.columns if col not in self.target_cols]

        missing_features = [col for col in self.feature_cols if col not in self.dataframe.columns]
        missing_targets = [col for col in self.target_cols if col not in self.dataframe.columns]

        if missing_features:
            raise ValueError(f"Feature columns not found in DataFrame: {missing_features}")
        if missing_targets:
            raise ValueError(f"Target columns not found in DataFrame: {missing_targets}")
        if len(self.dataframe) == 0:
            raise ValueError("DataFrame is empty")

        missing_in_targets = self.dataframe[self.target_cols].isnull().sum()
        if missing_in_targets.any():
            counts = {col: int(n) for col, n in missing_in_targets.items() if n > 0}
            raise ValueError(f"Target columns contain missing values {counts}; drop or fill those rows first.")

        feature_df, target_df = self.preprocess_dataframe(self.dataframe, self.feature_cols, self.target_cols)

        self.X = torch.tensor(feature_df.values, dtype=torch.float32)

        if self.task == 'classification':
            if target_df.shape[1] == 1:
                values = target_df.iloc[:, 0]
                if self.n_bins > 0:
                    self.y = torch.tensor(self._bin_targets(values), dtype=torch.long)
                else:
                    self.y = torch.tensor(self._encode_class_labels(values), dtype=torch.long)
            else:
                # Multi-label: one 0/1 indicator column per label
                try:
                    self.y = torch.tensor(target_df.to_numpy(dtype=np.float32), dtype=torch.float32)
                except (TypeError, ValueError):
                    raise ValueError("Multi-column classification targets must be numeric 0/1 indicator columns")
        elif self.task == 'regression':
            self.y = torch.tensor(target_df.values, dtype=torch.float32)
            if self.y.dim() == 2 and self.y.shape[1] == 1:
                self.y = self.y.flatten()
        else:
            raise ValueError(f"Unsupported task type: {self.task}. Use 'classification' or 'regression'")

        if self.task == 'classification' and self.y.dim() == 1:
            self.n_classes = len(self.class_to_idx)

    def get_dataset_info(self) -> Dict[str, Union[str, int, float, list, Dict, None]]:
        """Get comprehensive information about the dataset."""
        info = {
            'source': 'DataFrame',
            'task': self.task,
            'num_samples': len(self),
            'num_features': self.X.shape[1] if self.X is not None else 0,
            'feature_columns': self.feature_cols,
            'target_columns': self.target_cols,
            'features_shape': tuple(self.X.shape) if self.X is not None else None,
            'targets_shape': tuple(self.y.shape) if self.y is not None else None,
            'features_dtype': str(self.X.dtype) if self.X is not None else None,
            'targets_dtype': str(self.y.dtype) if self.y is not None else None,
        }

        # Add classification-specific information
        if self.task == 'classification' and self.y is not None:
            if self.y.dim() == 1:
                info['num_classes'] = len(self.class_to_idx)
                info['classes'] = list(self.class_to_idx.keys())
            else:
                info['num_labels'] = self.y.shape[1]

        # Add regression-specific information
        if self.task == 'regression' and self.y is not None:
            info['num_targets'] = self.y.shape[1] if self.y.dim() == 2 else 1

        info['preprocessing'] = self.get_preprocessing_info()
        return info
