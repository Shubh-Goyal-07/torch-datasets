import logging
import warnings
import torch
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from collections.abc import Callable
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


class BaseTabularDataset(Dataset):
    """Base class for tabular datasets.

    Preprocessing (missing-value filling, categorical encoding and scaling) is *fitted*
    on the first dataframe the dataset processes. Pass an already-built dataset as
    ``fit_from`` to reuse its fitted state instead, so that validation/test data is
    transformed exactly like the training data (no refitting, no leakage).
    """

    def __init__(
        self,
        handle_missing: bool = True,
        fill_missing: str = 'mean',
        scaling_type: Optional[str] = 'standard',
        scaling_features: list = None,
        encode_categorical: bool = True,
        drop_duplicates: bool = False,
        n_bins: int = 0,
        transform: Callable = None,
        target_transform: Callable = None,
        fit_from: Optional["BaseTabularDataset"] = None
    ) -> None:
        """Initialize the dataset.

        Args:
            handle_missing (bool, optional): Whether to handle missing values. Defaults to True.
            fill_missing (str, optional): Strategy for missing values ('mean', 'median', 'mode', 'constant'). Defaults to 'mean'.
            scaling_type (Optional[str], optional): Type of scaling ('standard', 'minmax', 'none'; None is treated as 'none'). Defaults to 'standard'.
            scaling_features (list, optional): List of features to scale (None = all numeric). Defaults to None.
            encode_categorical (bool, optional): Whether to encode categorical variables. Defaults to True.
            drop_duplicates (bool, optional): Whether to remove duplicate rows. Defaults to False.
            n_bins (int, optional): Number of bins to discretize a continuous classification target into. Defaults to 0 (no binning).
            transform (Callable, optional): Optional transform to be applied on features. Defaults to None.
            target_transform (Callable, optional): Optional transform to be applied on targets. Defaults to None.
            fit_from (Optional[BaseTabularDataset], optional): An already-built dataset whose fitted preprocessing
                (fill values, encoders, scaler) and preprocessing options are reused. Defaults to None.
        """
        if fit_from is not None:
            if not isinstance(fit_from, BaseTabularDataset) or not fit_from._is_fitted:
                raise ValueError("fit_from must be a tabular dataset that has already been built.")
            # Preprocessing options must match the fitted state, so take them from fit_from
            handle_missing = fit_from.handle_missing
            fill_missing = fit_from.fill_missing
            scaling_type = fit_from.scaling_type
            scaling_features = fit_from.scaling_features
            encode_categorical = fit_from.encode_categorical
            n_bins = fit_from.n_bins

        self.handle_missing = handle_missing
        self.fill_missing = fill_missing
        self.scaling_type = 'none' if scaling_type is None else scaling_type
        self.scaling_features = list(scaling_features) if scaling_features is not None else []
        self.encode_categorical = encode_categorical
        self.drop_duplicates = drop_duplicates

        self.transform = transform
        self.target_transform = target_transform
        self.X = None
        self.y = None

        self.scaler = None
        self.label_encoders = {}
        self._fill_values = {}
        self._scaled_cols = []
        self._is_fitted = False

        self.n_bins = n_bins

        self._validate_init_params()

        if fit_from is not None:
            self._copy_fitted_state(fit_from)

    def _validate_init_params(self) -> None:
        """Validate initialization parameters.

        Raises:
            ValueError: If invalid fill_missing strategy is provided.
            ValueError: If invalid scaling_type is provided.
        """
        valid_fill_strategies = ['mean', 'median', 'mode', 'constant']
        if self.fill_missing not in valid_fill_strategies:
            raise ValueError(f"fill_missing must be one of {valid_fill_strategies}")

        valid_scaling_types = ['standard', 'minmax', 'none']
        if self.scaling_type not in valid_scaling_types:
            raise ValueError(f"scaling_type must be one of {valid_scaling_types} (or None)")

    def _copy_fitted_state(self, other: "BaseTabularDataset") -> None:
        """Reuse the fitted preprocessing state of another dataset.

        Args:
            other (BaseTabularDataset): An already-built dataset.
        """
        self.scaler = other.scaler
        self.label_encoders = dict(other.label_encoders)
        self._fill_values = dict(other._fill_values)
        self._scaled_cols = list(other._scaled_cols)
        self._is_fitted = True

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        if self.X is None:
            return 0
        return len(self.X)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the features and target.
        """
        if self.X is None or self.y is None:
            raise RuntimeError("Dataset not initialized. Call make_dataset() first.")

        x = self.X[idx]
        y = self.y[idx]

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        return x, y

    @staticmethod
    def _as_category_strings(series: pd.Series) -> pd.Series:
        """Convert a column to strings for encoding, with missing values as 'unknown'.

        Args:
            series (pd.Series): Input column.

        Returns:
            pd.Series: Column of strings.
        """
        values = series.astype(object)
        return values.where(values.notna(), 'unknown').astype(str)

    @staticmethod
    def _is_categorical(series: pd.Series) -> bool:
        """Check whether a column holds categorical (non-numeric) values.

        Args:
            series (pd.Series): Input column.

        Returns:
            bool: True for object, string and category columns.
        """
        return (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        )

    def _compute_fill_values(self, df: pd.DataFrame, feature_cols: list) -> Dict:
        """Compute the value used to fill missing entries of each feature column.

        Args:
            df (pd.DataFrame): Dataframe to fit on.
            feature_cols (list): List of feature columns.

        Returns:
            Dict: Mapping of column name to fill value.
        """
        fill_values = {}
        for col in feature_cols:
            series = df[col]
            is_numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)

            if self.fill_missing == 'constant':
                fill_values[col] = 0 if is_numeric else 'unknown'
                continue

            if is_numeric and self.fill_missing == 'mean':
                value = series.mean()
            elif is_numeric and self.fill_missing == 'median':
                value = series.median()
            else:
                # 'mode', and the fallback for non-numeric columns
                mode = series.mode()
                value = mode.iloc[0] if len(mode) > 0 else None

            if value is not None and not pd.isna(value):
                fill_values[col] = value

        return fill_values

    def _handle_missing_values(self, df: pd.DataFrame, feature_cols: list, fit: bool = True) -> pd.DataFrame:
        """Handle missing values in the dataframe.

        Args:
            df (pd.DataFrame): Input dataframe.
            feature_cols (list): List of feature columns.
            fit (bool, optional): Compute fill values from this dataframe (True) or reuse the fitted ones (False). Defaults to True.

        Returns:
            pd.DataFrame: Dataframe with missing values handled.
        """
        if fit:
            self._fill_values = self._compute_fill_values(df, feature_cols)

        fill_values = {col: val for col, val in self._fill_values.items() if col in feature_cols}
        if not fill_values or not df[feature_cols].isnull().values.any():
            return df

        df_filled = df.copy()
        for col, value in fill_values.items():
            if df_filled[col].isnull().any():
                if isinstance(df_filled[col].dtype, pd.CategoricalDtype) and value not in df_filled[col].cat.categories:
                    df_filled[col] = df_filled[col].cat.add_categories([value])
                df_filled[col] = df_filled[col].fillna(value)

        return df_filled


    def _encode_categorical_features(self, df: pd.DataFrame, feature_cols: list, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features in the dataframe.

        Categories not seen while fitting are encoded as -1 (with a warning).

        Args:
            df (pd.DataFrame): Input dataframe.
            feature_cols (list): List of feature columns to encode.
            fit (bool, optional): Fit new encoders on this dataframe (True) or reuse the fitted ones (False). Defaults to True.

        Returns:
            pd.DataFrame: Dataframe with categorical features encoded.
        """
        df_encoded = df.copy()

        if fit:
            self.label_encoders = {}
            for col in feature_cols:
                if self._is_categorical(df_encoded[col]):
                    encoder = LabelEncoder()
                    df_encoded[col] = encoder.fit_transform(self._as_category_strings(df_encoded[col]))
                    self.label_encoders[col] = encoder
            return df_encoded

        for col, encoder in self.label_encoders.items():
            if col not in feature_cols:
                continue
            mapping = {cls: idx for idx, cls in enumerate(encoder.classes_)}
            encoded = self._as_category_strings(df_encoded[col]).map(mapping)
            unseen = encoded.isna()
            if unseen.any():
                examples = sorted(set(self._as_category_strings(df_encoded[col])[unseen]))[:5]
                warnings.warn(
                    f"Column '{col}': {int(unseen.sum())} value(s) not seen during fitting "
                    f"(e.g. {examples}) were encoded as -1."
                )
                encoded = encoded.fillna(-1)
            df_encoded[col] = encoded.astype(int)

        return df_encoded


    def _scale_features(self, feature_df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale features in the dataframe.

        Args:
            feature_df (pd.DataFrame): Input dataframe.
            fit (bool, optional): Fit a new scaler on this dataframe (True) or reuse the fitted one (False). Defaults to True.

        Returns:
            pd.DataFrame: Dataframe with features scaled.
        """
        if fit:
            # Identify numeric columns
            numeric_cols = feature_df.select_dtypes(include=[np.number]).columns

            # Identify columns to scale
            if self.scaling_features:
                self._scaled_cols = [col for col in self.scaling_features if col in numeric_cols]
            else:
                self._scaled_cols = list(numeric_cols)

            if len(self._scaled_cols) == 0:
                return feature_df

            self.scaler = StandardScaler() if self.scaling_type == 'standard' else MinMaxScaler()
            df_scaled = feature_df.copy()
            df_scaled[self._scaled_cols] = self.scaler.fit_transform(df_scaled[self._scaled_cols])
            return df_scaled

        if self.scaler is None or len(self._scaled_cols) == 0:
            return feature_df

        df_scaled = feature_df.copy()
        df_scaled[self._scaled_cols] = self.scaler.transform(df_scaled[self._scaled_cols])
        return df_scaled

    def preprocess_dataframe(self, df: pd.DataFrame, feature_cols: list, target_cols: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Preprocess the dataframe.

        The first call fits the preprocessing (unless the dataset was created with ``fit_from``);
        later calls reuse the fitted state.

        Args:
            df (pd.DataFrame): Input dataframe.
            feature_cols (list): List of feature columns.
            target_cols (list): List of target columns.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Processed features and targets.
        """
        fit = not self._is_fitted
        df_processed = df.copy()

        # Drop duplicates
        if self.drop_duplicates:
            initial_len = len(df_processed)
            df_processed = df_processed.drop_duplicates()
            dropped = initial_len - len(df_processed)
            if dropped > 0:
                logger.info("Dropped %d duplicate row(s).", dropped)

        # Handle missing values
        if self.handle_missing:
            df_processed = self._handle_missing_values(df_processed, feature_cols, fit=fit)

        # Encode categorical features
        if self.encode_categorical:
            df_processed = self._encode_categorical_features(df_processed, feature_cols, fit=fit)

        # Separate features and targets
        try:
            feature_df = df_processed[feature_cols].copy()
            target_df = df_processed[target_cols].copy()
        except KeyError as e:
            raise ValueError(f"Column not found in dataframe: {e}")

        # Scale features
        if self.scaling_type != 'none':
            feature_df = self._scale_features(feature_df, fit=fit)

        self._is_fitted = True
        return feature_df, target_df

    def get_preprocessing_info(self) -> Dict:
        """Get preprocessing information.

        Returns:
            Dict: Dictionary containing preprocessing information.
        """
        info = {
            'handle_missing': self.handle_missing,
            'fill_missing': self.fill_missing,
            'scaling_type': self.scaling_type,
            'scaling_features': self.scaling_features,
            'encode_categorical': self.encode_categorical,
            'drop_duplicates': self.drop_duplicates,
            'is_fitted': self._is_fitted,
            'scaler_type': type(self.scaler).__name__ if self.scaler else None,
            'encoded_features': list(self.label_encoders.keys()),
            'dataset_shape': (len(self.X), self.X.shape[1]) if self.X is not None else None
        }
        return info
