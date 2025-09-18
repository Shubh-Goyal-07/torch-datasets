from __future__ import annotations
from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from typing import List, Callable, Optional, Tuple, Dict, Any, Union


class BaseTabularDataset(Dataset):
    def __init__(
        self,
        handle_missing: bool = True,
        fill_missing: str = 'mean',
        scaling_type: str = 'standard',
        scaling_features: Optional[List[str]] = None,
        encode_categorical: bool = True,
        drop_duplicates: bool = False,
        transform: Optional[Callable[[Any], Any]] = None,
        label_transform: Optional[Callable[[Any], Any]] = None
    ) -> None:
        """
        A base dataset class for tabular data with preprocessing capabilities.

        Args:
            handle_missing (bool): Whether to handle missing values automatically.
            fill_missing (str): Strategy for missing values. Options: 'mean', 'median', 'mode', 'constant'.
            scaling_type (str): Type of scaling. Options: 'standard', 'minmax', 'none'.
            scaling_features (Optional[List[str]]): Specific features to scale. If None, all numeric features are scaled.
            encode_categorical (bool): Whether to encode categorical variables.
            drop_duplicates (bool): Whether to remove duplicate rows.
            transform (Optional[Callable]): Optional transform to be applied on features.
            label_transform (Optional[Callable]): Optional transform to be applied on labels.
        """
        super().__init__()

        self.handle_missing: bool = handle_missing
        self.fill_missing: str = fill_missing
        self.scaling_type: str = scaling_type
        self.scaling_features: List[str] = scaling_features or []
        self.encode_categorical: bool = encode_categorical
        self.drop_duplicates: bool = drop_duplicates

        self.transform: Optional[Callable] = transform
        self.label_transform: Optional[Callable] = label_transform
        self.features: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None

        self.scaler: Optional[Union[StandardScaler, MinMaxScaler]] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self._is_fitted: bool = False

        self._validate_init_params()

    def _validate_init_params(self) -> None:
        """Validate initialization parameters."""
        valid_fill_strategies = ['mean', 'median', 'mode', 'constant']
        assert self.fill_missing in valid_fill_strategies, \
            f"fill_missing must be one of {valid_fill_strategies}, got {self.fill_missing}"

        valid_scaling_types = ['standard', 'minmax', 'none']
        assert self.scaling_type in valid_scaling_types, \
            f"scaling_type must be one of {valid_scaling_types}, got {self.scaling_type}"

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return 0 if self.features is None else len(self.features)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        Retrieve a single sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[Any, Any]: (features, label) for the sample.
        """
        assert self.features is not None and self.labels is not None, \
            "Dataset not initialized. Call preprocess_dataframe() or a dataset setup method first."

        feature = self.features[idx]
        label = self.labels[idx]

        if self.transform:
            feature = self.transform(feature)
        if self.label_transform:
            label = self.label_transform(label)

        return feature, label

    def preprocess_dataframe(
        self, df: pd.DataFrame, feature_cols: List[str], label_cols: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply preprocessing steps to a pandas DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.
            feature_cols (List[str]): List of feature column names.
            label_cols (List[str]): List of label column names.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Processed feature and label DataFrames.
        """
        assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"
        assert all(col in df.columns for col in feature_cols), \
            f"Some feature columns not found in DataFrame: {set(feature_cols) - set(df.columns)}"
        assert all(col in df.columns for col in label_cols), \
            f"Some label columns not found in DataFrame: {set(label_cols) - set(df.columns)}"

        df_processed = df.copy()

        if self.drop_duplicates:
            df_processed = df_processed.drop_duplicates()

        if self.handle_missing:
            df_processed = self._handle_missing_values(df_processed, feature_cols)

        if self.encode_categorical:
            df_processed = self._encode_categorical_features(df_processed, feature_cols)

        feature_df = df_processed[feature_cols].copy()
        label_df = df_processed[label_cols].copy()

        if self.scaling_type != 'none':
            feature_df = self._scale_features(feature_df)

        return feature_df, label_df

    def _handle_missing_values(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        Handle missing values in the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.
            feature_cols (List[str]): Columns to check for missing values.

        Returns:
            pd.DataFrame: DataFrame with missing values handled.
        """
        missing_before = df[feature_cols].isnull().sum().sum()
        if missing_before == 0:
            return df

        df_filled = df.copy()

        if self.fill_missing == 'mean':
            numeric_cols = df_filled[feature_cols].select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df_filled[col].isnull().any():
                    df_filled[col].fillna(df_filled[col].mean(), inplace=True)

            non_numeric_cols = df_filled[feature_cols].select_dtypes(exclude=[np.number]).columns
            for col in non_numeric_cols:
                if df_filled[col].isnull().any() and not df_filled[col].mode().empty:
                    df_filled[col].fillna(df_filled[col].mode()[0], inplace=True)

        elif self.fill_missing == 'median':
            numeric_cols = df_filled[feature_cols].select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df_filled[col].isnull().any():
                    df_filled[col].fillna(df_filled[col].median(), inplace=True)

            non_numeric_cols = df_filled[feature_cols].select_dtypes(exclude=[np.number]).columns
            for col in non_numeric_cols:
                if df_filled[col].isnull().any() and not df_filled[col].mode().empty:
                    df_filled[col].fillna(df_filled[col].mode()[0], inplace=True)

        elif self.fill_missing == 'mode':
            for col in feature_cols:
                if df_filled[col].isnull().any() and not df_filled[col].mode().empty:
                    df_filled[col].fillna(df_filled[col].mode()[0], inplace=True)

        elif self.fill_missing == 'constant':
            for col in feature_cols:
                if df_filled[col].isnull().any():
                    if pd.api.types.is_numeric_dtype(df_filled[col]):
                        df_filled[col].fillna(0, inplace=True)
                    else:
                        df_filled[col].fillna('unknown', inplace=True)

        missing_after = df_filled[feature_cols].isnull().sum().sum()
        assert missing_after == 0, "Failed to fill all missing values"
        return df_filled

    def _encode_categorical_features(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        Encode categorical features using LabelEncoder.

        Args:
            df (pd.DataFrame): Input DataFrame.
            feature_cols (List[str]): Feature columns.

        Returns:
            pd.DataFrame: Encoded DataFrame.
        """
        df_encoded = df.copy()
        categorical_cols = df_encoded[feature_cols].select_dtypes(include=['object', 'category']).columns

        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(
                    df_encoded[col].astype(str).fillna('unknown')
                )
            else:
                df_encoded[col] = self.label_encoders[col].transform(
                    df_encoded[col].astype(str).fillna('unknown')
                )

        return df_encoded

    def _scale_features(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numeric features.

        Args:
            feature_df (pd.DataFrame): Feature DataFrame.

        Returns:
            pd.DataFrame: Scaled features.
        """
        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
        cols_to_scale = [col for col in (self.scaling_features or numeric_cols) if col in numeric_cols]

        if not cols_to_scale:
            return feature_df

        df_scaled = feature_df.copy()

        if not self._is_fitted:
            if self.scaling_type == 'standard':
                self.scaler = StandardScaler()
            elif self.scaling_type == 'minmax':
                self.scaler = MinMaxScaler()
            df_scaled[cols_to_scale] = self.scaler.fit_transform(df_scaled[cols_to_scale])
            self._is_fitted = True
        else:
            df_scaled[cols_to_scale] = self.scaler.transform(df_scaled[cols_to_scale])

        return df_scaled

    def get_preprocessing_info(self) -> Dict[str, Any]:
        """
        Get information about preprocessing configuration and dataset state.

        Returns:
            Dict[str, Any]: Dictionary with preprocessing metadata.
        """
        return {
            'handle_missing': self.handle_missing,
            'fill_missing': self.fill_missing,
            'scaling_type': self.scaling_type,
            'scaling_features': self.scaling_features,
            'encode_categorical': self.encode_categorical,
            'drop_duplicates': self.drop_duplicates,
            'is_fitted': self._is_fitted,
            'scaler_type': type(self.scaler).__name__ if self.scaler else None,
            'encoded_features': list(self.label_encoders.keys()),
            'dataset_shape': (
                (len(self.features), self.features.shape[1])
                if self.features is not None
                else None
            ),
        }
