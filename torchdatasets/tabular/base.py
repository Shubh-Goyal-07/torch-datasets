import torch
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from collections.abc import Callable
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from torch.utils.data import Dataset


class BaseTabularDataset(Dataset):
    """Base class for tabular datasets"""

    def __init__(
        self,
        handle_missing: bool = True,
        fill_missing: str = 'mean',
        scaling_type: str = 'standard',
        scaling_features: list = None,
        encode_categorical: bool = True,
        drop_duplicates: bool = False,
        n_bins: int = 0,
        transform: Callable = None,
        target_transform: Callable = None
    ) -> None:
        """Initialize the dataset.

        Args:
            handle_missing (bool, optional): Whether to handle missing values. Defaults to True.
            fill_missing (str, optional): Strategy for missing values ('mean', 'median', 'mode', 'constant'). Defaults to 'mean'.
            scaling_type (str, optional): Type of scaling ('standard', 'minmax', 'none'). Defaults to 'standard'.
            scaling_features (list, optional): List of features to scale (None = all numeric). Defaults to None.
            encode_categorical (bool, optional): Whether to encode categorical variables. Defaults to True.
            drop_duplicates (bool, optional): Whether to remove duplicate rows. Defaults to False.
            transform (Callable, optional): Optional transform to be applied on features. Defaults to None.
            target_transform (Callable, optional): Optional transform to be applied on targets. Defaults to None.
        """
        self.handle_missing = handle_missing
        self.fill_missing = fill_missing
        self.scaling_type = scaling_type
        self.scaling_features = scaling_features if scaling_features is not None else []
        self.encode_categorical = encode_categorical
        self.drop_duplicates = drop_duplicates
        
        self.transform = transform
        self.target_transform = target_transform
        self.X = None
        self.y = None
        
        self.scaler = None
        self.label_encoders = {}
        self._is_fitted = False

        self.n_bins = n_bins
        
        self._validate_init_params()

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
            raise ValueError(f"scaling_type must be one of {valid_scaling_types}")

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
    
    def _handle_missing_values(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        """Handle missing values in the dataframe.

        Args:
            df (pd.DataFrame): Input dataframe.
            feature_cols (list): List of feature columns.

        Returns:
            pd.DataFrame: Dataframe with missing values handled.
        """
        missing_before = df[feature_cols].isnull().sum().sum()
        
        if missing_before == 0:
            print(f"No missing values found")
            return df
                
        df_filled = df.copy()
        
        if self.fill_missing == 'mean':
            numeric_cols = df_filled[feature_cols].select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df_filled[col].isnull().any():
                    df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
            
            non_numeric_cols = df_filled[feature_cols].select_dtypes(exclude=[np.number]).columns
            for col in non_numeric_cols:
                if df_filled[col].isnull().any() and len(df_filled[col].mode()) > 0:
                    df_filled[col] = df_filled[col].fillna(df_filled[col].mode()[0])
        
        elif self.fill_missing == 'median':
            numeric_cols = df_filled[feature_cols].select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df_filled[col].isnull().any():
                    df_filled[col] = df_filled[col].fillna(df_filled[col].median())
            
            non_numeric_cols = df_filled[feature_cols].select_dtypes(exclude=[np.number]).columns
            for col in non_numeric_cols:
                if df_filled[col].isnull().any() and len(df_filled[col].mode()) > 0:
                    df_filled[col] = df_filled[col].fillna(df_filled[col].mode()[0])
        
        elif self.fill_missing == 'mode':
            for col in feature_cols:
                if df_filled[col].isnull().any() and len(df_filled[col].mode()) > 0:
                    df_filled[col] = df_filled[col].fillna(df_filled[col].mode()[0])
        
        elif self.fill_missing == 'constant':
            for col in feature_cols:
                if df_filled[col].isnull().any():
                    if pd.api.types.is_numeric_dtype(df_filled[col]):
                        df_filled[col] = df_filled[col].fillna(0)
                    else:
                        df_filled[col] = df_filled[col].fillna('unknown')
        
        return df_filled
    

    def _encode_categorical_features(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        """Encode categorical features in the dataframe.

        Args:
            df (pd.DataFrame): Input dataframe.
            feature_cols (list): List of feature columns to encode.

        Returns:
            pd.DataFrame: Dataframe with categorical features encoded.
        """
        df_encoded = df.copy()
        categorical_cols = df_encoded[feature_cols].select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) == 0:
            return df_encoded
        
        # Encode categorical features
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = df_encoded[col].astype(str).fillna('unknown')
                df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col])
            else:
                df_encoded[col] = df_encoded[col].astype(str).fillna('unknown')
                df_encoded[col] = self.label_encoders[col].transform(df_encoded[col])
        
        return df_encoded


    def _scale_features(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Scale features in the dataframe.

        Args:
            feature_df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with features scaled.
        """

        # Identify numeric columns
        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
        
        # Identify columns to scale
        if self.scaling_features:
            cols_to_scale = [col for col in self.scaling_features if col in numeric_cols]
        else:
            cols_to_scale = list(numeric_cols)
        
        if len(cols_to_scale) == 0:
            return feature_df
                
        df_scaled = feature_df.copy()
        
        # if scaler is not fitted, fit it
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
    
    def preprocess_dataframe(self, df: pd.DataFrame, feature_cols: list, target_cols: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Preprocess the dataframe.

        Args:
            df (pd.DataFrame): Input dataframe.
            feature_cols (list): List of feature columns.
            target_cols (list): List of target columns.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Processed features and targets.
        """

        df_processed = df.copy()
        
        # Drop duplicates
        if self.drop_duplicates:
            initial_len = len(df_processed)
            df_processed = df_processed.drop_duplicates()
            dropped = initial_len - len(df_processed)
            if dropped > 0:
                print(f"Dropped {dropped} duplicate row(s).")
        
        # Handle missing values
        if self.handle_missing:
            df_processed = self._handle_missing_values(df_processed, feature_cols)
        
        # Encode categorical features
        if self.encode_categorical:
            df_processed = self._encode_categorical_features(df_processed, feature_cols)
        
        # Separate features and targets
        try:
            feature_df = df_processed[feature_cols].copy()
            target_df = df_processed[target_cols].copy()
        except KeyError as e:
            raise ValueError(f"Column not found in dataframe: {e}")
        
        # Scale features
        if self.scaling_type != 'none':
            feature_df = self._scale_features(feature_df)
        
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
