from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from typing import List, Callable


class BaseTabularDataset(Dataset):
    def __init__(
        self,
        handle_missing: bool = True,
        fill_missing: str = 'mean',
        scaling_type: str = 'standard',
        scaling_features: List = None,
        encode_categorical: bool = True,
        drop_duplicates: bool = False,
        transform: Callable = None,
        target_transform: Callable = None
    ):
        """
        Initialize base tabular dataset with preprocessing options.
        
        Args:
            handle_missing: Whether to handle missing values
            fill_missing: Strategy for missing values ('mean', 'median', 'mode', 'constant')
            scaling_type: Type of scaling ('standard', 'minmax', 'none')
            scaling_features: List of specific features to scale (None = all numeric)
            encode_categorical: Whether to encode categorical variables
            drop_duplicates: Whether to remove duplicate rows
            transform: Optional transform to be applied on features
            target_transform: Optional transform to be applied on targets
        """
        super().__init__()
        
        self.handle_missing = handle_missing
        self.fill_missing = fill_missing
        self.scaling_type = scaling_type
        self.scaling_features = scaling_features or []
        self.encode_categorical = encode_categorical
        self.drop_duplicates = drop_duplicates
        
        self.transform = transform
        self.target_transform = target_transform
        self.X = None
        self.y = None
        
        self.scaler = None
        self.label_encoders = {}
        self._is_fitted = False
        
        self._validate_init_params()
    

    def _validate_init_params(self):
        valid_fill_strategies = ['mean', 'median', 'mode', 'constant']
        if self.fill_missing not in valid_fill_strategies:
            raise ValueError(f"fill_missing must be one of {valid_fill_strategies}")
        
        valid_scaling_types = ['standard', 'minmax', 'none']
        if self.scaling_type not in valid_scaling_types:
            raise ValueError(f"scaling_type must be one of {valid_scaling_types}")
    

    def __len__(self):
        if self.X is None:
            return 0
        return len(self.X)
    

    def __getitem__(self, idx):
        if self.X is None or self.y is None:
            raise RuntimeError("Dataset not initialized. Call make_dataset() first.")
        
        x = self.X[idx]
        y = self.y[idx]
        
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        
        return x, y
    

    def preprocess_dataframe(self, df: pd.DataFrame, feature_cols: list, target_cols: list):
        """
        Apply all preprocessing steps to the dataframe.
        
        Args:
            df: Input pandas DataFrame
            feature_cols: List of feature column names
            target_cols: List of target column names
            
        Returns:
            Tuple of (processed_feature_df, processed_target_df)
        """
        df_processed = df.copy()
        
        if self.drop_duplicates:
            initial_len = len(df_processed)
            df_processed = df_processed.drop_duplicates()
            dropped = initial_len - len(df_processed)
        
        if self.handle_missing:
            df_processed = self._handle_missing_values(df_processed, feature_cols)
        
        if self.encode_categorical:
            df_processed = self._encode_categorical_features(df_processed, feature_cols)
        
        try:
            feature_df = df_processed[feature_cols].copy()
            target_df = df_processed[target_cols].copy()
        except KeyError as e:
            raise ValueError(f"Column not found in dataframe: {e}")
        
        if self.scaling_type != 'none':
            feature_df = self._scale_features(feature_df)
        
        return feature_df, target_df
    

    def _handle_missing_values(self, df: pd.DataFrame, feature_cols: list):
        missing_before = df[feature_cols].isnull().sum().sum()
        
        if missing_before == 0:
            print(f"No missing values found")
            return df
                
        df_filled = df.copy()
        
        if self.fill_missing == 'mean':
            numeric_cols = df_filled[feature_cols].select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df_filled[col].isnull().any():
                    df_filled[col].fillna(df_filled[col].mean(), inplace=True)
            
            non_numeric_cols = df_filled[feature_cols].select_dtypes(exclude=[np.number]).columns
            for col in non_numeric_cols:
                if df_filled[col].isnull().any() and len(df_filled[col].mode()) > 0:
                    df_filled[col].fillna(df_filled[col].mode()[0], inplace=True)
        
        elif self.fill_missing == 'median':
            numeric_cols = df_filled[feature_cols].select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df_filled[col].isnull().any():
                    df_filled[col].fillna(df_filled[col].median(), inplace=True)
            
            non_numeric_cols = df_filled[feature_cols].select_dtypes(exclude=[np.number]).columns
            for col in non_numeric_cols:
                if df_filled[col].isnull().any() and len(df_filled[col].mode()) > 0:
                    df_filled[col].fillna(df_filled[col].mode()[0], inplace=True)
        
        elif self.fill_missing == 'mode':
            for col in feature_cols:
                if df_filled[col].isnull().any() and len(df_filled[col].mode()) > 0:
                    df_filled[col].fillna(df_filled[col].mode()[0], inplace=True)
        
        elif self.fill_missing == 'constant':
            for col in feature_cols:
                if df_filled[col].isnull().any():
                    if df_filled[col].dtype in [np.number]:
                        df_filled[col].fillna(0, inplace=True)
                    else:
                        df_filled[col].fillna('unknown', inplace=True)
        
        missing_after = df_filled[feature_cols].isnull().sum().sum()
        
        return df_filled
    

    def _encode_categorical_features(self, df: pd.DataFrame, feature_cols: list):
        df_encoded = df.copy()
        categorical_cols = df_encoded[feature_cols].select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) == 0:
            return df_encoded
                
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = df_encoded[col].astype(str).fillna('unknown')
                df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col])
            else:
                df_encoded[col] = df_encoded[col].astype(str).fillna('unknown')
                df_encoded[col] = self.label_encoders[col].transform(df_encoded[col])
        
        return df_encoded


    def _scale_features(self, feature_df: pd.DataFrame):
        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
        
        if self.scaling_features:
            cols_to_scale = [col for col in self.scaling_features if col in numeric_cols]
        else:
            cols_to_scale = list(numeric_cols)
        
        if len(cols_to_scale) == 0:
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
    
    
    def get_preprocessing_info(self):
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
