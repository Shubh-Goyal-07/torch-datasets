import pandas as pd
import torch
from typing import List, Dict, Optional, Callable, Union
from .base import BaseTabularDataset

class TabularDatasetFromDataFrame(BaseTabularDataset):
    """Tabular dataset from a pandas DataFrame"""
    
    def __init__(
        self, 
        dataframe: pd.DataFrame,
        target_cols: Union[str, List[str]],
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
        target_transform: Optional[Callable] = None
    ) -> None:
        """Initialize DataFrame tabular dataset.
        
        Args:
            dataframe: Input pandas DataFrame
            target_cols: List of target column names (required)
            feature_cols: List of feature column names (None = auto-detect)
            task: Task type ('classification' or 'regression')
            handle_missing: Whether to handle missing values
            fill_missing: Strategy for missing values ('mean', 'median', 'mode', 'constant')
            scaling_type: Type of scaling ('standard', 'minmax', 'none')
            scaling_features: List of specific features to scale (None = all numeric)
            encode_categorical: Whether to encode categorical variables
            drop_duplicates: Whether to remove duplicate rows
            n_bins: Number of bins to use for discretization. Only applicable when task is 'classification'
            transform: Optional transform for features
            target_transform: Optional transform for targets
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame")
            
        self.dataframe = dataframe.copy()
        self.feature_cols = feature_cols
        self.target_cols = target_cols if isinstance(target_cols, list) else [target_cols] if target_cols else None
        self.task = task
        
        if self.target_cols is None:
            raise ValueError("target_cols must be specified")
            
        super().__init__(
            handle_missing=handle_missing,
            fill_missing=fill_missing,
            scaling_type=scaling_type,
            scaling_features=scaling_features,
            encode_categorical=encode_categorical,
            drop_duplicates=drop_duplicates,
            n_bins=n_bins,
            transform=transform,
            target_transform=target_transform
        )
        
        self.make_dataset()

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
            
        feature_df, target_df = self.preprocess_dataframe(self.dataframe, self.feature_cols, self.target_cols)
        
        self.X = torch.tensor(feature_df.values, dtype=torch.float32)
        
        if self.task == 'classification':
            if target_df.shape[1] == 1:
                if self.n_bins > 0:
                    self.y = pd.cut(target_df.values.flatten(), bins=self.n_bins, labels=False, include_lowest=True)
                    self.y = torch.tensor(self.y, dtype=torch.long)
                else:
                    self.y = torch.tensor(target_df.values.flatten(), dtype=torch.long)
            else:
                self.y = torch.tensor(target_df.values, dtype=torch.float32)
        elif self.task == 'regression':
            self.y = torch.tensor(target_df.values, dtype=torch.float32)
            if self.y.dim() == 2 and self.y.shape[1] == 1:
                self.y = self.y.flatten()  
        else:
            raise ValueError(f"Unsupported task type: {self.task}. Use 'classification' or 'regression'")
            
        if self.task == 'classification' and self.y.dim() == 1:
            unique_classes = torch.unique(self.y)
            self.n_classes = len(unique_classes)
            self.class_to_idx = {int(cls): idx for idx, cls in enumerate(unique_classes)}

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
                unique_classes = torch.unique(self.y)
                info['num_classes'] = len(unique_classes)
                info['classes'] = unique_classes.tolist()
            else:
                info['num_labels'] = self.y.shape[1]
                
        # Add regression-specific information
        if self.task == 'regression' and self.y is not None:
            info['num_targets'] = self.y.shape[1] if self.y.dim() == 2 else 1
            
        info['preprocessing'] = self.get_preprocessing_info()        
        return info