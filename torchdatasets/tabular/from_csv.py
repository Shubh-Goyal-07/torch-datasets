from pathlib import Path
import pandas as pd
import torch
from .base import BaseTabularDataset


class TabularDatasetFromCSVXLSX(BaseTabularDataset):    
    def __init__(
        self, 
        file_path,
        feature_cols=None,
        target_cols=None,
        task='classification',
        # Preprocessing options (passed to base class)
        handle_missing=True,
        fill_missing='mean',
        scaling_type='standard',
        scaling_features=None,
        encode_categorical=True,
        drop_duplicates=False,
        # Transform options
        transform=None,
        target_transform=None
    ):
        """
        Initialize CSV/XLSX tabular dataset.
        
        Args:
            file_path: Path to CSV or Excel file
            feature_cols: List of feature column names (None = auto-detect)
            target_cols: List of target column names (required)
            task: Task type ('classification' or 'regression')
            handle_missing: Whether to handle missing values
            fill_missing: Strategy for missing values ('mean', 'median', 'mode', 'constant')
            scaling_type: Type of scaling ('standard', 'minmax', 'none')
            scaling_features: List of specific features to scale (None = all numeric)
            encode_categorical: Whether to encode categorical variables
            drop_duplicates: Whether to remove duplicate rows
            transform: Optional transform for features
            target_transform: Optional transform for targets
        """
        self.file_path = Path(file_path)
        self.feature_cols = feature_cols
        self.target_cols = target_cols if isinstance(target_cols, list) else [target_cols] if target_cols else None
        self.task = task
        
        if self.target_cols is None:
            raise ValueError("target_cols must be specified")
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        super().__init__(
            handle_missing=handle_missing,
            fill_missing=fill_missing,
            scaling_type=scaling_type,
            scaling_features=scaling_features,
            encode_categorical=encode_categorical,
            drop_duplicates=drop_duplicates,
            transform=transform,
            target_transform=target_transform
        )
        
        self.make_dataset()


    def make_dataset(self):
        """Load data from file and apply preprocessing pipeline."""
        
        if self.file_path.suffix.lower() == '.csv':
            df = pd.read_csv(self.file_path)
        elif self.file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(self.file_path)
        else:
            raise ValueError(f"Unsupported file format: {self.file_path.suffix}. Supported: .csv, .xlsx, .xls")
        
        
        if self.feature_cols is None:
            self.feature_cols = [col for col in df.columns if col not in self.target_cols]
        
        missing_features = [col for col in self.feature_cols if col not in df.columns]
        missing_targets = [col for col in self.target_cols if col not in df.columns]
        
        if missing_features:
            raise ValueError(f"Feature columns not found in data: {missing_features}")
        if missing_targets:
            raise ValueError(f"Target columns not found in data: {missing_targets}")
        
        feature_df, target_df = self.preprocess_dataframe(df, self.feature_cols, self.target_cols)
        
        self.X = torch.tensor(feature_df.values, dtype=torch.float32)
        
        if self.task == 'classification':
            if target_df.shape[1] == 1:
                self.y = torch.tensor(target_df.values.flatten(), dtype=torch.long)
            else:
                self.y = torch.tensor(target_df.values, dtype=torch.float32)
        elif self.task == 'regression':
            self.y = torch.tensor(target_df.values, dtype=torch.float32)
            if self.y.dim() == 2 and self.y.shape[1] == 1:
                self.y = self.y.flatten()  # Single target regression
        else:
            raise ValueError(f"Unsupported task type: {self.task}. Use 'classification' or 'regression'")
        
        if self.task == 'classification' and self.y.dim() == 1:
            unique_classes = torch.unique(self.y)


    def get_dataset_info(self):
        """Get comprehensive information about the dataset."""
        info = {
            'file_path': str(self.file_path),
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
        if self.task == 'classification' and self.y is not None:
            if self.y.dim() == 1:
                unique_classes = torch.unique(self.y)
                info['num_classes'] = len(unique_classes)
                info['classes'] = unique_classes.tolist()
            else:
                info['num_labels'] = self.y.shape[1]
        
        info['preprocessing'] = self.get_preprocessing_info()
        return info
        

    