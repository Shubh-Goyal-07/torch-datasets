import torch
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Callable, Union

from .from_dataframe import TabularDatasetFromDataFrame
from torchdatasets._internal.io.common import load_csv_or_excel


class TabularDatasetFromCSVXLSX(TabularDatasetFromDataFrame):
    """Tabular dataset from CSV or Excel file"""
    
    def __init__(
            self, 
            file_path: str,
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
        """Initialize the dataset from a CSV or Excel file.
        
        Args:
            file_path (str): Path to the CSV or Excel file.
            target_cols (Union[str, List[str]]): List of target column names.
            feature_cols (Optional[Union[str, List[str]]], optional): List of feature column names. Defaults to None.
            task (str, optional): Type of task, options ['classification', 'regression']. Defaults to 'classification'.
            handle_missing (bool, optional): Whether to handle missing values. Defaults to True.
            fill_missing (str, optional): Method to fill missing values, options ['mean', 'median', 'mode', None]. Defaults to 'mean'.
            scaling_type (str, optional): Type of scaling to apply, options ['standard', 'minmax', None]. Defaults to 'standard'.
            scaling_features (Optional[List[str]], optional): List of features to scale. Defaults to None. If None, all numerical features will be scaled.
            encode_categorical (bool, optional): Whether to encode categorical features. Defaults to True.
            drop_duplicates (bool, optional): Whether to drop duplicate rows. Defaults to True.
            n_bins (int, optional): Number of bins to use for discretization. Defaults to 0. Only applicable when task is 'classification', otherwise it will be ignored.
            transform (Optional[Callable], optional): Optional transform to be applied to the features. Defaults to None.
            target_transform (Optional[Callable], optional): Optional transform to be applied to the targets. Defaults to None.
        """

        self.file_path = Path(file_path)
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
            
        df = load_csv_or_excel(self.file_path)
        
        super().__init__(
            dataframe=df,
            target_cols=target_cols,
            feature_cols=feature_cols,
            task=task,
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

    def get_dataset_info(self) -> Dict[str, Union[str, int, float, list, Dict, None]]:
        """Get comprehensive information about the dataset."""
        info = super().get_dataset_info()
        info['file_path'] = str(self.file_path)
        info['source'] = 'CSV/Excel'
        return info
        

    