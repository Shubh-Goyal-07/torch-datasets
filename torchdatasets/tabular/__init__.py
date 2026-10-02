"""Tabular datasets (pandas DataFrame, CSV or Excel).

    from torchdatasets.tabular import TabularDatasetFromCSVXLSX
"""

from .base import BaseTabularDataset
from .from_dataframe import TabularDatasetFromDataFrame
from .from_csv import TabularDatasetFromCSVXLSX

__all__ = [
    "BaseTabularDataset",
    "TabularDatasetFromDataFrame",
    "TabularDatasetFromCSVXLSX",
]
