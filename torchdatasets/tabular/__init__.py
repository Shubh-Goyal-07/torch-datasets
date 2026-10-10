"""Tabular datasets (pandas DataFrame, CSV or Excel).

    from torchdatasets.tabular import TabularCSVDataset
"""

from .base import BaseTabularDataset
from .from_dataframe import TabularDataFrameDataset
from .from_csv import TabularCSVDataset

__all__ = [
    "BaseTabularDataset",
    "TabularDataFrameDataset",
    "TabularCSVDataset",
]
