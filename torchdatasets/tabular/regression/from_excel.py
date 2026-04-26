from pathlib import Path
from typing import Callable, Optional, Sequence

import pandas as pd
import torch

from torchdatasets._internal.io.common import load_csv_or_excel
from torchdatasets.tabular.regression.base import BaseTabularRegressionDataset
from torchdatasets.tabular.regression.from_csv import _prepare_regression_dataframe


class TabularRegressionFromExcel(BaseTabularRegressionDataset):
    def __init__(
        self,
        file_path: str | Path,
        *,
        target_column: str,
        feature_columns: Optional[Sequence[str]] = None,
        drop_columns: Optional[Sequence[str]] = None,
        fill_missing_with: float | int = 0.0,
        normalize: bool = False,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        return_dict: bool = False,
        sample_weight_column: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        sheet_name: str | int = 0,
        read_excel_kwargs: Optional[dict] = None,
    ) -> None:
        excel_path = Path(file_path)
        if excel_path.suffix.lower() not in {".xls", ".xlsx"}:
            raise ValueError("TabularRegressionFromExcel expects an .xls or .xlsx file.")
        if read_excel_kwargs:
            dataframe = pd.read_excel(excel_path, sheet_name=sheet_name, **read_excel_kwargs)
        else:
            dataframe = load_csv_or_excel(excel_path)

        features, targets, sample_weights, feature_names = _prepare_regression_dataframe(
            dataframe=dataframe,
            target_column=target_column,
            feature_columns=feature_columns,
            drop_columns=drop_columns,
            fill_missing_with=fill_missing_with,
            normalize=normalize,
            sample_weight_column=sample_weight_column,
            dtype=dtype,
        )

        super().__init__(
            features=features,
            targets=targets,
            transform=transform,
            target_transform=target_transform,
            return_dict=return_dict,
            sample_weights=sample_weights,
            feature_names=feature_names,
            target_name=target_column,
        )
