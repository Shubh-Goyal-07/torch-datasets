from pathlib import Path
from typing import Callable, Optional, Sequence

import pandas as pd
import torch

from torchdatasets._internal.io.common import load_csv_or_excel
from torchdatasets.tabular.regression.base import BaseTabularRegressionDataset


class TabularRegressionFromCSV(BaseTabularRegressionDataset):
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
        read_csv_kwargs: Optional[dict] = None,
    ) -> None:
        csv_path = Path(file_path)
        if read_csv_kwargs:
            dataframe = pd.read_csv(csv_path, **read_csv_kwargs)
        else:
            dataframe = load_csv_or_excel(csv_path)

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


def _prepare_regression_dataframe(
    *,
    dataframe: pd.DataFrame,
    target_column: str,
    feature_columns: Optional[Sequence[str]],
    drop_columns: Optional[Sequence[str]],
    fill_missing_with: float | int,
    normalize: bool,
    sample_weight_column: Optional[str],
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], list[str]]:
    if target_column not in dataframe.columns:
        raise ValueError(f"target_column '{target_column}' not found in dataframe.")
    if dataframe.empty:
        raise ValueError("Input dataframe is empty.")

    if drop_columns:
        existing_drops = [column for column in drop_columns if column in dataframe.columns]
        dataframe = dataframe.drop(columns=existing_drops)

    if feature_columns is None:
        reserved_columns = {target_column}
        if sample_weight_column:
            reserved_columns.add(sample_weight_column)
        selected_features = [column for column in dataframe.columns if column not in reserved_columns]
    else:
        selected_features = [column for column in feature_columns if column in dataframe.columns]

    if not selected_features:
        raise ValueError("No valid feature columns found for regression dataset.")

    working_frame = dataframe.copy()
    if sample_weight_column and sample_weight_column not in working_frame.columns:
        raise ValueError(f"sample_weight_column '{sample_weight_column}' not found in dataframe.")

    # Feature: automatic categorical encoding for non-numeric feature columns.
    for column in selected_features:
        if pd.api.types.is_numeric_dtype(working_frame[column]):
            continue
        working_frame[column] = pd.factorize(working_frame[column].astype(str))[0]

    feature_frame = working_frame[selected_features].fillna(fill_missing_with)
    target_series = pd.to_numeric(working_frame[target_column], errors="coerce").fillna(fill_missing_with)

    if normalize:
        # Feature: optional z-score normalization for regression feature tensors.
        means = feature_frame.mean()
        stds = feature_frame.std().replace(0, 1)
        feature_frame = (feature_frame - means) / stds

    features_tensor = torch.tensor(feature_frame.to_numpy(), dtype=dtype)
    targets_tensor = torch.tensor(target_series.to_numpy(), dtype=dtype)

    sample_weights_tensor = None
    if sample_weight_column:
        weights = pd.to_numeric(working_frame[sample_weight_column], errors="coerce").fillna(1.0).to_numpy()
        sample_weights_tensor = torch.tensor(weights, dtype=torch.float32)

    return features_tensor, targets_tensor, sample_weights_tensor, selected_features
