from typing import Any, Callable, Optional, Sequence

import torch
from torch.utils.data import Dataset


class BaseTabularRegressionDataset(Dataset):
    def __init__(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        *,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        return_dict: bool = False,
        sample_weights: Optional[torch.Tensor] = None,
        feature_names: Optional[Sequence[str]] = None,
        target_name: Optional[str] = None,
    ) -> None:
        self.features = features
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        self.return_dict = return_dict
        self.sample_weights = sample_weights
        self.feature_names = list(feature_names) if feature_names is not None else []
        self.target_name = target_name

        self._validate_inputs()

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, idx: int) -> Any:
        feature_row = self.features[idx]
        target_value = self.targets[idx]

        if self.transform:
            feature_row = self.transform(feature_row)
        if self.target_transform:
            target_value = self.target_transform(target_value)

        if not self.return_dict:
            if self.sample_weights is None:
                return feature_row, target_value
            return feature_row, target_value, self.sample_weights[idx]

        result = {
            "features": feature_row,
            "target": target_value,
            "index": idx,
        }
        if self.sample_weights is not None:
            result["sample_weight"] = self.sample_weights[idx]
        if self.feature_names:
            result["feature_names"] = self.feature_names
        if self.target_name:
            result["target_name"] = self.target_name
        return result

    def _validate_inputs(self) -> None:
        if self.features.ndim != 2:
            raise ValueError("features must be a 2D tensor with shape [n_samples, n_features].")
        if self.targets.ndim not in (1, 2):
            raise ValueError("targets must be 1D or 2D tensor for regression.")
        if len(self.features) != len(self.targets):
            raise ValueError("features and targets must have the same number of samples.")
        if self.sample_weights is not None and len(self.sample_weights) != len(self.targets):
            raise ValueError("sample_weights must match number of samples.")
