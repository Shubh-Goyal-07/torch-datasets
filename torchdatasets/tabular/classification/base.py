from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

import torch
from torch.utils.data import Dataset


class BaseTabularClassificationDataset(Dataset):
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
        class_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.features = features
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        self.return_dict = return_dict
        self.sample_weights = sample_weights
        self.feature_names = list(feature_names) if feature_names is not None else []
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}
        self.class_count: Dict[int, int] = {}
        self.class_names = list(class_names) if class_names is not None else []

        self._validate_inputs()
        self._create_metadata()

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
        return result

    def _validate_inputs(self) -> None:
        if self.features.ndim != 2:
            raise ValueError("features must be a 2D tensor with shape [n_samples, n_features].")
        if self.targets.ndim != 1:
            raise ValueError("targets must be a 1D tensor for single-label classification.")
        if len(self.features) != len(self.targets):
            raise ValueError("features and targets must have the same number of samples.")
        if self.sample_weights is not None and len(self.sample_weights) != len(self.targets):
            raise ValueError("sample_weights must match number of samples.")

    def _create_metadata(self) -> None:
        counts = defaultdict(int)
        for label in self.targets.tolist():
            counts[int(label)] += 1
        self.class_count = dict(counts)

        if not self.class_names:
            unique_labels = sorted(int(label) for label in set(self.targets.tolist()))
            self.class_names = [str(label) for label in unique_labels]

        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}
        self.idx_to_class = {idx: class_name for class_name, idx in self.class_to_idx.items()}

    def get_class_distribution(self) -> Dict[int, float]:
        total = max(1, len(self.targets))
        return {label: count / total for label, count in self.class_count.items()}

    @staticmethod
    def build_inverse_frequency_weights(targets: Iterable[int]) -> torch.Tensor:
        counts = defaultdict(int)
        for target in targets:
            counts[int(target)] += 1

        weights = [1.0 / counts[int(target)] for target in targets]
        return torch.tensor(weights, dtype=torch.float32)