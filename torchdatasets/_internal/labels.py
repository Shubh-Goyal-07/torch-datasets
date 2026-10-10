"""Read per-sample labels from datasets, subsets and concatenations without loading any data."""

from collections import defaultdict
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Subset


def get_targets(dataset: Any) -> List[Any]:
    """Return the label of every sample of ``dataset``, in index order.

    Works for any dataset with a ``targets`` attribute (all labelled torchdatasets classes,
    torchvision's ImageFolder/MNIST, ...), and for ``Subset``/``ConcatDataset`` built on them.

    Args:
        dataset (Any): The dataset.

    Raises:
        TypeError: If the labels can't be read (e.g. a segmentation dataset, whose labels are masks).

    Returns:
        List[Any]: One label per sample.
    """
    if isinstance(dataset, Subset):
        parent = get_targets(dataset.dataset)
        return [parent[int(i)] for i in dataset.indices]
    if isinstance(dataset, ConcatDataset):
        return [target for part in dataset.datasets for target in get_targets(part)]

    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise TypeError(
            f"{type(dataset).__name__} has no `targets`, so its labels can't be read without loading "
            "the data. Use a classification or tabular dataset (segmentation datasets have none)."
        )
    if isinstance(targets, torch.Tensor):
        targets = targets.tolist()
    targets = list(targets)
    if len(targets) != len(dataset):
        raise ValueError(f"{type(dataset).__name__}.targets has {len(targets)} entries for {len(dataset)} samples.")
    return targets


def require_class_labels(targets: Sequence[Any], purpose: str) -> List[int]:
    """Check that every target is a single integer class label.

    Args:
        targets (Sequence[Any]): Labels from :func:`get_targets`.
        purpose (str): What needs the labels, for the error message (e.g. "stratify").

    Raises:
        ValueError: If a target is a list (multi-label), a float (regression) or anything else.

    Returns:
        List[int]: The labels as plain ints.
    """
    for target in targets:
        if isinstance(target, (bool, np.bool_)) or not isinstance(target, (int, np.integer)):
            raise ValueError(
                f"{purpose} needs one integer class label per sample, but got {target!r} "
                f"({type(target).__name__}). Multi-label and regression targets are not supported."
            )
    return [int(target) for target in targets]


def indices_by_class(labels: Sequence[int]) -> Dict[int, List[int]]:
    """Group sample indices by class label.

    Args:
        labels (Sequence[int]): One class label per sample.

    Returns:
        Dict[int, List[int]]: Class label -> sorted indices of its samples, in sorted label order.
    """
    groups: Dict[int, List[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        groups[label].append(index)
    return {label: groups[label] for label in sorted(groups)}
