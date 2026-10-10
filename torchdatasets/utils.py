"""Splitting datasets into train/validation/test parts.

    from torchdatasets.utils import train_val_split

    train_ds, val_ds = train_val_split(dataset, val_ratio=0.2, stratify=True,
                                       train_transform=train_tf, val_transform=val_tf)

Each part can have its own transform. Tabular datasets are rebuilt so their preprocessing is
fitted on the training rows only (the other parts reuse it through ``fit_from``).
"""

import copy
import math
import sys
import warnings
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
from torch.utils.data import Dataset, Subset

from torchdatasets._internal.labels import get_targets, require_class_labels

__all__ = ["TransformSubset", "train_val_split", "train_val_test_split"]


class TransformSubset(Subset):
    """A ``Subset`` whose samples can use their own transform.

    With a plain ``torch.utils.data.Subset``, samples always go through the wrapped dataset's
    transform, and setting ``subset.transform`` does nothing. ``TransformSubset`` keeps a shallow
    copy of the dataset with ``transform`` replaced instead: the copy shares the dataset's samples,
    and the dataset's own ``__getitem__`` applies the transform as usual (so segmentation datasets
    still call ``transform(image, mask)``).

    The subset's transform replaces the dataset's (it isn't added on top). Setting
    ``subset.transform`` later works too, and ``subset.transform = None`` removes it.

    ``targets`` gives the labels of the kept samples, in subset order.
    """

    def __init__(self, dataset: Dataset, indices: Sequence[int], *, transform: Optional[Callable] = None) -> None:
        """Initialize the subset.

        Args:
            dataset (Dataset): The dataset to take samples from. It needs a ``transform`` attribute
                if a transform is given.
            indices (Sequence[int]): Indices of the samples to keep.
            transform (Optional[Callable], optional): Transform for these samples. None keeps the
                dataset's own transform. Defaults to None.
        """
        super().__init__(dataset, [int(i) for i in indices])
        self._owns_dataset = False
        if transform is not None:
            self.transform = transform

    def __copy__(self) -> "TransformSubset":
        # a copy shares the dataset it wraps, so it must copy it again before changing its transform
        clone = type(self).__new__(type(self))
        clone.__dict__.update(self.__dict__)
        clone._owns_dataset = False
        return clone

    @property
    def transform(self) -> Optional[Callable]:
        """The transform applied to this subset's samples."""
        return getattr(self.dataset, "transform", None)

    @transform.setter
    def transform(self, transform: Optional[Callable]) -> None:
        if not hasattr(self.dataset, "transform"):
            raise TypeError(
                f"{type(self.dataset).__name__} has no `transform` attribute, so a subset of it can't "
                "have its own transform."
            )
        if not self._owns_dataset:
            # copy, so the other subsets (and the original dataset) keep their transform
            self.dataset = copy.copy(self.dataset)
            self._owns_dataset = True
        self.dataset.transform = transform

    @property
    def targets(self) -> List[Any]:
        """Label of every sample in the subset, in subset order (needs a dataset with labels)."""
        return get_targets(self)


def train_val_split(
    dataset: Dataset,
    *,
    val_ratio: float = 0.2,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    stratify: bool = False,
    random_state: Optional[int] = 42,
) -> Tuple[Dataset, Dataset]:
    """Split a dataset into a training and a validation part.

    Each part is a :class:`TransformSubset` with its own transform. A tabular dataset is instead
    rebuilt into two ``TabularDataFrameDataset`` objects: the training part refits the feature
    preprocessing (fill values, category encoding, scaling) on its own rows, and the validation
    part reuses it with ``fit_from``, so nothing about the validation rows leaks into training.
    The class list comes from the whole dataset, so every part uses the same class indices.

    A part's transform replaces the dataset's own transform; leaving it as None keeps the
    dataset's. To drop the transform of a part afterwards, set ``part.transform = None``.

    If a class ends up with no training samples (possible without ``stratify``), a warning says so.

    Args:
        dataset (Dataset): The dataset to split.
        val_ratio (float, optional): Fraction of samples for validation (0 < val_ratio < 1;
            rounded up to whole samples). Defaults to 0.2.
        train_transform (Optional[Callable], optional): Transform for the training samples.
            None keeps the dataset's own transform. Defaults to None.
        val_transform (Optional[Callable], optional): Transform for the validation samples.
            None keeps the dataset's own transform. Defaults to None.
        stratify (bool, optional): Keep the class proportions the same in both parts. Needs one
            integer class label per sample (``dataset.targets``). Defaults to False.
        random_state (Optional[int], optional): Seed for the shuffle; None gives a different split
            every call. Defaults to 42.

    Raises:
        ValueError: If val_ratio is out of range, a part would be empty, or stratify is used with
            labels that aren't single integer classes.
        TypeError: If stratify is used on a dataset without ``targets``.

    Returns:
        Tuple[Dataset, Dataset]: (train, val).
    """
    train, val = _split(dataset, [val_ratio], [train_transform, val_transform], stratify, random_state)
    return train, val


def train_val_test_split(
    dataset: Dataset,
    *,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    test_transform: Optional[Callable] = None,
    stratify: bool = False,
    random_state: Optional[int] = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    """Split a dataset into training, validation and test parts.

    Works like :func:`train_val_split` (transforms, class list, warnings). For tabular datasets,
    the validation and test parts both reuse the training part's preprocessing.

    Args:
        dataset (Dataset): The dataset to split.
        val_ratio (float, optional): Fraction of samples for validation. Defaults to 0.1.
        test_ratio (float, optional): Fraction of samples for testing. Defaults to 0.1.
        train_transform (Optional[Callable], optional): Transform for the training samples. Defaults to None.
        val_transform (Optional[Callable], optional): Transform for the validation samples. Defaults to None.
        test_transform (Optional[Callable], optional): Transform for the test samples. Defaults to None.
        stratify (bool, optional): Keep the class proportions the same in every part. Defaults to False.
        random_state (Optional[int], optional): Seed for the shuffle. Defaults to 42.

    Raises:
        ValueError: If the ratios are out of range (each in (0, 1), together below 1), a part would
            be empty, or stratify is used with labels that aren't single integer classes.
        TypeError: If stratify is used on a dataset without ``targets``.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: (train, val, test).
    """
    train, val, test = _split(
        dataset, [val_ratio, test_ratio], [train_transform, val_transform, test_transform], stratify, random_state
    )
    return train, val, test


def _split(
    dataset: Dataset,
    ratios: List[float],
    transforms: List[Optional[Callable]],
    stratify: bool,
    random_state: Optional[int],
) -> List[Dataset]:
    """Split into a training part plus one part per ratio, each with its transform."""
    labels = require_class_labels(get_targets(dataset), "stratify") if stratify else _class_labels(dataset)
    parts = _split_indices(len(dataset), ratios, labels if stratify else None, random_state)
    if labels is not None:
        _warn_about_classes_missing_from_training(dataset, labels, parts[0])

    tabular = sys.modules.get("torchdatasets.tabular.from_dataframe")
    if tabular is not None and isinstance(dataset, tabular.TabularDataFrameDataset):
        return _split_tabular(dataset, parts, transforms)
    return [TransformSubset(dataset, indices, transform=transform) for indices, transform in zip(parts, transforms)]


def _class_labels(dataset: Dataset) -> Optional[List[int]]:
    """The dataset's single-label class targets, or None if it has none (or they aren't classes)."""
    try:
        return require_class_labels(get_targets(dataset), "")
    except (TypeError, ValueError):
        return None


def _warn_about_classes_missing_from_training(dataset: Dataset, labels: List[int], train_indices: List[int]) -> None:
    missing = sorted(set(labels) - {labels[i] for i in train_indices})
    if missing:
        names = getattr(dataset, "idx_to_class", None) or {}
        warnings.warn(
            f"{len(missing)} class(es) have no samples in the training part "
            f"(e.g. {[names.get(label, label) for label in missing[:3]]}); "
            "pass stratify=True to keep every class in every part.",
            stacklevel=4,
        )


def _split_indices(
    n_samples: int,
    ratios: List[float],
    labels: Optional[List[int]],
    random_state: Optional[int],
) -> List[List[int]]:
    """Shuffle (and optionally stratify) sample indices into a training part and one part per ratio.

    Returns:
        List[List[int]]: Sorted indices of the training part, then of each ratio's part.
    """
    from sklearn.model_selection import train_test_split

    for ratio in ratios:
        if not 0 < ratio < 1:
            raise ValueError(f"Split ratios must be between 0 and 1 (exclusive), got {ratio}.")
    if sum(ratios) >= 1:
        raise ValueError(f"The split ratios {ratios} add up to {sum(ratios):g}; leave something for training.")

    # whole-sample sizes (rounded up, like scikit-learn), computed up front so that splitting
    # parts off one after another doesn't change them
    sizes = [math.ceil(round(ratio * n_samples, 6)) for ratio in ratios]
    if n_samples - sum(sizes) < 1:
        raise ValueError(f"A dataset of {n_samples} samples is too small for the ratios {ratios}.")

    rest = np.arange(n_samples)
    label_array = None if labels is None else np.asarray(labels)
    parts: List[List[int]] = []
    for size in reversed(sizes):
        rest, part = train_test_split(
            rest,
            test_size=size,
            random_state=random_state,
            stratify=None if label_array is None else label_array[rest],
        )
        parts.insert(0, sorted(int(i) for i in part))
    return [sorted(int(i) for i in rest)] + parts


def _split_tabular(dataset: Any, parts: List[List[int]], transforms: List[Optional[Callable]]) -> List[Dataset]:
    """Rebuild a tabular dataset's rows into parts, fitting the preprocessing on the first one."""
    from torchdatasets.tabular.from_dataframe import TabularDataFrameDataset

    # the dataset's rows are its dataframe's rows, minus duplicates if it dropped them
    frame = dataset.dataframe.drop_duplicates() if dataset.drop_duplicates else dataset.dataframe
    if len(frame) != len(dataset):
        raise RuntimeError(f"Expected {len(dataset)} rows in the dataset's dataframe, found {len(frame)}.")

    def pick(transform: Optional[Callable]) -> Optional[Callable]:
        return dataset.transform if transform is None else transform

    # keep the whole dataset's class indices in every part (a rare class may miss the training rows)
    single_class_column = dataset.task == "classification" and len(dataset.target_cols) == 1 and not dataset.n_bins
    label_map = dict(dataset.class_to_idx) if single_class_column else None

    train = TabularDataFrameDataset(
        frame.iloc[parts[0]],
        target_cols=dataset.target_cols,
        feature_cols=dataset.feature_cols,
        task=dataset.task,
        handle_missing=dataset.handle_missing,
        fill_missing=dataset.fill_missing,
        scaling_type=dataset.scaling_type,
        scaling_features=dataset.scaling_features or None,
        encode_categorical=dataset.encode_categorical,
        drop_duplicates=dataset.drop_duplicates,
        n_bins=dataset.n_bins,
        transform=pick(transforms[0]),
        target_transform=dataset.target_transform,
        label_map=label_map,
    )
    return [train] + [
        TabularDataFrameDataset(
            frame.iloc[rows],
            fit_from=train,
            drop_duplicates=dataset.drop_duplicates,
            transform=pick(transform),
            target_transform=dataset.target_transform,
        )
        for rows, transform in zip(parts[1:], transforms[1:])
    ]
