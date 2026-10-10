from typing import Any, Callable, Optional, Tuple

from torch.utils.data import Dataset

from torchdatasets.metric_learning.base import BaseMetricLearningWrapper


class TripletWrapper(BaseMetricLearningWrapper):
    """Triplets for triplet-loss training.

    Item ``i`` is ``(anchor, positive, negative)``: the anchor is sample ``i``, the positive another
    sample of its class (the anchor itself only if the class has one sample), and the negative a
    sample of a different class.
    """

    def __init__(
            self,
            dataset: Dataset,
            *,
            transform_pos: Optional[Callable] = None,
            transform_neg: Optional[Callable] = None,
            random_state: Optional[int] = None
        ) -> None:
        """Initialize the wrapper.

        Args:
            dataset (Dataset): A dataset with single-label integer ``targets`` and at least 2 classes.
            transform_pos (Optional[Callable], optional): Extra transform for the positive. Defaults to None.
            transform_neg (Optional[Callable], optional): Extra transform for the negative. Defaults to None.
            random_state (Optional[int], optional): Fix the positive and negative drawn for each index.
                Defaults to None (new ones on every access).

        Raises:
            ValueError: If the dataset has fewer than 2 classes.
        """
        super().__init__(dataset, random_state=random_state)
        self.transform_pos = transform_pos
        self.transform_neg = transform_neg
        self._require_two_classes()
        self._warn_about_single_sample_classes()

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any]:
        """Get the triplet anchored at sample ``idx``.

        Args:
            idx (int): Index of the anchor.

        Returns:
            Tuple[Any, Any, Any]: (anchor, positive, negative).
        """
        idx = self._check_index(idx)
        rng = self._rng(idx)
        positive_idx = self._positive(idx, rng)
        negative_idx = self._negative(idx, rng)

        anchor, positive, negative = self._load(idx), self._load(positive_idx), self._load(negative_idx)
        if self.transform_pos:
            positive = self.transform_pos(positive)
        if self.transform_neg:
            negative = self.transform_neg(negative)
        return anchor, positive, negative
