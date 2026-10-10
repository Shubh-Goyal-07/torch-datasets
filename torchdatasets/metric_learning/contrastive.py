from typing import Any, Callable, Optional, Tuple

from torch.utils.data import Dataset

from torchdatasets.metric_learning.base import BaseMetricLearningWrapper


class ContrastiveWrapper(BaseMetricLearningWrapper):
    """Pairs for contrastive learning.

    Item ``i`` is ``((x_i, x_other), pair_label)``: with probability ``same_class_prob`` the other
    sample comes from the same class (``pair_label == 1``), otherwise from a different class
    (``pair_label == 0``). A positive partner is never sample ``i`` itself, unless its class has
    only one sample.
    """

    def __init__(
            self,
            dataset: Dataset,
            same_class_prob: float = 0.5,
            transform_pair: Optional[Callable] = None,
            random_state: Optional[int] = None
        ) -> None:
        """Initialize the wrapper.

        Args:
            dataset (Dataset): A dataset with single-label integer ``targets``.
            same_class_prob (float, optional): Probability that a pair is positive (same class). Defaults to 0.5.
            transform_pair (Optional[Callable], optional): Extra transform for the second sample of each pair
                (the first one only gets the dataset's own transform). Defaults to None.
            random_state (Optional[int], optional): Fix the partner drawn for each index. Defaults to None
                (new partners on every access).

        Raises:
            ValueError: If same_class_prob is outside [0, 1], or negatives are needed but the dataset has one class.
        """
        if not 0 <= same_class_prob <= 1:
            raise ValueError(f"same_class_prob must be between 0 and 1, got {same_class_prob}.")
        super().__init__(dataset, random_state)
        self.same_class_prob = same_class_prob
        self.transform_pair = transform_pair
        if same_class_prob < 1:
            self._require_two_classes()
        if same_class_prob > 0:
            self._warn_about_single_sample_classes()

    def __getitem__(self, idx: int) -> Tuple[Tuple[Any, Any], int]:
        """Get the pair for sample ``idx``.

        Args:
            idx (int): Index of the first sample of the pair.

        Returns:
            Tuple[Tuple[Any, Any], int]: ``((first, second), 1)`` for a same-class pair, ``((first, second), 0)`` otherwise.
        """
        idx = self._check_index(idx)
        rng = self._rng(idx)
        if rng.random() < self.same_class_prob:
            pair_label, other = 1, self._positive(idx, rng)
        else:
            pair_label, other = 0, self._negative(idx, rng)

        first, second = self._load(idx), self._load(other)
        if self.transform_pair:
            second = self.transform_pair(second)
        return (first, second), pair_label
