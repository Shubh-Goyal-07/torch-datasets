import random
import warnings
from typing import Any, Dict, List, Optional, Union

from torch.utils.data import Dataset

from torchdatasets._internal.labels import get_targets, indices_by_class, require_class_labels


class BaseMetricLearningWrapper(Dataset):
    """Base class for wrappers that sample pairs, triplets or episodes from a labelled dataset.

    Works with any dataset that exposes ``targets`` (every torchdatasets classification and
    tabular classification dataset, their splits, ``Subset``/``ConcatDataset`` of them, and
    torchvision's ImageFolder/MNIST-style datasets). Labels are read once, without loading data;
    each label must be a single integer class.

    Randomness: with ``random_state=None`` every access draws new partners from Python's ``random``
    module, which PyTorch reseeds in each DataLoader worker and epoch. With an integer, index ``i``
    always gets the same partners (useful for a fixed validation set), in any worker.
    """

    def __init__(self, dataset: Dataset, *, random_state: Optional[int] = None) -> None:
        """Initialize the wrapper.

        Args:
            dataset (Dataset): A dataset with single-label integer ``targets``.
            random_state (Optional[int], optional): Fix the partners drawn for each index. Defaults to None.

        Raises:
            TypeError: If the dataset has no ``targets``.
            ValueError: If the targets aren't single integer class labels, or the dataset is empty.
        """
        self.dataset = dataset
        self.random_state = random_state
        self._targets: List[int] = require_class_labels(get_targets(dataset), type(self).__name__)
        if not self._targets:
            raise ValueError(f"{type(self).__name__} needs a non-empty dataset.")

        self.class_to_indices: Dict[int, List[int]] = indices_by_class(self._targets)
        self.labels: List[int] = list(self.class_to_indices)

        # where each sample and each class sit in the lists above, for O(1) "anything but this one" draws
        self._position = [0] * len(self._targets)
        for members in self.class_to_indices.values():
            for position, index in enumerate(members):
                self._position[index] = position
        self._label_position = {label: position for position, label in enumerate(self.labels)}

    def __len__(self) -> int:
        """Return the number of samples in the wrapped dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Any:
        raise NotImplementedError("Subclasses must implement __getitem__")

    def _check_index(self, idx: int) -> int:
        """Turn a possibly negative index into a position, raising IndexError when out of range."""
        size = len(self)
        position = idx + size if idx < 0 else idx
        if not 0 <= position < size:
            raise IndexError(f"Index {idx} is out of range for {type(self).__name__} of length {size}.")
        return position

    def _rng(self, idx: int) -> Union[random.Random, Any]:
        """Random source for one access: the shared ``random`` module, or one fixed by (random_state, idx)."""
        if self.random_state is None:
            return random
        return random.Random(f"{self.random_state}-{idx}")

    def _load(self, idx: int) -> Any:
        """The sample at ``idx`` of the wrapped dataset, without its label (or path)."""
        return self.dataset[idx][0]

    def _positive(self, idx: int, rng: Any) -> int:
        """Another sample of ``idx``'s class; ``idx`` itself only if its class has one sample."""
        members = self.class_to_indices[self._targets[idx]]
        if len(members) == 1:
            return idx
        position = rng.randrange(len(members) - 1)
        if position >= self._position[idx]:
            position += 1  # skip the anchor itself
        return members[position]

    def _negative(self, idx: int, rng: Any) -> int:
        """A sample of another class: the class is drawn uniformly, then a sample within it."""
        position = rng.randrange(len(self.labels) - 1)
        if position >= self._label_position[self._targets[idx]]:
            position += 1  # skip the anchor's class
        return rng.choice(self.class_to_indices[self.labels[position]])

    def _require_two_classes(self) -> None:
        if len(self.labels) < 2:
            raise ValueError(
                f"{type(self).__name__} needs at least 2 classes to draw negatives; "
                f"the dataset has {len(self.labels)}."
            )

    def _warn_about_single_sample_classes(self) -> None:
        single = [label for label, members in self.class_to_indices.items() if len(members) == 1]
        if single:
            warnings.warn(
                f"{len(single)} class(es) have a single sample (e.g. class {single[0]}), so their "
                "positive is the anchor itself; only a transform makes the two differ.",
                stacklevel=3,
            )
