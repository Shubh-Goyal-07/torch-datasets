from typing import Any, Callable, List, Optional, Tuple

from torch.utils.data import Dataset

from torchdatasets.metric_learning.base import BaseMetricLearningWrapper


class FewShotWrapper(BaseMetricLearningWrapper):
    """N-way K-shot episodes for few-shot learning.

    Each item is an episode ``(support_set, query_set)``: ``n_way`` classes are drawn, and from each
    class ``k_shot`` support and ``q_query`` query samples (all different). Both sets are lists of
    ``(sample, label)``, grouped by class. Labels are the class's position in the episode
    (``0..n_way-1``), or the dataset's own class index with ``relabel=False``.

    Only classes with at least ``k_shot + q_query`` samples take part. Episodes are lists, so use
    ``DataLoader(wrapper, batch_size=None)`` and stack the samples in your training step.
    """

    def __init__(
            self,
            dataset: Dataset,
            n_way: int = 5,
            k_shot: int = 1,
            q_query: int = 15,
            transform_support: Optional[Callable] = None,
            transform_query: Optional[Callable] = None,
            n_episodes: Optional[int] = None,
            relabel: bool = True,
            random_state: Optional[int] = None
        ) -> None:
        """Initialize the wrapper.

        Args:
            dataset (Dataset): A dataset with single-label integer ``targets``.
            n_way (int, optional): Classes per episode. Defaults to 5.
            k_shot (int, optional): Support samples per class. Defaults to 1.
            q_query (int, optional): Query samples per class. Defaults to 15.
            transform_support (Optional[Callable], optional): Extra transform for support samples. Defaults to None.
            transform_query (Optional[Callable], optional): Extra transform for query samples. Defaults to None.
            n_episodes (Optional[int], optional): Episodes per pass (the wrapper's length). Defaults to None
                (as many as the dataset has samples).
            relabel (bool, optional): Label samples 0..n_way-1 by their class's position in the episode;
                False keeps the dataset's class indices. Defaults to True.
            random_state (Optional[int], optional): Fix the episode drawn for each index (e.g. a fixed set of
                evaluation episodes). Defaults to None (new episodes on every access).

        Raises:
            ValueError: If a size is invalid, or fewer than n_way classes have k_shot + q_query samples.
        """
        if n_way < 1 or k_shot < 1 or q_query < 0:
            raise ValueError(f"Need n_way >= 1, k_shot >= 1 and q_query >= 0, got {n_way}, {k_shot}, {q_query}.")
        if n_episodes is not None and n_episodes < 1:
            raise ValueError(f"n_episodes must be at least 1, got {n_episodes}.")
        super().__init__(dataset, random_state)
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.transform_support = transform_support
        self.transform_query = transform_query
        self.n_episodes = n_episodes
        self.relabel = relabel

        per_class = k_shot + q_query
        self.episode_classes: List[int] = [
            label for label, members in self.class_to_indices.items() if len(members) >= per_class
        ]
        if len(self.episode_classes) < n_way:
            raise ValueError(
                f"A {n_way}-way episode needs {n_way} classes with at least {per_class} samples each "
                f"(k_shot + q_query), but only {len(self.episode_classes)} of the dataset's "
                f"{len(self.labels)} classes have that many."
            )

    def __len__(self) -> int:
        """Return the number of episodes per pass."""
        return self.n_episodes if self.n_episodes is not None else len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[List[Tuple[Any, int]], List[Tuple[Any, int]]]:
        """Draw an episode.

        Args:
            idx (int): Episode index (only matters with random_state, which fixes each episode).

        Returns:
            Tuple[List[Tuple[Any, int]], List[Tuple[Any, int]]]: (support_set, query_set).
        """
        idx = self._check_index(idx)
        rng = self._rng(idx)
        support_set: List[Tuple[Any, int]] = []
        query_set: List[Tuple[Any, int]] = []

        for episode_label, cls in enumerate(rng.sample(self.episode_classes, self.n_way)):
            label = episode_label if self.relabel else cls
            picked = rng.sample(self.class_to_indices[cls], self.k_shot + self.q_query)
            for i in picked[:self.k_shot]:
                sample = self._load(i)
                support_set.append((self.transform_support(sample) if self.transform_support else sample, label))
            for i in picked[self.k_shot:]:
                sample = self._load(i)
                query_set.append((self.transform_query(sample) if self.transform_query else sample, label))

        return support_set, query_set
