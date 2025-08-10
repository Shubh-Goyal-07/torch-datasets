from .wrappers import BaseMetricLearningWrapper
import random
from collections import defaultdict

class FewShotWrapper(BaseMetricLearningWrapper):
    """
    Few-shot episodic dataset.
    Each __getitem__ returns an episode: (support_set, query_set)
    support_set: list of (image, label)
    query_set: list of (image, label)
    """
    def __init__(self, dataset, n_way=5, k_shot=1, q_query=15, transform_support=None, transform_query=None):
        super().__init__(dataset)
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.transform_support = transform_support
        self.transform_query = transform_query

    def __getitem__(self, _):
        # Sample N classes
        selected_classes = random.sample(self.labels, self.n_way)
        support_set, query_set = [], []

        for cls in selected_classes:
            indices = self.class_to_indices[cls]
            selected_indices = random.sample(indices, self.k_shot + self.q_query)
            support_idx = selected_indices[:self.k_shot]
            query_idx = selected_indices[self.k_shot:]

            for idx in support_idx:
                img, label = self.dataset[idx][:2]
                if self.transform_support:
                    img = self.transform_support(img)
                support_set.append((img, label))

            for idx in query_idx:
                img, label = self.dataset[idx][:2]
                if self.transform_query:
                    img = self.transform_query(img)
                query_set.append((img, label))

        return support_set, query_set

    def __len__(self):
        # Episodic datasets are often "infinite", but we return len(dataset) for compatibility
        return len(self.dataset)


def make_few_shot(dataset, n_way=5, k_shot=1, q_query=15, transform_support=None, transform_query=None):
    return FewShotWrapper(dataset, n_way, k_shot, q_query, transform_support, transform_query)
