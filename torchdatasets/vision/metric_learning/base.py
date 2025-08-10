import random
from torch.utils.data import Dataset

class BaseMetricLearningWrapper(Dataset):
    """
    Base wrapper for metric-learning style datasets.
    Takes an existing dataset and changes its sampling strategy.
    """
    def __init__(self, dataset):
        self.dataset = dataset
        # Precompute mapping: class_id -> indices
        self.class_to_indices = {}
        for idx, (_, label) in enumerate(dataset.samples):
            self.class_to_indices.setdefault(label, []).append(idx)
        self.labels = list(self.class_to_indices.keys())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        raise NotImplementedError