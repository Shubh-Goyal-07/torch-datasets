from .base import BaseMetricLearningWrapper
import random

class TripletWrapper(BaseMetricLearningWrapper):
    """
    Returns: (anchor, positive, negative)
    """
    def __init__(self, dataset, transform_pos=None, transform_neg=None):
        super().__init__(dataset)
        self.transform_pos = transform_pos
        self.transform_neg = transform_neg

    def __getitem__(self, idx):
        anchor, label_anchor = self.dataset[idx][:2]

        # Positive from same class
        idx_pos = random.choice(self.class_to_indices[label_anchor])
        positive, _ = self.dataset[idx_pos][:2]
        if self.transform_pos:
            positive = self.transform_pos(positive)

        # Negative from different class
        neg_label = random.choice([l for l in self.labels if l != label_anchor])
        idx_neg = random.choice(self.class_to_indices[neg_label])
        negative, _ = self.dataset[idx_neg][:2]
        if self.transform_neg:
            negative = self.transform_neg(negative)

        return anchor, positive, negative


def make_triplet(dataset, transform_pos=None, transform_neg=None):
    return TripletWrapper(dataset, transform_pos, transform_neg)
