from .base import BaseMetricLearningWrapper
import random

class ContrastiveWrapper(BaseMetricLearningWrapper):
    """
    Returns: ((img1, img2), label)
    label = 1 if same class, 0 if different class
    """
    def __init__(self, dataset, same_class_prob=0.5, transform_pair=None):
        super().__init__(dataset)
        self.same_class_prob = same_class_prob
        self.transform_pair = transform_pair

    def __getitem__(self, idx):
        img1, label1 = self.dataset[idx][:2]
        if random.random() < self.same_class_prob:
            # Positive
            pair_label = 1
            idx2 = random.choice(self.class_to_indices[label1])
        else:
            # Negative
            pair_label = 0
            neg_label = random.choice([l for l in self.labels if l != label1])
            idx2 = random.choice(self.class_to_indices[neg_label])

        img2, _ = self.dataset[idx2][:2]
        if self.transform_pair:
            img2 = self.transform_pair(img2)
        return (img1, img2), pair_label


def make_contrastive(dataset, same_class_prob=0.5, transform_pair=None):
    return ContrastiveWrapper(dataset, same_class_prob, transform_pair)
