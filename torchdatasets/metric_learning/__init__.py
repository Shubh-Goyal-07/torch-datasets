"""Wrappers that turn a labelled dataset into pairs, triplets or few-shot episodes.

    from torchdatasets.metric_learning import TripletWrapper

They work with any dataset that has single-label ``targets`` (image, audio or tabular).
"""

from .base import BaseMetricLearningWrapper
from .contrastive import ContrastiveWrapper
from .few_shot import FewShotWrapper
from .triplet import TripletWrapper

__all__ = [
    "BaseMetricLearningWrapper",
    "ContrastiveWrapper",
    "FewShotWrapper",
    "TripletWrapper",
]
