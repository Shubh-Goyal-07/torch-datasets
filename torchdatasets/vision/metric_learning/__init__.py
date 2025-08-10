from .contrastive import make_contrastive, ContrastiveWrapper
from .triplet import make_triplet, TripletWrapper
from .few_shot import make_few_shot, FewShotWrapper

__all__ = [
    "make_contrastive", "ContrastiveWrapper",
    "make_triplet", "TripletWrapper",
    "make_few_shot", "FewShotWrapper"
]
