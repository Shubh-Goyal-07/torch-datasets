"""Image datasets: classification and segmentation.

    from torchdatasets.image import ImageSubdirDataset, ImageSegMultiDirDataset
"""

from .classification import (
    BaseImageClassificationDataset,
    ImageCSVDataset,
    ImageSingleDirDataset,
    ImageSubdirDataset,
)
from .segmentation import (
    BaseImageSegmentationDataset,
    ImageSegCSVDataset,
    ImageSegMultiDirDataset,
    ImageSegSingleDirDataset,
)

__all__ = [
    # classification
    "BaseImageClassificationDataset",
    "ImageCSVDataset",
    "ImageSingleDirDataset",
    "ImageSubdirDataset",
    # segmentation
    "BaseImageSegmentationDataset",
    "ImageSegCSVDataset",
    "ImageSegMultiDirDataset",
    "ImageSegSingleDirDataset",
]
