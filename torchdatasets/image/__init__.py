"""Image datasets: classification and segmentation.

    from torchdatasets.image import ImageSubdirDataset, ImageSegMultidirDataset
"""

from .classification import (
    BaseImageClassificationDataset,
    ImageCSVXLSXDataset,
    ImageSingleDirDataset,
    ImageSubdirDataset,
)
from .segmentation import (
    BaseImageSegmentationDataset,
    ImageSegCSVXLSXDataset,
    ImageSegMultidirDataset,
    ImageSegSingleDirDataset,
)

__all__ = [
    # classification
    "BaseImageClassificationDataset",
    "ImageCSVXLSXDataset",
    "ImageSingleDirDataset",
    "ImageSubdirDataset",
    # segmentation
    "BaseImageSegmentationDataset",
    "ImageSegCSVXLSXDataset",
    "ImageSegMultidirDataset",
    "ImageSegSingleDirDataset",
]
