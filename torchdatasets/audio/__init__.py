"""Audio datasets.

    from torchdatasets.audio import AudioSubdirDataset, AudioCSVXLSXDataset
"""

from .classification import (
    BaseAudioClassificationDataset,
    AudioCSVXLSXDataset,
    AudioSingleDirDataset,
    AudioSubdirDataset,
)

__all__ = [
    "BaseAudioClassificationDataset",
    "AudioCSVXLSXDataset",
    "AudioSingleDirDataset",
    "AudioSubdirDataset",
]
