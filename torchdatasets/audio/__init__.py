"""Audio datasets.

    from torchdatasets.audio import AudioSubdirDataset, AudioCSVDataset
"""

from .classification import (
    BaseAudioClassificationDataset,
    AudioCSVDataset,
    AudioSingleDirDataset,
    AudioSubdirDataset,
)

__all__ = [
    "BaseAudioClassificationDataset",
    "AudioCSVDataset",
    "AudioSingleDirDataset",
    "AudioSubdirDataset",
]
