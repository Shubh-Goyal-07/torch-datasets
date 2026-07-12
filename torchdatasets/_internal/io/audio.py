import torch
import torchaudio
from pathlib import Path
from typing import Optional, Tuple

# Default audio extensions
DEFAULT_AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac", ".ogg"]


def load_audio(path: Path, sample_rate: Optional[int] = None) -> Tuple[torch.Tensor, int]:
    """Load audio file.

    Args:
        path (Path): Path of the audio file.
        sample_rate (Optional[int], optional): Sample rate of the audio file. Defaults to None.

    Returns:
        Tuple[torch.Tensor, int]: Waveform and sample rate.
    """
    waveform, orig_sr = torchaudio.load(path)

    if sample_rate and orig_sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_sr, sample_rate)
        waveform = resampler(waveform)

    return waveform, sample_rate or orig_sr
