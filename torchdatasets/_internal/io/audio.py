import numpy as np
import soundfile as sf
import torch
import torchaudio
from pathlib import Path
from typing import Optional, Tuple

# Default audio extensions
DEFAULT_AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac", ".ogg"]


def load_audio(path: Path, sample_rate: Optional[int] = None) -> Tuple[torch.Tensor, int]:
    """Load audio file.

    Decoding uses soundfile (libsndfile) rather than torchaudio.load, which needs the
    separate torchcodec package from torchaudio 2.9 onwards.

    Args:
        path (Path): Path of the audio file.
        sample_rate (Optional[int], optional): Sample rate of the audio file. Defaults to None.

    Returns:
        Tuple[torch.Tensor, int]: Waveform (float32, shape [channels, samples]) and sample rate.
    """
    data, orig_sr = sf.read(str(path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(np.ascontiguousarray(data.T))

    if sample_rate and orig_sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_sr, sample_rate)
        waveform = resampler(waveform)

    return waveform, sample_rate or orig_sr
