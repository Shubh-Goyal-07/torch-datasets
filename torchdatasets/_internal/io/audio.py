import torch
import torchaudio
from pathlib import Path
from typing import Optional, Tuple

DEFAULT_AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac", ".ogg"]


def load_audio(path: Path, sample_rate: Optional[int] = None) -> Tuple[torch.Tensor, int]:
    waveform, orig_sr = torchaudio.load(path)

    if sample_rate and orig_sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_sr, sample_rate)
        waveform = resampler(waveform)

    return waveform, sample_rate or orig_sr
