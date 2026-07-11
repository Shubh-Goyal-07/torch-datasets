"""Tests for torchdatasets._internal.io.audio — audio loading utility."""

import torch
import pytest

from torchdatasets._internal.io.audio import DEFAULT_AUDIO_EXTENSIONS, load_audio

# Guard: torchaudio is needed for both the source module and these tests.
torchaudio = pytest.importorskip("torchaudio")


def _create_wav(path, sample_rate=16000, duration_ms=100, channels=1):
    """Helper to write a tiny WAV file."""
    n_samples = int(sample_rate * duration_ms / 1000)
    waveform = torch.randn(channels, n_samples)
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), waveform, sample_rate)
    return path, sample_rate, n_samples


class TestLoadAudio:
    """Tests for load_audio."""

    def test_returns_tensor_and_sample_rate(self, tmp_path):
        path, sr, _ = _create_wav(tmp_path / "test.wav")
        waveform, returned_sr = load_audio(path)
        assert isinstance(waveform, torch.Tensor)
        assert returned_sr == sr

    def test_waveform_shape(self, tmp_path):
        channels = 2
        path, sr, n_samples = _create_wav(
            tmp_path / "stereo.wav", channels=channels
        )
        waveform, _ = load_audio(path)
        assert waveform.shape[0] == channels

    def test_resampling(self, tmp_path):
        orig_sr = 16000
        target_sr = 8000
        path, _, _ = _create_wav(
            tmp_path / "test.wav", sample_rate=orig_sr, duration_ms=200
        )
        waveform, returned_sr = load_audio(path, sample_rate=target_sr)
        assert returned_sr == target_sr
        # Resampled waveform should have roughly half the samples
        expected_samples = int(orig_sr * 200 / 1000) * target_sr // orig_sr
        assert abs(waveform.shape[1] - expected_samples) <= 1

    def test_no_resampling_when_rate_matches(self, tmp_path):
        sr = 16000
        path, _, n_samples = _create_wav(
            tmp_path / "test.wav", sample_rate=sr, duration_ms=100
        )
        waveform, returned_sr = load_audio(path, sample_rate=sr)
        assert returned_sr == sr
        assert waveform.shape[1] == n_samples

    def test_none_sample_rate_returns_original(self, tmp_path):
        sr = 22050
        path, _, _ = _create_wav(tmp_path / "test.wav", sample_rate=sr)
        waveform, returned_sr = load_audio(path, sample_rate=None)
        assert returned_sr == sr


class TestDefaultAudioExtensions:
    """Tests for extension constants."""

    def test_contains_wav(self):
        assert ".wav" in DEFAULT_AUDIO_EXTENSIONS

    def test_all_lowercase_dotted(self):
        for ext in DEFAULT_AUDIO_EXTENSIONS:
            assert ext == ext.lower()
            assert ext.startswith(".")
