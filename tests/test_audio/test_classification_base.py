"""Tests for BaseAudioClassificationDataset contract."""

import torch
import pytest

torchaudio = pytest.importorskip("torchaudio")

from torchdatasets.audio.classification.base import BaseAudioClassificationDataset


def _create_wav(path, sample_rate=16000, duration_ms=50, channels=1):
    """Write a tiny WAV file and return the path."""
    n_samples = int(sample_rate * duration_ms / 1000)
    waveform = torch.randn(channels, n_samples)
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), waveform, sample_rate)
    return path


class ConcreteAudioClassificationDataset(BaseAudioClassificationDataset):
    """Minimal concrete subclass for testing the base class."""

    def __init__(self, samples, class_to_idx, **kwargs):
        super().__init__(**kwargs)
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.create_metadata()


class TestLen:
    def test_length(self, tmp_path):
        paths = [_create_wav(tmp_path / f"s{i}.wav") for i in range(3)]
        samples = [(p, i) for i, p in enumerate(paths)]
        ds = ConcreteAudioClassificationDataset(samples, {"a": 0, "b": 1, "c": 2})
        assert len(ds) == 3

    def test_empty(self):
        ds = ConcreteAudioClassificationDataset([], {})
        assert len(ds) == 0


class TestGetItem:
    def test_returns_waveform_and_label(self, tmp_path):
        path = _create_wav(tmp_path / "test.wav")
        ds = ConcreteAudioClassificationDataset(
            [(path, 0)], {"cls": 0}
        )
        waveform, label = ds[0]
        assert isinstance(waveform, torch.Tensor)
        assert label == 0

    def test_return_path_gives_3_tuple(self, tmp_path):
        path = _create_wav(tmp_path / "test.wav")
        ds = ConcreteAudioClassificationDataset(
            [(path, 0)], {"cls": 0}, return_path=True
        )
        result = ds[0]
        assert len(result) == 3
        assert str(path) in result[2]

    def test_transform_applied(self, tmp_path):
        path = _create_wav(tmp_path / "test.wav")
        transform = lambda x: x * 0  # noqa: E731
        ds = ConcreteAudioClassificationDataset(
            [(path, 0)], {"cls": 0}, transform=transform
        )
        waveform, _ = ds[0]
        assert torch.all(waveform == 0)


class TestMetadata:
    def test_idx_to_class_is_inverse(self, tmp_path):
        path = _create_wav(tmp_path / "test.wav")
        class_to_idx = {"cat": 0, "dog": 1}
        ds = ConcreteAudioClassificationDataset(
            [(path, 0), (path, 1)], class_to_idx
        )
        for cls, idx in class_to_idx.items():
            assert ds.idx_to_class[idx] == cls

    def test_class_count_single_labels(self, tmp_path):
        path = _create_wav(tmp_path / "test.wav")
        samples = [(path, 0), (path, 0), (path, 1)]
        ds = ConcreteAudioClassificationDataset(samples, {"a": 0, "b": 1})
        assert ds.class_count[0] == 2
        assert ds.class_count[1] == 1

    def test_class_count_multi_labels(self, tmp_path):
        path = _create_wav(tmp_path / "test.wav")
        samples = [(path, [0, 1]), (path, [1])]
        ds = ConcreteAudioClassificationDataset(samples, {"a": 0, "b": 1})
        assert ds.class_count[0] == 1
        assert ds.class_count[1] == 2


class TestExtensions:
    def test_default_includes_wav(self):
        ds = ConcreteAudioClassificationDataset([], {})
        assert ".wav" in ds.extensions

    def test_custom_extensions(self):
        ds = ConcreteAudioClassificationDataset(
            [], {}, extensions=[".FLAC", ".OGG"]
        )
        assert ds.extensions == {".flac", ".ogg"}
