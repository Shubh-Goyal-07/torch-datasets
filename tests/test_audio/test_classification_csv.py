"""Tests for AudioCSVXLSXDataset — loads audio from CSV metadata."""

import torch
import pytest

torchaudio = pytest.importorskip("torchaudio")

from torchdatasets.audio.classification.from_csv import AudioCSVXLSXDataset


def _create_wav(path, sample_rate=16000, duration_ms=50, channels=1):
    """Write a tiny WAV file."""
    n_samples = int(sample_rate * duration_ms / 1000)
    waveform = torch.randn(channels, n_samples)
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), waveform, sample_rate)
    return path


class TestSingleLabel:
    def test_length(self, tmp_path, tmp_csv_factory):
        p1 = _create_wav(tmp_path / "a.wav")
        p2 = _create_wav(tmp_path / "b.wav")
        csv_path = tmp_csv_factory(
            "data.csv",
            [
                {"path": str(p1), "label": "speech"},
                {"path": str(p2), "label": "music"},
            ],
        )
        ds = AudioCSVXLSXDataset(file_path=csv_path)
        assert len(ds) == 2

    def test_getitem(self, tmp_path, tmp_csv_factory):
        p1 = _create_wav(tmp_path / "a.wav")
        csv_path = tmp_csv_factory(
            "data.csv",
            [{"path": str(p1), "label": "speech"}],
        )
        ds = AudioCSVXLSXDataset(file_path=csv_path)
        waveform, label = ds[0]
        assert isinstance(waveform, torch.Tensor)
        assert isinstance(label, int)

    def test_class_mapping(self, tmp_path, tmp_csv_factory):
        p1 = _create_wav(tmp_path / "a.wav")
        p2 = _create_wav(tmp_path / "b.wav")
        csv_path = tmp_csv_factory(
            "data.csv",
            [
                {"path": str(p1), "label": "speech"},
                {"path": str(p2), "label": "music"},
            ],
        )
        ds = AudioCSVXLSXDataset(file_path=csv_path)
        assert "speech" in ds.class_to_idx
        assert "music" in ds.class_to_idx


class TestMultiLabel:
    def test_multi_label(self, tmp_path, tmp_csv_factory):
        p1 = _create_wav(tmp_path / "a.wav")
        csv_path = tmp_csv_factory(
            "data.csv",
            [{"path": str(p1), "label": "speech,music"}],
        )
        ds = AudioCSVXLSXDataset(file_path=csv_path, label_sep=",")
        _, label = ds[0]
        assert isinstance(label, list)
        assert len(label) == 2


class TestReturnPath:
    def test_return_path(self, tmp_path, tmp_csv_factory):
        p1 = _create_wav(tmp_path / "a.wav")
        csv_path = tmp_csv_factory(
            "data.csv",
            [{"path": str(p1), "label": "speech"}],
        )
        ds = AudioCSVXLSXDataset(file_path=csv_path, return_path=True)
        result = ds[0]
        assert len(result) == 3


class TestCustomColumns:
    def test_custom_column_names(self, tmp_path, tmp_csv_factory):
        p1 = _create_wav(tmp_path / "a.wav")
        csv_path = tmp_csv_factory(
            "data.csv",
            [{"audio_path": str(p1), "class": "speech"}],
        )
        ds = AudioCSVXLSXDataset(
            file_path=csv_path, path_col="audio_path", label_col="class"
        )
        assert len(ds) == 1


class TestErrors:
    def test_missing_audio_skipped(self, tmp_path, tmp_csv_factory):
        p1 = _create_wav(tmp_path / "exists.wav")
        csv_path = tmp_csv_factory(
            "data.csv",
            [
                {"path": str(p1), "label": "speech"},
                {"path": str(tmp_path / "missing.wav"), "label": "music"},
            ],
        )
        ds = AudioCSVXLSXDataset(file_path=csv_path)
        assert len(ds) == 1

    def test_no_valid_entries_raises(self, tmp_path, tmp_csv_factory):
        csv_path = tmp_csv_factory(
            "data.csv",
            [{"path": str(tmp_path / "missing.wav"), "label": "speech"}],
        )
        with pytest.raises(AssertionError, match="No valid audio entries"):
            AudioCSVXLSXDataset(file_path=csv_path)

    def test_missing_columns_raises(self, tmp_path, tmp_csv_factory):
        csv_path = tmp_csv_factory("data.csv", [{"x": 1, "y": 2}])
        with pytest.raises(AssertionError, match="path.*label"):
            AudioCSVXLSXDataset(file_path=csv_path)
