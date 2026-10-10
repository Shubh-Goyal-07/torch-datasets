"""Tests for AudioSingleDirDataset — one folder, class read from the file name prefix."""

import pytest
import torch

sf = pytest.importorskip("soundfile")

from torchdatasets.audio.classification.from_singledir import AudioSingleDirDataset


def _create_wav(path, sample_rate=16000, duration_ms=50, channels=1):
    """Write a tiny WAV file."""
    n_samples = int(sample_rate * duration_ms / 1000)
    waveform = torch.randn(channels, n_samples)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform.T.numpy(), sample_rate)
    return path


@pytest.fixture
def prefixed_clips(tmp_path):
    """speech x2, music x1 in one folder. Returns the folder."""
    for name in ["speech_1.wav", "music_0.wav", "speech_0.wav"]:
        _create_wav(tmp_path / "clips" / name)
    return tmp_path / "clips"


class TestBasicLoading:
    def test_length_and_classes(self, prefixed_clips):
        ds = AudioSingleDirDataset(prefixed_clips)
        assert len(ds) == 3
        assert ds.class_to_idx == {"music": 0, "speech": 1}
        assert ds.class_count == {0: 1, 1: 2}

    def test_samples_sorted_and_targets(self, prefixed_clips):
        ds = AudioSingleDirDataset(prefixed_clips)
        assert [p.name for p, _ in ds.samples] == ["music_0.wav", "speech_0.wav", "speech_1.wav"]
        assert ds.targets == [0, 1, 1]

    def test_getitem_returns_waveform_and_label(self, prefixed_clips):
        waveform, label = AudioSingleDirDataset(prefixed_clips)[0]
        assert waveform.dtype == torch.float32
        assert waveform.shape == (1, 800)  # 50 ms at 16 kHz
        assert label == 0

    def test_sample_rate_resamples(self, prefixed_clips):
        waveform, _ = AudioSingleDirDataset(prefixed_clips, sample_rate=8000)[0]
        assert waveform.shape == (1, 400)

    def test_return_path(self, prefixed_clips):
        _, _, path = AudioSingleDirDataset(prefixed_clips, return_path=True)[0]
        assert path.endswith("music_0.wav")

    def test_transform_applied(self, prefixed_clips):
        ds = AudioSingleDirDataset(prefixed_clips, transform=lambda w: w.mean(dim=0))
        waveform, _ = ds[0]
        assert waveform.shape == (800,)


class TestOptions:
    def test_custom_delimiter_and_extensions(self, tmp_path):
        _create_wav(tmp_path / "clips" / "dog-bark-1.wav")
        _create_wav(tmp_path / "clips" / "cat-meow-1.wav")
        (tmp_path / "clips" / "cat-notes.txt").write_text("not audio")
        ds = AudioSingleDirDataset(tmp_path / "clips", delimiter="-", extensions=[".WAV"])
        assert ds.class_to_idx == {"cat": 0, "dog": 1}
        assert len(ds) == 2

    def test_label_map(self, prefixed_clips):
        ds = AudioSingleDirDataset(prefixed_clips, label_map={"speech": 0, "music": 1})
        assert ds.targets == [1, 0, 0]

    def test_label_map_missing_class_raises(self, prefixed_clips):
        with pytest.raises(ValueError, match="music"):
            AudioSingleDirDataset(prefixed_clips, label_map={"speech": 0})


class TestErrors:
    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Directory not found"):
            AudioSingleDirDataset(tmp_path / "nope")

    def test_no_audio_raises(self, tmp_path):
        (tmp_path / "clips").mkdir()
        with pytest.raises(FileNotFoundError, match="No files with extensions"):
            AudioSingleDirDataset(tmp_path / "clips")


class TestHiddenFiles:
    def test_hidden_files_are_ignored(self, prefixed_clips):
        (prefixed_clips / "._speech_0.wav").write_bytes(b"macOS metadata")
        ds = AudioSingleDirDataset(prefixed_clips)
        assert len(ds) == 3
        assert sorted(ds.class_to_idx) == ["music", "speech"]
