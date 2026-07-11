"""Tests for AudioSubdirDataset — loads audio from class-per-subdirectory layout."""

import torch
import pytest

torchaudio = pytest.importorskip("torchaudio")

from torchdatasets.audio.classification.from_subdir import AudioSubdirDataset


def _create_wav(path, sample_rate=16000, duration_ms=50, channels=1):
    """Write a tiny WAV file."""
    n_samples = int(sample_rate * duration_ms / 1000)
    waveform = torch.randn(channels, n_samples)
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), waveform, sample_rate)
    return path


@pytest.fixture
def audio_subdir_tree(tmp_path):
    """Creates class subdirectories with WAV files."""
    classes = {"speech": 3, "music": 2}
    for cls_name, count in classes.items():
        for i in range(count):
            _create_wav(tmp_path / cls_name / f"sample_{i}.wav")
    return tmp_path, classes


class TestBasicLoading:
    def test_length(self, audio_subdir_tree):
        root, classes = audio_subdir_tree
        ds = AudioSubdirDataset(root_dir=root)
        assert len(ds) == sum(classes.values())

    def test_getitem_returns_waveform_and_label(self, audio_subdir_tree):
        root, _ = audio_subdir_tree
        ds = AudioSubdirDataset(root_dir=root)
        waveform, label = ds[0]
        assert isinstance(waveform, torch.Tensor)
        assert isinstance(label, int)


class TestClassMapping:
    def test_class_to_idx(self, audio_subdir_tree):
        root, classes = audio_subdir_tree
        ds = AudioSubdirDataset(root_dir=root)
        assert set(ds.class_to_idx.keys()) == set(classes.keys())

    def test_inverse_consistency(self, audio_subdir_tree):
        root, _ = audio_subdir_tree
        ds = AudioSubdirDataset(root_dir=root)
        for cls, idx in ds.class_to_idx.items():
            assert ds.idx_to_class[idx] == cls

    def test_class_count(self, audio_subdir_tree):
        root, classes = audio_subdir_tree
        ds = AudioSubdirDataset(root_dir=root)
        for cls, expected in classes.items():
            idx = ds.class_to_idx[cls]
            assert ds.class_count[idx] == expected


class TestReturnPath:
    def test_return_path(self, audio_subdir_tree):
        root, _ = audio_subdir_tree
        ds = AudioSubdirDataset(root_dir=root, return_path=True)
        result = ds[0]
        assert len(result) == 3


class TestExtensionFilter:
    def test_filters_by_extension(self, tmp_path):
        _create_wav(tmp_path / "cls" / "a.wav")
        _create_wav(tmp_path / "cls" / "b.wav")
        # Create a non-matching file
        (tmp_path / "cls" / "c.txt").write_text("not audio")
        ds = AudioSubdirDataset(root_dir=tmp_path)
        assert len(ds) == 2  # .txt excluded


class TestErrors:
    def test_empty_dir_raises(self, tmp_path):
        (tmp_path / "empty_class").mkdir()
        with pytest.raises(AssertionError, match="No valid samples"):
            AudioSubdirDataset(root_dir=tmp_path)
