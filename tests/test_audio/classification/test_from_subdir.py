"""Tests for AudioSubdirDataset — loads audio from class-per-subdirectory layout."""

import torch
import pytest

sf = pytest.importorskip("soundfile")

from torchdatasets.audio.classification.from_subdir import AudioSubdirDataset


def _create_wav(path, sample_rate=16000, duration_ms=50, channels=1):
    """Write a tiny WAV file."""
    n_samples = int(sample_rate * duration_ms / 1000)
    waveform = torch.randn(channels, n_samples)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform.T.numpy(), sample_rate)
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
        with pytest.raises(FileNotFoundError, match="No files with extensions"):
            AudioSubdirDataset(root_dir=tmp_path)

    def test_no_class_folders_raises(self, tmp_path):
        _create_wav(tmp_path / "loose.wav")
        with pytest.raises(FileNotFoundError, match="No class folders"):
            AudioSubdirDataset(root_dir=tmp_path)

    def test_missing_root_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Directory not found"):
            AudioSubdirDataset(root_dir=tmp_path / "nope")


class TestOrdering:
    def test_samples_are_sorted_within_each_class(self, tmp_path):
        for name in ["c.wav", "a.wav", "b.wav"]:
            _create_wav(tmp_path / "cls" / name)
        ds = AudioSubdirDataset(root_dir=tmp_path)
        assert [p.name for p, _ in ds.samples] == ["a.wav", "b.wav", "c.wav"]


class TestHiddenFiles:
    def test_hidden_folders_and_files_are_ignored(self, audio_subdir_tree):
        root, classes = audio_subdir_tree
        (root / ".ipynb_checkpoints").mkdir()
        (root / ".ipynb_checkpoints" / "sample_0.wav").write_bytes(b"not audio")
        (root / "speech" / "._sample_0.wav").write_bytes(b"macOS metadata")
        ds = AudioSubdirDataset(root)
        assert sorted(ds.class_to_idx) == sorted(classes)
        assert len(ds) == sum(classes.values())
