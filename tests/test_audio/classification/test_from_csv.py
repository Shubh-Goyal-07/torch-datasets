"""Tests for AudioCSVDataset — loads audio from CSV metadata."""

import torch
import pytest

sf = pytest.importorskip("soundfile")

from torchdatasets.audio.classification.from_csv import AudioCSVDataset


def _create_wav(path, sample_rate=16000, duration_ms=50, channels=1):
    """Write a tiny WAV file."""
    n_samples = int(sample_rate * duration_ms / 1000)
    waveform = torch.randn(channels, n_samples)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform.T.numpy(), sample_rate)
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
        ds = AudioCSVDataset(file_path=csv_path)
        assert len(ds) == 2

    def test_getitem(self, tmp_path, tmp_csv_factory):
        p1 = _create_wav(tmp_path / "a.wav")
        csv_path = tmp_csv_factory(
            "data.csv",
            [{"path": str(p1), "label": "speech"}],
        )
        ds = AudioCSVDataset(file_path=csv_path)
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
        ds = AudioCSVDataset(file_path=csv_path)
        assert "speech" in ds.class_to_idx
        assert "music" in ds.class_to_idx


class TestMultiLabel:
    def test_multi_label(self, tmp_path, tmp_csv_factory):
        p1 = _create_wav(tmp_path / "a.wav")
        csv_path = tmp_csv_factory(
            "data.csv",
            [{"path": str(p1), "label": "speech,music"}],
        )
        ds = AudioCSVDataset(file_path=csv_path, label_sep=",")
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
        ds = AudioCSVDataset(file_path=csv_path, return_path=True)
        result = ds[0]
        assert len(result) == 3


class TestCustomColumns:
    def test_custom_column_names(self, tmp_path, tmp_csv_factory):
        p1 = _create_wav(tmp_path / "a.wav")
        csv_path = tmp_csv_factory(
            "data.csv",
            [{"audio_path": str(p1), "class": "speech"}],
        )
        ds = AudioCSVDataset(
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
        with pytest.warns(UserWarning, match=r"Skipped 1 of 2 row\(s\).*missing\.wav"):
            ds = AudioCSVDataset(file_path=csv_path)
        assert len(ds) == 1

    def test_no_valid_entries_raises(self, tmp_path, tmp_csv_factory):
        csv_path = tmp_csv_factory(
            "data.csv",
            [{"path": str(tmp_path / "missing.wav"), "label": "speech"}],
        )
        with pytest.raises(FileNotFoundError, match="No valid entries"):
            AudioCSVDataset(file_path=csv_path)

    def test_missing_columns_raises(self, tmp_path, tmp_csv_factory):
        csv_path = tmp_csv_factory("data.csv", [{"x": 1, "y": 2}])
        with pytest.raises(ValueError, match="path.*label"):
            AudioCSVDataset(file_path=csv_path)


class TestExcelFile:
    def test_loads_xlsx(self, tmp_path):
        pd = pytest.importorskip("pandas")
        p1, p2 = _create_wav(tmp_path / "a.wav"), _create_wav(tmp_path / "b.wav")
        xlsx_path = tmp_path / "data.xlsx"
        pd.DataFrame({"path": [str(p1), str(p2)], "label": ["speech", "music"]}).to_excel(xlsx_path, index=False)
        ds = AudioCSVDataset(file_path=xlsx_path)
        assert len(ds) == 2
        assert ds.class_to_idx == {"music": 0, "speech": 1}


class TestSkippedRows:
    def test_blank_cells_are_skipped_with_a_warning(self, tmp_path, tmp_csv_factory):
        p1 = _create_wav(tmp_path / "a.wav")
        p2 = _create_wav(tmp_path / "b.wav")
        csv_path = tmp_csv_factory(
            "data.csv",
            [
                {"path": str(p1), "label": "speech"},
                {"path": str(p2), "label": None},
                {"path": None, "label": "music"},
            ],
        )
        with pytest.warns(UserWarning, match=r"Skipped 2 of 3 row\(s\)") as record:
            ds = AudioCSVDataset(csv_path)
        assert record[0].filename == __file__  # the warning points at the caller
        assert len(ds) == 1
        assert ds.class_to_idx == {"speech": 0}


class TestPaths:
    def test_relative_paths_start_from_the_csv_folder(self, tmp_path, monkeypatch):
        clip = _create_wav(tmp_path / "data" / "clips" / "a.wav")
        (tmp_path / "data" / "clips.csv").write_text("path,label\nclips/a.wav,speech\n")
        (tmp_path / "elsewhere").mkdir()
        monkeypatch.chdir(tmp_path / "elsewhere")
        ds = AudioCSVDataset(tmp_path / "data" / "clips.csv", return_path=True)
        assert ds.samples[0][0] == clip
        assert ds[0][2] == str(clip)
