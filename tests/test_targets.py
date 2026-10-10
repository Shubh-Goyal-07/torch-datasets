"""The shared label interface: every labelled dataset exposes ``targets``.

``targets[i]`` must equal the label ``dataset[i]`` returns, and reading it must not load
any image or audio file (splitting and sampling helpers rely on both).
"""

from unittest import mock

import pandas as pd
import pytest
import torch

sf = pytest.importorskip("soundfile")

from torchdatasets.audio import AudioCSVDataset, AudioSingleDirDataset, AudioSubdirDataset
from torchdatasets.image import (
    ImageCSVDataset,
    ImageSegMultiDirDataset,
    ImageSingleDirDataset,
    ImageSubdirDataset,
)
from torchdatasets.tabular import TabularCSVDataset, TabularDataFrameDataset


def _wav(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), torch.zeros(160).numpy(), 16000)
    return path


def _as_plain(label):
    """Labels from __getitem__ can be tensors (tabular); compare as Python values."""
    return label.tolist() if isinstance(label, torch.Tensor) else label


@pytest.fixture
def labelled_datasets(tmp_path, tmp_image_factory, tmp_csv_factory):
    """One instance of every labelled dataset class, each with more than one class."""
    for cls, n in {"cat": 2, "dog": 1}.items():
        for i in range(n):
            tmp_image_factory(f"img_subdir/{cls}/{i}.png")
            tmp_image_factory(f"img_single/{cls}_{i}.png")
            _wav(tmp_path / "aud_subdir" / cls / f"{i}.wav")
            _wav(tmp_path / "aud_single" / f"{cls}_{i}.wav")

    img_rows = [
        {"path": str(tmp_image_factory("img_csv/a.png")), "label": "dog"},
        {"path": str(tmp_image_factory("img_csv/b.png")), "label": "cat,dog"},
    ]
    aud_rows = [
        {"path": str(_wav(tmp_path / "aud_csv" / "a.wav")), "label": "music"},
        {"path": str(_wav(tmp_path / "aud_csv" / "b.wav")), "label": "speech"},
    ]
    df = pd.DataFrame({"x": [0.1, 0.2, 0.3], "y": ["b", "a", "b"]})
    df.to_csv(tmp_path / "tab.csv", index=False)

    return {
        "ImageSubdirDataset": ImageSubdirDataset(tmp_path / "img_subdir"),
        "ImageSingleDirDataset": ImageSingleDirDataset(tmp_path / "img_single"),
        "ImageCSVDataset (multi-label)": ImageCSVDataset(
            tmp_csv_factory("img.csv", img_rows), label_sep=","
        ),
        "AudioSubdirDataset": AudioSubdirDataset(tmp_path / "aud_subdir"),
        "AudioSingleDirDataset": AudioSingleDirDataset(tmp_path / "aud_single"),
        "AudioCSVDataset": AudioCSVDataset(tmp_csv_factory("aud.csv", aud_rows)),
        "TabularDataFrameDataset": TabularDataFrameDataset(df, target_cols="y"),
        "TabularCSVDataset": TabularCSVDataset(tmp_path / "tab.csv", target_cols="y"),
    }


def test_targets_match_getitem_labels(labelled_datasets):
    for name, ds in labelled_datasets.items():
        targets = ds.targets
        assert isinstance(targets, list), name
        assert len(targets) == len(ds), name
        assert targets == [_as_plain(ds[i][1]) for i in range(len(ds))], name


def test_targets_are_plain_python_values(labelled_datasets):
    for name, ds in labelled_datasets.items():
        for target in ds.targets:
            values = target if isinstance(target, list) else [target]
            assert all(type(v) in (int, float) for v in values), name


def test_reading_targets_loads_no_files(labelled_datasets):
    with mock.patch("torchdatasets.image.classification.base.load_image") as load_image, \
            mock.patch("torchdatasets.audio.classification.base.load_audio") as load_audio:
        for ds in labelled_datasets.values():
            ds.targets
    load_image.assert_not_called()
    load_audio.assert_not_called()


def test_single_label_targets_index_class_to_idx(labelled_datasets):
    ds = labelled_datasets["ImageSubdirDataset"]
    assert [ds.idx_to_class[t] for t in ds.targets] == ["cat", "cat", "dog"]


def test_multi_label_targets_are_index_lists(labelled_datasets):
    ds = labelled_datasets["ImageCSVDataset (multi-label)"]
    assert ds.targets == [[ds.class_to_idx["dog"]], [ds.class_to_idx["cat"], ds.class_to_idx["dog"]]]


def test_targets_follow_samples(labelled_datasets):
    """targets is derived from samples, so it can't go stale if samples are filtered."""
    ds = labelled_datasets["ImageSubdirDataset"]
    ds.samples = ds.samples[1:]
    assert ds.targets == [0, 1]


def test_segmentation_datasets_have_no_targets(image_seg_paired_dirs):
    img_dir, mask_dir, _ = image_seg_paired_dirs
    assert not hasattr(ImageSegMultiDirDataset(img_dir, mask_dir), "targets")
