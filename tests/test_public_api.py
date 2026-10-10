"""Public import paths: every dataset class is importable from its modality package."""

import importlib
import subprocess
import sys

import pytest

import torchdatasets

EXPECTED = {
    "torchdatasets.image": {
        "BaseImageClassificationDataset": "torchdatasets.image.classification.base",
        "ImageCSVDataset": "torchdatasets.image.classification.from_csv",
        "ImageSingleDirDataset": "torchdatasets.image.classification.from_singledir",
        "ImageSubdirDataset": "torchdatasets.image.classification.from_subdir",
        "BaseImageSegmentationDataset": "torchdatasets.image.segmentation.base",
        "ImageSegCSVDataset": "torchdatasets.image.segmentation.from_csv",
        "ImageSegMultiDirDataset": "torchdatasets.image.segmentation.from_multidir",
        "ImageSegSingleDirDataset": "torchdatasets.image.segmentation.from_singledir",
    },
    "torchdatasets.audio": {
        "BaseAudioClassificationDataset": "torchdatasets.audio.classification.base",
        "AudioCSVDataset": "torchdatasets.audio.classification.from_csv",
        "AudioSingleDirDataset": "torchdatasets.audio.classification.from_singledir",
        "AudioSubdirDataset": "torchdatasets.audio.classification.from_subdir",
    },
    "torchdatasets.tabular": {
        "BaseTabularDataset": "torchdatasets.tabular.base",
        "TabularDataFrameDataset": "torchdatasets.tabular.from_dataframe",
        "TabularCSVDataset": "torchdatasets.tabular.from_csv",
    },
    "torchdatasets.metric_learning": {
        "BaseMetricLearningWrapper": "torchdatasets.metric_learning.base",
        "ContrastiveWrapper": "torchdatasets.metric_learning.contrastive",
        "TripletWrapper": "torchdatasets.metric_learning.triplet",
        "FewShotWrapper": "torchdatasets.metric_learning.few_shot",
    },
    "torchdatasets.utils": {
        "TransformSubset": "torchdatasets.utils",
        "train_val_split": "torchdatasets.utils",
        "train_val_test_split": "torchdatasets.utils",
    },
}


@pytest.mark.parametrize("package", sorted(EXPECTED))
def test_package_exports(package):
    module = importlib.import_module(package)
    assert sorted(module.__all__) == sorted(EXPECTED[package])
    for name, defined_in in EXPECTED[package].items():
        # the short import path gives the same class as the full module path
        assert getattr(module, name) is getattr(importlib.import_module(defined_in), name)


# Only the data source may be passed by position; every option is keyword-only, so options
# can be added or reordered after 1.0 without breaking anyone's code.
POSITIONAL = {
    "ImageSubdirDataset": ["root_dir"],
    "ImageSingleDirDataset": ["root_dir"],
    "ImageCSVDataset": ["file_path"],
    "ImageSegSingleDirDataset": ["root_dir"],
    "ImageSegMultiDirDataset": ["image_dir", "mask_dir"],
    "ImageSegCSVDataset": ["file_path"],
    "AudioSubdirDataset": ["root_dir"],
    "AudioSingleDirDataset": ["root_dir"],
    "AudioCSVDataset": ["file_path"],
    "TabularDataFrameDataset": ["dataframe"],
    "TabularCSVDataset": ["file_path"],
    "BaseImageClassificationDataset": [],
    "BaseImageSegmentationDataset": [],
    "BaseAudioClassificationDataset": [],
    "BaseTabularDataset": [],
    "BaseMetricLearningWrapper": ["dataset"],
    "ContrastiveWrapper": ["dataset"],
    "TripletWrapper": ["dataset"],
    "FewShotWrapper": ["dataset"],
    "TransformSubset": ["dataset", "indices"],
    "train_val_split": ["dataset"],
    "train_val_test_split": ["dataset"],
}


def test_every_public_name_has_a_positional_rule():
    exported = {name for names in EXPECTED.values() for name in names}
    assert exported == set(POSITIONAL)


@pytest.mark.parametrize("package, name", [(p, n) for p, names in sorted(EXPECTED.items()) for n in sorted(names)])
def test_only_the_data_source_is_positional(package, name):
    import inspect

    obj = getattr(importlib.import_module(package), name)
    params = inspect.signature(obj.__init__ if inspect.isclass(obj) else obj).parameters.values()
    positional = [p.name for p in params if p.kind is p.POSITIONAL_OR_KEYWORD and p.name != "self"]
    assert positional == POSITIONAL[name]
    assert not any(p.kind is p.VAR_POSITIONAL for p in params)


@pytest.mark.parametrize(
    "package, names",
    [
        ("torchdatasets.image.classification", ["ImageSubdirDataset", "ImageSingleDirDataset", "ImageCSVDataset"]),
        ("torchdatasets.image.segmentation", ["ImageSegSingleDirDataset", "ImageSegMultiDirDataset", "ImageSegCSVDataset"]),
        ("torchdatasets.audio.classification", ["AudioSubdirDataset", "AudioSingleDirDataset", "AudioCSVDataset"]),
    ],
)
def test_task_packages_export_their_classes(package, names):
    module = importlib.import_module(package)
    for name in names:
        assert name in module.__all__
        assert hasattr(module, name)


def test_top_level_gives_subpackages_lazily():
    assert torchdatasets.__all__ == ["audio", "image", "metric_learning", "tabular", "utils"]
    assert torchdatasets.image.ImageSubdirDataset.__name__ == "ImageSubdirDataset"
    assert torchdatasets.utils.train_val_split.__name__ == "train_val_split"
    assert "tabular" in dir(torchdatasets) and "metric_learning" in dir(torchdatasets)
    with pytest.raises(AttributeError):
        torchdatasets.video


def test_getattr_imports_modality_on_demand():
    assert torchdatasets.__getattr__("audio") is importlib.import_module("torchdatasets.audio")


def test_version_is_a_string():
    assert isinstance(torchdatasets.__version__, str)
    assert torchdatasets.__version__


def test_version_fallback_when_not_installed():
    from importlib.metadata import PackageNotFoundError
    from unittest import mock

    try:
        with mock.patch("importlib.metadata.version", side_effect=PackageNotFoundError):
            importlib.reload(torchdatasets)
        assert torchdatasets.__version__ == "0.0.0+unknown"
    finally:
        importlib.reload(torchdatasets)


@pytest.mark.parametrize("package", ["torchdatasets.tabular", "torchdatasets.utils", "torchdatasets.metric_learning"])
def test_import_does_not_load_image_or_audio_stack(package):
    """Tabular data and the modality-agnostic helpers must not import torchvision or torchaudio."""
    code = (
        f"import sys, {package}; "
        "print(any(m in sys.modules for m in ('torchvision', 'torchaudio', 'cv2', 'torchdatasets.image')))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
