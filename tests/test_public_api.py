"""Public import paths: every dataset class is importable from its modality package."""

import importlib
import subprocess
import sys

import pytest

import torchdatasets

EXPECTED = {
    "torchdatasets.image": {
        "BaseImageClassificationDataset": "torchdatasets.image.classification.base",
        "ImageCSVXLSXDataset": "torchdatasets.image.classification.from_csv",
        "ImageSingleDirDataset": "torchdatasets.image.classification.from_singledir",
        "ImageSubdirDataset": "torchdatasets.image.classification.from_subdir",
        "BaseImageSegmentationDataset": "torchdatasets.image.segmentation.base",
        "ImageSegCSVXLSXDataset": "torchdatasets.image.segmentation.from_csv",
        "ImageSegMultidirDataset": "torchdatasets.image.segmentation.from_multidir",
        "ImageSegSingleDirDataset": "torchdatasets.image.segmentation.from_singledir",
    },
    "torchdatasets.audio": {
        "BaseAudioClassificationDataset": "torchdatasets.audio.classification.base",
        "AudioCSVXLSXDataset": "torchdatasets.audio.classification.from_csv",
        "AudioSingleDirDataset": "torchdatasets.audio.classification.from_singledir",
        "AudioSubdirDataset": "torchdatasets.audio.classification.from_subdir",
    },
    "torchdatasets.tabular": {
        "BaseTabularDataset": "torchdatasets.tabular.base",
        "TabularDatasetFromDataFrame": "torchdatasets.tabular.from_dataframe",
        "TabularDatasetFromCSVXLSX": "torchdatasets.tabular.from_csv",
    },
}


@pytest.mark.parametrize("package", sorted(EXPECTED))
def test_package_exports(package):
    module = importlib.import_module(package)
    assert sorted(module.__all__) == sorted(EXPECTED[package])
    for name, defined_in in EXPECTED[package].items():
        # the short import path gives the same class as the full module path
        assert getattr(module, name) is getattr(importlib.import_module(defined_in), name)


@pytest.mark.parametrize(
    "package, names",
    [
        ("torchdatasets.image.classification", ["ImageSubdirDataset", "ImageSingleDirDataset", "ImageCSVXLSXDataset"]),
        ("torchdatasets.image.segmentation", ["ImageSegSingleDirDataset", "ImageSegMultidirDataset", "ImageSegCSVXLSXDataset"]),
        ("torchdatasets.audio.classification", ["AudioSubdirDataset", "AudioSingleDirDataset", "AudioCSVXLSXDataset"]),
    ],
)
def test_task_packages_export_their_classes(package, names):
    module = importlib.import_module(package)
    for name in names:
        assert name in module.__all__
        assert hasattr(module, name)


def test_top_level_gives_modalities_lazily():
    assert torchdatasets.__all__ == ["audio", "image", "tabular"]
    assert torchdatasets.image.ImageSubdirDataset.__name__ == "ImageSubdirDataset"
    assert "tabular" in dir(torchdatasets)
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


def test_tabular_import_does_not_load_image_or_audio_stack():
    """``import torchdatasets`` / the tabular package must not import torchvision or torchaudio."""
    code = (
        "import sys, torchdatasets.tabular; "
        "print(any(m in sys.modules for m in ('torchvision', 'torchaudio', 'cv2', 'torchdatasets.image')))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
