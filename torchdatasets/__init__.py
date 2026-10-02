"""PyTorch datasets for common on-disk data layouts.

Import dataset classes from their modality::

    from torchdatasets.image import ImageSubdirDataset, ImageSegMultidirDataset
    from torchdatasets.audio import AudioCSVXLSXDataset
    from torchdatasets.tabular import TabularDatasetFromCSVXLSX

The modality subpackages are imported on first use, so ``import torchdatasets`` stays cheap
and a tabular-only project never loads the image or audio stack.
"""

import importlib as _importlib
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("torch-custom-datasets")
except _PackageNotFoundError:  # running from a source checkout that was never installed
    __version__ = "0.0.0+unknown"

__all__ = ["audio", "image", "tabular"]


def __getattr__(name: str):
    """Import a modality subpackage the first time it is accessed (``torchdatasets.image``)."""
    if name in __all__:
        return _importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
