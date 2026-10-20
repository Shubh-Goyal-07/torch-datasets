# torch-datasets

[![PyPI](https://img.shields.io/pypi/v/torch-custom-datasets)](https://pypi.org/project/torch-custom-datasets/)
[![Python](https://img.shields.io/pypi/pyversions/torch-custom-datasets)](https://pypi.org/project/torch-custom-datasets/)
[![Tests](https://github.com/Shubh-Goyal-07/torch-datasets/actions/workflows/tests.yml/badge.svg)](https://github.com/Shubh-Goyal-07/torch-datasets/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/Shubh-Goyal-07/torch-datasets/blob/main/LICENSE)

**One toolkit for all your PyTorch data loading needs.**

---

## 🚀 Overview

**torch-datasets** gives you ready-made PyTorch datasets for the ways image, audio and tabular
data are usually stored: one folder per class, one folder with the class in each file name,
CSV/Excel files, and pandas DataFrames. On top of that it splits datasets into
train/validation/test parts with their own transforms, and wraps them for contrastive, triplet
and few-shot training.

Every dataset is a regular `torch.utils.data.Dataset`, so it works with `DataLoader`,
samplers and your own transforms.

---

## 📦 Installation

```bash
pip install torch-custom-datasets
```

The package is installed as `torch-custom-datasets` and imported as `torchdatasets`. It needs
Python 3.10 or newer and PyTorch 2.1 or newer (with torchvision and torchaudio), which pip installs
if they are missing. For a CUDA build of PyTorch, install PyTorch first by following
[pytorch.org](https://pytorch.org/get-started/locally/).

Excel files: `.xlsx` works out of the box; old `.xls` files also need `pip install xlrd`.

To install the latest development version from GitHub instead:

```bash
pip install git+https://github.com/Shubh-Goyal-07/torch-datasets.git
```

---

## 🧑‍💻 Quick Start

```python
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from torchdatasets.image import ImageSubdirDataset

# Images are loaded as uint8 tensors of shape [3, H, W], so use torchvision.transforms.v2
# (v2.ToDtype(..., scale=True) takes the place of the old ToTensor())
transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToDtype(torch.float32, scale=True),
])

# Load images from a directory structure: root/class_x/xxx.png
dataset = ImageSubdirDataset("path/to/images", transform=transform)
print(dataset.class_to_idx)  # e.g. {'cat': 0, 'dog': 1}

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
for images, labels in dataloader:
    # Your model training code here
    pass
```

> **Batching:** some datasets can return samples of different shapes (multi-label labels,
> audio clips of different lengths, images of different sizes), which the default
> `DataLoader` can't stack. See
> [BATCHING.md](https://github.com/Shubh-Goyal-07/torch-datasets/blob/main/BATCHING.md) for what
> to watch for and example `collate_fn`s.

---

## 📚 Available datasets

| Modality | Task | Data layout | Class | Import from |
| -------- | ---- | ----------- | ----- | ----------- |
| Image | Classification (single / multi-label) | One folder per class (`cat,dog/` for several labels) | `ImageSubdirDataset` | `torchdatasets.image` |
| Image | Classification | One folder, class in the file name (`cat_001.jpg`) | `ImageSingleDirDataset` | `torchdatasets.image` |
| Image | Classification (single / multi-label) | CSV or Excel file of paths and labels | `ImageCSVDataset` | `torchdatasets.image` |
| Image | Segmentation | One folder: `img.jpg` + `img{suffix}.png` | `ImageSegSingleDirDataset` | `torchdatasets.image` |
| Image | Segmentation | `images/` + `masks/` folders | `ImageSegMultiDirDataset` | `torchdatasets.image` |
| Image | Segmentation | CSV or Excel file with `image` and `mask` path columns | `ImageSegCSVDataset` | `torchdatasets.image` |
| Audio | Classification (single / multi-label) | One folder per class (`music,speech/` for several labels) | `AudioSubdirDataset` | `torchdatasets.audio` |
| Audio | Classification | One folder, class in the file name (`speech_001.wav`) | `AudioSingleDirDataset` | `torchdatasets.audio` |
| Audio | Classification (single / multi-label) | CSV or Excel file of paths and labels | `AudioCSVDataset` | `torchdatasets.audio` |
| Tabular | Classification / regression | pandas DataFrame | `TabularDataFrameDataset` | `torchdatasets.tabular` |
| Tabular | Classification / regression | CSV or Excel file | `TabularCSVDataset` | `torchdatasets.tabular` |

One line per layout:

```python
from torchdatasets.audio import AudioSubdirDataset
from torchdatasets.image import ImageCSVDataset, ImageSegMultiDirDataset, ImageSingleDirDataset

photos = ImageSingleDirDataset("data/photos")                  # cat_001.jpg -> class "cat"
tagged = ImageCSVDataset("labels.csv", label_sep=",")          # path,label columns; "cat,dog" -> 2 labels
scenes = ImageSegMultiDirDataset("data/images", "data/masks")  # images/x.jpg + masks/x.png
clips = AudioSubdirDataset("data/audio", sample_rate=16000)    # (waveform [channels, samples], label)
```

Image datasets return `uint8` tensors of shape `[3, H, W]` and audio datasets return `float32`
waveforms of shape `[channels, samples]`. Pass `return_path=True` to get each sample's file path
as a third item.

Segmentation datasets return an image and its mask as torchvision `tv_tensors`, and call the
transform as `transform(image, mask)`, so v2 transforms (flips, crops, resizes) are applied to
both consistently.

The single-folder classification datasets read the class from the part of the file name before
the first `delimiter` (default `"_"`); pass `label_map={"cat": 0, "dog": 1}` to fix the class
indices yourself.

With `label_sep=","`, `ImageSubdirDataset` and `AudioSubdirDataset` read several labels from a
folder name: the files in `root/cat,dog/` have both labels, and the classes are the individual
labels (`cat`, `dog`). As with the multi-label CSV datasets, each label is then a list of class
indices (see BATCHING.md to turn them into multi-hot vectors).

Classification and tabular datasets expose `targets`: the label of every sample, in index order,
read without loading any files. Use it for class counts, weights or samplers:

```python
from torch.utils.data import WeightedRandomSampler

targets = torch.tensor(dataset.targets)
sample_weights = (1.0 / torch.bincount(targets).float())[targets]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(targets))
dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

### What gets loaded

- **Folders:** only files with an allowed extension are read (compared case-insensitively; change
  them with `extensions=[...]`). The defaults are `.jpg .jpeg .png .bmp .tiff .tif .webp` for
  images and `.wav .mp3 .flac .ogg` for audio. Hidden files and folders (names starting with
  `.`, such as `.DS_Store`, macOS `._photo.jpg` files or `.ipynb_checkpoints`) are skipped, and
  names are sorted case-sensitively (`Cat` before `bird`, as in torchvision's `ImageFolder`), so
  samples and class indices are the same on every OS.
- **CSV/Excel files of paths** (image, audio and segmentation datasets): paths can be absolute,
  or relative to the CSV/Excel file's own folder, whatever the working directory is. Datasets
  keep absolute paths, so `return_path=True` gives absolute paths too. A row with an empty cell,
  or whose file is missing or has another extension, is skipped, and you get one warning saying
  how many rows were skipped, with their row numbers (as a spreadsheet numbers them, the header
  being row 1) and the paths that were looked for. Cells are read as text, exactly as written
  apart from surrounding spaces: `007` and `7` are different labels, and a label such as `NA` or
  `None` is kept.
- **Paths** can contain any characters, including non-ASCII ones on Windows.
- **Errors:** if a dataset can't find anything to load (a missing folder, no files with the
  allowed extensions, no usable CSV row) it raises `FileNotFoundError`; a CSV/Excel file without
  the expected columns, or an invalid value, raises `ValueError`; an argument of the wrong type
  raises `TypeError`.

---

## 📊 Tabular data

For tabular data, build validation/test sets with `fit_from` so they reuse the training set's
fitted preprocessing (fill values, category encoding, scaling, label mapping):

```python
from torchdatasets.tabular import TabularCSVDataset

train_ds = TabularCSVDataset("train.csv", target_cols="label")
val_ds = TabularCSVDataset("val.csv", fit_from=train_ds)
```

Class labels are numbered in sorted order; pass `label_map={"low": 0, "medium": 1, "high": 2}` to
choose the numbers yourself (for example, for ordered classes).

---

## ✂️ Splitting into train / validation / test

`train_val_split` and `train_val_test_split` give every part its own transform, and keep the class
proportions with `stratify=True` (which reads `targets`):

```python
from torchdatasets.utils import train_val_split

train_ds, val_ds = train_val_split(
    dataset, val_ratio=0.2, stratify=True,
    train_transform=v2.Compose([v2.RandomResizedCrop((224, 224)), v2.ToDtype(torch.float32, scale=True)]),
    val_transform=transform,
)
```

A part's transform replaces the dataset's own transform. Leave it as `None` to keep the dataset's,
or set `val_ds.transform = None` afterwards to drop it.

A tabular dataset is split by rows and rebuilt: the training part fits the preprocessing on its own
rows and the other parts reuse it through `fit_from`, so nothing about the validation rows leaks
into training. Every part keeps the whole dataset's class numbers. If a class ends up with no
training samples (possible without `stratify`), you get a warning. Splits are reproducible
(`random_state=42` by default).

---

## 🔗 Pairs, triplets and few-shot episodes

The wrappers in `torchdatasets.metric_learning` work with any dataset that has single-label
`targets` (image, audio or tabular, and splits of them):

```python
from torchdatasets.metric_learning import ContrastiveWrapper, FewShotWrapper, TripletWrapper

pairs = ContrastiveWrapper(train_ds)        # ((x1, x2), 1 if same class else 0)
triplets = TripletWrapper(train_ds)         # (anchor, positive, negative)
episodes = FewShotWrapper(train_ds, n_way=2, k_shot=5, q_query=5)

pair_loader = DataLoader(pairs, batch_size=32, shuffle=True)
anchor, positive, negative = triplets[0]
support_set, query_set = episodes[0]        # lists of (sample, label), labels 0..n_way-1
episode_loader = DataLoader(episodes, batch_size=None)  # episodes are lists, so don't batch them
```

Partners are drawn again on every access; pass `random_state=...` to fix them per index (for
example, a fixed set of validation pairs or evaluation episodes).

---

## 🔒 API stability

From 1.0, the public API (everything importable from `torchdatasets.image`, `torchdatasets.audio`,
`torchdatasets.tabular`, `torchdatasets.metric_learning` and `torchdatasets.utils`) follows
[semantic versioning](https://semver.org/): no breaking changes before 2.0. Modules under
`torchdatasets._internal` are private.

Only the data itself can be passed by position: the folder, file or DataFrame to load (the image
and mask folders for `ImageSegMultiDirDataset`), or the dataset to split or wrap (with its indices,
for `TransformSubset`). Every other argument is keyword-only, so write
`ImageSubdirDataset("data/train", transform=transform)`, not
`ImageSubdirDataset("data/train", transform)`.

Upgrading from 0.0.3? The two classes it had moved from `torchdatasets.vision.classification` to
`torchdatasets.image`, and images are now tensors rather than PIL images; see the
[CHANGELOG](https://github.com/Shubh-Goyal-07/torch-datasets/blob/main/CHANGELOG.md).

---

## 🛠️ Project Status

| Feature                                          | Status       |
| ------------------------------------------------ | ------------ |
| Image classification and segmentation datasets   | ✅ Available |
| Audio classification datasets                    | ✅ Available |
| Tabular datasets                                 | ✅ Available |
| Train / validation / test splits                 | ✅ Available |
| Pairs, triplets and few-shot episodes            | ✅ Available |
| Text datasets                                    | ⏳ Planned   |
| Multimodal datasets                              | ⏳ Planned   |
| Benchmarking tools                               | ⏳ Planned   |

Feature suggestions and pull requests are welcome!

---

## 🤝 Contributing

See [CONTRIBUTING.md](https://github.com/Shubh-Goyal-07/torch-datasets/blob/main/CONTRIBUTING.md)
for how to set up a development install, run the tests and add a dataset.

---

## 📄 License

This project is licensed under the **Apache License 2.0**.
See the full [LICENSE](https://github.com/Shubh-Goyal-07/torch-datasets/blob/main/LICENSE) file for details.

---

## 📬 Stay Connected

* 📘 [PyTorch Documentation](https://pytorch.org/docs/stable/data.html)
* 🐞 [Report Issues](https://github.com/Shubh-Goyal-07/torch-datasets/issues)
* ⭐ [Star the Repo](https://github.com/Shubh-Goyal-07/torch-datasets) to follow development

---

> *torch-datasets: Because data loading shouldn't slow you down.*
