# torch-datasets

Note: This is a placeholder release to reserve the package name. A proper release will follow soon.

**One toolkit for all your PyTorch data loading needs.**  
_Still cooking — stay tuned!_

---

## 🚀 Overview

**torch-datasets** is a unified, flexible, and extensible toolkit for handling dataset loading in PyTorch. Whether you're working with images, text, audio, tabular data, or custom formats, `torch-datasets` aims to make your data loading pipeline simple, efficient, and scalable.

> 🔧 This project is currently under active development — stay tuned for updates and releases!

---

## ✨ Features (Planned)

- Built-in support for popular dataset formats (images, text, tabular, audio)
- Easy-to-use wrappers for custom datasets
- Fast data loading using multiprocessing and caching
- Simple APIs for train/val/test splitting
- Seamless integration with `torch.utils.data.DataLoader`
- Directory-based loading with automatic labeling
- Plugin system for extending dataset types
- Designed with reproducibility and best practices in mind

---

## Installation

**Coming soon to PyPI!**

For now, you can install the development version directly from source:

```bash
git clone https://github.com/Shubh-Goyal-07/torch-datasets.git
cd torch-datasets
pip install -e .
````

---

## 🧑‍💻 Quick Start

```python
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from torchdatasets.image.classification.from_subdir import ImageSubdirDataset

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
> `DataLoader` can't stack. See [BATCHING.md](BATCHING.md) for what to watch for and
> example `collate_fn`s.

### Available datasets

| Modality | Task | Data layout | Class | Import from |
| -------- | ---- | ----------- | ----- | ----------- |
| Image | Classification | One folder per class | `ImageSubdirDataset` | `torchdatasets.image.classification.from_subdir` |
| Image | Classification (single / multi-label) | CSV or Excel file of paths and labels | `ImageCSVXLSXDataset` | `torchdatasets.image.classification.from_csv` |
| Image | Segmentation | One folder: `img.jpg` + `img{suffix}.png` | `ImageSegSingleDirDataset` | `torchdatasets.image.segmentation.from_singledir` |
| Image | Segmentation | `images/` + `masks/` folders | `ImageSegMultidirDataset` | `torchdatasets.image.segmentation.from_multidir` |
| Image | Segmentation | CSV or Excel file of image and mask paths | `ImageSegCSVXLSXDataset` | `torchdatasets.image.segmentation.from_csv` |
| Audio | Classification | One folder per class | `AudioSubdirDataset` | `torchdatasets.audio.classification.from_subdir` |
| Audio | Classification (single / multi-label) | CSV or Excel file of paths and labels | `AudioCSVXLSXDataset` | `torchdatasets.audio.classification.from_csv` |
| Tabular | Classification / regression | pandas DataFrame | `TabularDatasetFromDataFrame` | `torchdatasets.tabular` |
| Tabular | Classification / regression | CSV or Excel file | `TabularDatasetFromCSVXLSX` | `torchdatasets.tabular` |

Segmentation datasets return an image and its mask as torchvision `tv_tensors`, and call the
transform as `transform(image, mask)`, so v2 transforms (flips, crops, resizes) are applied to
both consistently.

For tabular data, build validation/test sets with `fit_from` so they reuse the training set's
fitted preprocessing (fill values, category encoding, scaling, label mapping):

```python
from torchdatasets.tabular import TabularDatasetFromCSVXLSX

train_ds = TabularDatasetFromCSVXLSX("train.csv", target_cols="label")
val_ds = TabularDatasetFromCSVXLSX("val.csv", fit_from=train_ds)
```

---

## 🛠️ Project Status

| Feature                  | Status         |
| ------------------------ | -------------- |
| Project initialized      | ✅ Complete     |
| Image datasets           | 🚧 In Progress |
| Tabular datasets         | 🚧 In Progress |
| Text datasets            | ⏳ Planned      |
| Audio datasets           | 🚧 In Progress |
| Multimodal datasets      | ⏳ Planned      |
| Benchmarking tools       | ⏳ Planned      |

Follow the repository for ongoing updates. Feature suggestions and pull requests are welcome!

---

## 🤝 Contributing

A full [CONTRIBUTING.md](CONTRIBUTING.md) is here.

---

## 📄 License

This project is licensed under the **Apache License 2.0**.
See the full [LICENSE](LICENSE) file for details.

---

## 📬 Stay Connected

* 📘 [PyTorch Documentation](https://pytorch.org/docs/stable/data.html)
* 🐞 [Report Issues](https://github.com/Shubh-Goyal-07/torch-datasets/issues)
* ⭐ [Star the Repo](https://github.com/Shubh-Goyal-07/torch-datasets) to follow development

---

> *torch-datasets: Because data loading shouldn't slow you down.*