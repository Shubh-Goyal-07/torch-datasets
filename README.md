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
git clone https://github.com/yourusername/torch-datasets.git
cd torch-datasets
pip install -e .
````

---

## 🧑‍💻 Quick Start

```python
from torchdatasets.vision.classification import FromSubDirDataset
from torch.utils.data import DataLoader
from torchvision import transforms

# Define any transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load images from a directory structure: root/class_x/xxx.png
dataset = ImageFolderDataset("path/to/images", transform=transform)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
for images, labels in dataloader:
    # Your model training code here
    pass
```

---

## 🛠️ Project Status

| Feature                  | Status         |
| ------------------------ | -------------- |
| Project initialized      | ✅ Complete     |
| Image datasets           | 🚧 In Progress |
| Text datasets            | ⏳ Planned      |
| Audio datasets           | ⏳ Planned      |
| Tabular datasets         | ⏳ Planned      |
| Custom dataset interface | ⏳ Planned      |
| Plugin support           | ⏳ Planned      |
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
