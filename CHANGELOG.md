# Changelog

All notable changes to this project are listed here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/): from 1.0.0 on, breaking changes only come with a new
major version.

## [1.0.0] - Unreleased

The first stable release, rewritten since 0.0.3.

### Added

- **Image classification:** `ImageSubdirDataset` (one folder per class), `ImageSingleDirDataset`
  (class in the file name) and `ImageCSVDataset` (CSV/Excel file of paths and labels, single- or
  multi-label).
- **Image segmentation:** `ImageSegSingleDirDataset`, `ImageSegMultiDirDataset` and
  `ImageSegCSVDataset`. Images and masks are torchvision `tv_tensors`, and the transform is called
  as `transform(image, mask)`, so v2 transforms change both the same way.
- **Audio classification:** `AudioSubdirDataset`, `AudioSingleDirDataset` and `AudioCSVDataset`,
  decoded with soundfile and optionally resampled (`sample_rate=`).
- **Tabular data:** `TabularDataFrameDataset` and `TabularCSVDataset`, with fitted preprocessing
  (fill values, category encoding, scaling, label mapping) that validation and test sets reuse
  through `fit_from=`, and `label_map=` to choose the class numbers.
- `targets` on every classification and tabular dataset: the label of each sample, read without
  loading any files.
- `torchdatasets.utils`: `train_val_split` and `train_val_test_split` (optionally stratified, with
  a transform per part) and `TransformSubset`.
- `torchdatasets.metric_learning`: `ContrastiveWrapper`, `TripletWrapper` and `FewShotWrapper`, plus
  `BaseMetricLearningWrapper` for your own.
- Base classes for your own layouts: `BaseImageClassificationDataset`,
  `BaseImageSegmentationDataset`, `BaseAudioClassificationDataset` and `BaseTabularDataset`.
- Excel files work out of the box (`openpyxl` is a dependency); old `.xls` files need `xlrd`.
- Python 3.12, 3.13 and 3.14 support. Tests run on Linux (Python 3.10–3.14) and Windows (Python
  3.10 and 3.14).

### How files are found

- Folder datasets read only files with an allowed extension (`extensions=`, compared
  case-insensitively; images default to `.jpg .jpeg .png .bmp .tiff .tif .webp`), skip hidden files
  and folders (`.DS_Store`, macOS `._photo.jpg` files, `.ipynb_checkpoints`), and sort names
  case-sensitively, so samples and class indices are the same on every OS.
- The image, audio and segmentation CSV/Excel datasets skip rows with an empty cell, a missing file
  or another extension, with one warning that says how many rows were skipped. Their cells are read
  as text, exactly as written (`007` and `7` are different labels; `NA` is a label, not a missing
  value).
- Images are decoded from the file's bytes, so paths with non-ASCII characters work on Windows.
- Missing data raises `FileNotFoundError`; invalid values and missing CSV/Excel columns raise
  `ValueError`; arguments of the wrong type raise `TypeError`.

### Changed (breaking)

Coming from 0.0.3:

- Python 3.10 or newer is required (0.0.3 supported 3.9–3.11).
- `torchdatasets.vision.classification` is now `torchdatasets.image`.
- Images are `uint8` tensors of shape `[3, H, W]` (`tv_tensors.Image`) instead of PIL images. Use
  `torchvision.transforms.v2`, and replace `ToTensor()` with `v2.ToDtype(torch.float32, scale=True)`.
- The folder argument is `root_dir` (was `root`), and every other argument is keyword-only:
  `ImageSubdirDataset("data/train", transform=t)`, not `ImageSubdirDataset("data/train", t)`.
- Only files with an image extension are loaded, hidden files are skipped, and samples are sorted
  (0.0.3 loaded every file in the folder, in file-system order).
- `ImageSingleDirDataset`: a `label_map` must name every class found, and a file name with nothing
  before the delimiter is an error (both raise `ValueError`).
- The `classes` attribute is gone: use `class_to_idx` (name → index) or `idx_to_class`
  (index → name).

If you installed the package from GitHub before 1.0 (the old `main` branch or `dev`), these names
changed too:

| Before | 1.0.0 |
| ------ | ----- |
| `torchdatasets.vision.classification.ImageCSVXLSXDataset` | `torchdatasets.image.ImageCSVDataset` |
| `torchdatasets.vision.segmentation.ImageSubdirDataset` | `torchdatasets.image.ImageSegMultiDirDataset` |
| `torchdatasets.vision.segmentation.ImageCSVXLSXDataset` | `torchdatasets.image.ImageSegCSVDataset` |
| `torchdatasets.vision.metric_learning` | `torchdatasets.metric_learning` |
| `make_contrastive`, `make_triplet`, `make_few_shot` | removed: create `ContrastiveWrapper`, `TripletWrapper` or `FewShotWrapper` directly |
| `ImageCSVXLSXDataset` (`torchdatasets.image`) | `ImageCSVDataset` |
| `ImageSegCSVXLSXDataset` | `ImageSegCSVDataset` |
| `ImageSegMultidirDataset` | `ImageSegMultiDirDataset` |
| `AudioCSVXLSXDataset` | `AudioCSVDataset` |
| `TabularDatasetFromDataFrame` | `TabularDataFrameDataset` |
| `TabularDatasetFromCSVXLSX` | `TabularCSVDataset` |
| argument `root` / `src_dir` | `root_dir` (attribute `root`) |
| argument `sep` | `label_sep` |
| argument `binary_mask` | `is_binary` |

Every option is keyword-only, and `ImageSegCSVDataset` reads the `image` and `mask` columns by
default.

### Migrating from 0.0.3

```python
# 0.0.3
from torchvision import transforms
from torchdatasets.vision.classification import ImageSubdirDataset

ds = ImageSubdirDataset("data/train", transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]))

# 1.0.0
import torch
from torchvision.transforms import v2
from torchdatasets.image import ImageSubdirDataset

ds = ImageSubdirDataset("data/train", transform=v2.Compose([v2.Resize((224, 224)), v2.ToDtype(torch.float32, scale=True)]))
```

## [0.0.3] - 2025-08-05

Early release that reserved the package name on PyPI: `ImageSubdirDataset` and
`ImageSingleDirDataset` in `torchdatasets.vision.classification`, returning PIL images.

[1.0.0]: https://github.com/Shubh-Goyal-07/torch-datasets/releases/tag/v1.0.0
[0.0.3]: https://pypi.org/project/torch-custom-datasets/0.0.3/
