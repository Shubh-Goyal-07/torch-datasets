# Contributing to the Torch Datasets Project

Hi everyone! 

I'm Shubh, I've started this project with Soham to build a collection of PyTorch dataset classes that are easy to use and help simplify and speed up data loading for everyone.

We welcome contributions! Please read the guidelines below before opening a pull request.

---

## Getting Started

1. **Fork & clone** the repository
2. **Install in dev mode**:
   ```bash
   pip install -e ".[dev]"
   ```
3. **Run the test suite** to make sure everything passes:
   ```bash
   pytest
   ```

---

## Testing Requirements for PRs

Every PR that adds or modifies a dataset class **must** include tests.

### Where to Put Tests

Create a test file in `tests/` that mirrors the source path:

| Source | Test |
|---|---|
| `torchdatasets/image/classification/from_csv.py` | `tests/test_image/classification/test_from_csv.py` |
| `torchdatasets/tabular/from_csv.py` | `tests/test_tabular/from_csv/test_from_csv.py` |
| `torchdatasets/audio/classification/from_subdir.py` | `tests/test_audio/classification/test_from_subdir.py` |

Every test folder needs an (empty) `__init__.py`, so that test files with the same name in
different folders (e.g. several `test_from_csv.py`) don't clash.

### What to Test

At minimum, every new dataset class must test:

1. `__len__` returns the correct sample count
2. `__getitem__` returns the expected tuple shape `(data, label)` 
3. `return_path=True` adds a path string to the output
4. **Error cases** — empty data, missing columns, invalid files, etc.
5. **Skipping** — folder datasets ignore hidden files (`._img.jpg`, `.ipynb_checkpoints`); CSV
   datasets warn about the rows they skip (check it with `pytest.warns`)

### Conventions for Dataset Classes

The public API is stable since 1.0.0: renaming or removing a public class, argument or attribute,
or changing what it returns, has to wait for 2.0. Adding things is fine.

- **Arguments:** only the data source is positional (the folder, file or DataFrame, or the dataset
  to wrap); put `*` right after it so every option is keyword-only, and add the class to
  `POSITIONAL` in `tests/test_public_api.py`. Options can then be added later, in any order,
  without breaking anyone's code.
- **Names:** loaders are named `<Modality><Layout>Dataset`, with the layouts `Subdir` (one folder
  per class), `SingleDir`, `MultiDir`, `CSV` (which covers Excel files too) and `DataFrame`;
  segmentation datasets start with `ImageSeg`. A dataset's folder argument is `root_dir`, a CSV/Excel file is
  `file_path`.
- **Export it:** add a new class to its package's `__init__.py` and `__all__` (e.g.
  `torchdatasets/image/classification/__init__.py` and `torchdatasets/image/__init__.py`), and
  to the list in `tests/test_public_api.py`.
- **Labels:** labelled datasets must keep `targets` working (one label per sample, read without
  loading files). Add the class to `tests/test_targets.py`.
- **Errors:** raise real exceptions, never `assert` (asserts are removed under `python -O`):
  `FileNotFoundError` for missing paths or when nothing loadable is found, `ValueError` for bad
  values or missing CSV/Excel columns, `TypeError` for arguments of the wrong type.
- **Files:** list folders with `list_files` / `list_subfolders` from `torchdatasets._internal.files`
  (sorted by name, the same way on every OS, and without hidden files), and read
  images with `load_image` / `load_mask` from `torchdatasets._internal.io.image`, which also open
  paths with non-ASCII characters on Windows.
- **Skipped rows:** when a CSV/Excel row can't be used, skip it and report all skipped rows with one
  warning (`warn_skipped_rows` in `torchdatasets._internal.io.common`), never silently.
- **Warnings:** point a warning at the user's line with `stacklevel`. In tests, a warning that
  points at the test's own code fails the test unless it is expected (see `filterwarnings` in
  `pyproject.toml`), so wrap expected warnings in `pytest.warns`.
- **Changelog:** add a line to `CHANGELOG.md` for anything users will notice.

### How to Create Test Data

Use **fixtures from `tests/conftest.py`** to create synthetic test data (tiny images, masks, CSV files); the audio tests write short WAV files with `soundfile`. **Never commit real datasets** to the repository.

```python
def test_my_dataset(tmp_image_factory, tmp_csv_factory):
    img = tmp_image_factory("cats/img_0.png")
    csv = tmp_csv_factory("data.csv", [{"path": str(img), "label": "cat"}])
    ds = MyDataset(file_path=csv)
    assert len(ds) == 1
```

### Run Tests Locally

```bash
# Run all tests
pytest

# Run a specific test file
pytest tests/test_image/classification/test_from_subdir.py

# Run with coverage report
pytest --cov=torchdatasets --cov-report=term-missing
```

### CI Pipeline

All pushes and PRs to `main` and `dev` are tested with GitHub Actions on **Python 3.10–3.14 on
Linux** and **Python 3.10 and 3.14 on Windows**, with an 80% coverage minimum, and the package
build is checked. Your PR will not be merged unless all tests pass.

Publishing a GitHub release (tag `v<version>`, matching `version` in `pyproject.toml`) runs the
same tests and then uploads the package to PyPI.

---

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/)
- We use [Black](https://github.com/psf/black) with `line-length = 88`

---

## Questions?

Feel free to open an issue if you have questions or ideas!

Thanks so much for your interest and support! 