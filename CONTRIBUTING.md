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
| `torchdatasets/image/classification/from_csv.py` | `tests/test_image/test_classification_csv.py` |
| `torchdatasets/tabular/regression/from_csv.py` | `tests/test_tabular/test_regression_csv.py` |
| `torchdatasets/audio/classification/from_subdir.py` | `tests/test_audio/test_classification_subdir.py` |

### What to Test

At minimum, every new dataset class must test:

1. `__len__` returns the correct sample count
2. `__getitem__` returns the expected tuple shape `(data, label)` 
3. `return_path=True` adds a path string to the output
4. **Error cases** — empty data, missing columns, invalid files, etc.

### How to Create Test Data

Use **fixtures from `tests/conftest.py`** to create synthetic test data (tiny images, generated WAV files, CSV files). **Never commit real datasets** to the repository.

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
pytest tests/test_image/test_classification_subdir.py

# Run with coverage report
pytest --cov=torchdatasets --cov-report=term-missing
```

### CI Pipeline

All PRs are automatically tested via GitHub Actions across **Python 3.9, 3.10, and 3.11**. Your PR will not be merged unless all tests pass.

---

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/)
- We use [Black](https://github.com/psf/black) with `line-length = 88`

---

## Questions?

Feel free to open an issue if you have questions or ideas!

Thanks so much for your interest and support! 