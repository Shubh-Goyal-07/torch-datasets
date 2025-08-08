from pathlib import Path
import pandas as pd

from .base import BaseImageClassificationDataset


class ImageCSVXLSXDataset(BaseImageClassificationDataset):
    """
    Dataset from a CSV or XLSX file.
    Supports:
      - Customizable column names
      - Multi-label classification (comma/semicolon separated labels)
    """
    def __init__(self,
                 file_path,
                 transform=None,
                 path_col="path",
                 label_col="label",
                 sep=None,
                 extensions=None,
                 return_path=False):
        super().__init__(transform=transform, extensions=extensions, return_path=return_path)
        self.file_path = Path(file_path)
        self.path_col = path_col
        self.label_col = label_col
        self.sep = sep  # if None, assumes single-label; otherwise expects a string delimiter like ',' or ';'
        self.make_dataset()
        self.finalize()

    def make_dataset(self):
        if self.file_path.suffix.lower() == ".csv":
            df = pd.read_csv(self.file_path)
        elif self.file_path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(self.file_path)
        else:
            raise ValueError("Only .csv and .xlsx/.xls files are supported.")

        assert self.path_col in df.columns and self.label_col in df.columns, f"File must contain '{self.path_col}' and '{self.label_col}' columns."

        base_dir = self.file_path.parent
        samples = []
        classes = set()

        for _, row in df.iterrows():
            img_path = (base_dir / str(row[self.path_col])).resolve()
            label_cell = row[self.label_col]

            if self.sep is None:
                labels = [str(label_cell)]
            else:
                labels = [lbl.strip() for lbl in str(label_cell).split(self.sep)]

            if img_path.suffix.lower() in self.extensions and img_path.is_file():
                samples.append((img_path, labels if self.sep else labels[0]))
                classes.update(labels)

        assert len(samples) > 0, "No valid image entries found in CSV/XLSX."
        classes = sorted(classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        def encode_labels(lbl):
            if isinstance(lbl, list):
                return [self.class_to_idx[x] for x in lbl]
            return self.class_to_idx[lbl]

        self.samples = [(p, encode_labels(c)) for p, c in samples]
