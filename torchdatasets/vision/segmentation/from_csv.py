from pathlib import Path
import pandas as pd

from .base import BaseImageSegmentationDataset


class ImageCSVXLSXDataset(BaseImageSegmentationDataset):
    """
    Dataset from a CSV or XLSX file.
    Supports:
      - Customizable column names
      - Multi-label classification (comma/semicolon separated labels)
    """
    def __init__(self,
                 file_path,
                 transform=None,
                 image_col="path",
                 mask_col="label",
                 sep=None,
                 return_path=False,
                 extensions=None,
                 binary_mask=False
                ):
        super().__init__(transform=transform, return_path=return_path, extensions=extensions, binary_mask=binary_mask)
        self.file_path = Path(file_path)
        self.image_col = image_col
        self.mask_col = mask_col
        self.sep = sep  # if None, assumes single-label; otherwise expects a string delimiter like ',' or ';'
        self.make_dataset()
        # self.finalize()

    def make_dataset(self):
        if self.file_path.suffix.lower() == ".csv":
            df = pd.read_csv(self.file_path)
        elif self.file_path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(self.file_path)
        else:
            raise ValueError("Only .csv and .xlsx/.xls files are supported.")

        assert self.image_col in df.columns and self.mask_col in df.columns, f"File must contain '{self.image_col}' and '{self.mask_col}' columns."

        base_dir = self.file_path.parent
        samples = []

        for _, row in df.iterrows():
            img_path = (base_dir / str(row[self.image_col])).resolve()
            mask_path = (base_dir / str(row[self.mask_col])).resolve()
            
            if ((img_path.suffix.lower() in self.extensions and img_path.is_file()) and (img_path.suffix.lower() in self.extensions and img_path.is_file())):
                samples.append((img_path, mask_path))

        assert len(samples) > 0, "No valid image entries found in CSV/XLSX."
        self.samples = samples
