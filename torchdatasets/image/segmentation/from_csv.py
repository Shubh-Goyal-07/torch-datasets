from pathlib import Path
import pandas as pd
from typing import List, Optional, Callable, Union

from torchdatasets.image.segmentation.base import BaseImageSegmentationDataset
from torchdatasets._internal.io.common import load_csv_or_excel

class ImageSegCSVXLSXDataset(BaseImageSegmentationDataset):
    """
    Dataset from a CSV or XLSX file.
    Supports:
      - Customizable column names
      - Multi-label classification (comma/semicolon separated labels)
    """
    def __init__(
            self,
            file_path: Union[str, Path],
            transform: Optional[Callable] = None,
            image_col: str = "path",
            mask_col: str = "label",
            return_path: bool = False,
            extensions: Optional[List[str]] = None,
            binary_mask: bool = False
        ):
        super().__init__(transform=transform, return_path=return_path, extensions=extensions, binary_mask=binary_mask)
        self.file_path = Path(file_path)
        self.image_col = image_col
        self.mask_col = mask_col
        self.make_dataset()

    def make_dataset(self):
        df = load_csv_or_excel(self.file_path)

        assert self.image_col in df.columns and self.mask_col in df.columns, f"File must contain '{self.image_col}' and '{self.mask_col}' columns."

        base_dir = self.file_path.parent
        samples = []

        for _, row in df.iterrows():
            img_path = (base_dir / str(row[self.image_col])).resolve()
            mask_path = (base_dir / str(row[self.mask_col])).resolve()
            
            if ((img_path.suffix.lower() in self.extensions and img_path.is_file()) and (img_path.suffix.lower() in self.extensions and img_path.is_file())):
                samples.append((img_path, mask_path))

        assert len(samples) > 0, "No valid entries found in CSV/XLSX."
        self.samples = samples