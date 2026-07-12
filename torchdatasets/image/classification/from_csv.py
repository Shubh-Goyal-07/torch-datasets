from pathlib import Path
import pandas as pd
from typing import List, Optional, Callable, Union

from torchdatasets.image.classification.base import BaseImageClassificationDataset
from torchdatasets._internal.io.common import load_csv_or_excel


class ImageCSVXLSXDataset(BaseImageClassificationDataset):
    """Image classification dataset from CSV or Excel file.
    
    Supports the following:
    1. Dynamic column mapping.
    2. Both single-label and multi-label classification.
    3. Both CSV and Excel file formats.
    """
    
    def __init__(
            self,
            file_path: Union[str, Path],
            transform: Optional[Callable] = None,
            return_path: bool = False,
            extensions: Optional[List[str]] = None,
            path_col: str = "path",
            label_col: str = "label",
            label_sep: Optional[str] = None
        ) -> None:
        """Initialize the dataset from a CSV or Excel file.

        Args:
            file_path (Union[str, Path]): Path to the CSV or Excel file.
            transform (Optional[Callable], optional): Optional transform to be applied to the images. Defaults to None.
            return_path (bool, optional): Whether to return the path to the image. Defaults to False.
            extensions (Optional[List[str]], optional): Optional list of image extensions. Defaults to None.
            path_col (str, optional): Name of the column containing the path to the image. Defaults to "path".
            label_col (str, optional): Name of the column containing the label. Defaults to "label".
            label_sep (Optional[str], optional): Separator for multi-label classification. Defaults to None. If None, will assume single-label classification (i.e., no separator). If not None, it will use multi-label classification.
        """
        super().__init__(transform=transform, return_path=return_path, extensions=extensions)
        self.file_path: Path = Path(file_path)
        self.path_col: str = path_col
        self.label_col: str = label_col
        self.label_sep: Optional[str] = label_sep
        self._load_samples()
        self.create_metadata()

    def _load_samples(self) -> None:
        """Load samples from the CSV or Excel file."""

        df = load_csv_or_excel(self.file_path)

        # Check if the required columns are present in the DataFrame
        assert self.path_col in df.columns and self.label_col in df.columns, f"File must contain '{self.path_col}' and '{self.label_col}' columns."

        samples: List[tuple[Path, Union[str, List[str]]]] = []
        classes: set[str] = set()

        # Iterate over the rows of the DataFrame
        for _, row in df.iterrows():
            img_path = Path(row[self.path_col])
            label_cell = row[self.label_col]

            # Handle single-label vs multi-label classification
            if self.label_sep is None:
                labels: List[str] = [str(label_cell)]
            else:
                labels = [lbl.strip() for lbl in str(label_cell).split(self.label_sep)]

            # Check if the image exists and has a valid extension
            if img_path.suffix.lower() in self.extensions and img_path.is_file():
                samples.append((img_path, labels if self.label_sep else labels[0]))
                classes.update(labels)

        # Check if any valid samples were found
        assert len(samples) > 0, "No valid image entries found in CSV/XLSX."

        # Sort classes and create class_to_idx mapping
        classes = sorted(classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        self.samples = [(img_path, self.encode_labels(labels)) for img_pth, labels in samples]
