from pathlib import Path
import pandas as pd
from typing import List, Optional, Callable, Union

from torchdatasets.image.segmentation.base import BaseImageSegmentationDataset
from torchdatasets._internal.io.common import load_csv_or_excel


class ImageSegCSVXLSXDataset(BaseImageSegmentationDataset):
    """Image segmentation dataset from CSV or Excel file.
    
    Supports the following:
    1. Dynamic column mapping.
    2. Binary and multi-class segmentation.
    3. Both CSV and Excel file formats.
    """

    def __init__(
            self,
            file_path: Union[str, Path],
            transform: Optional[Callable] = None,
            image_col: str = "path",
            mask_col: str = "label",
            return_path: bool = False,
            extensions: Optional[List[str]] = None,
            is_binary: bool = False
        ):
        """Initialize the dataset.

        Args:
            file_path (Union[str, Path]): Path to the CSV or Excel file.
            transform (Optional[Callable], optional): Optional transform to be applied to the images. Defaults to None.
            image_col (str, optional): Column name for image paths. Defaults to "path".
            mask_col (str, optional): Column name for mask paths. Defaults to "label".
            return_path (bool, optional): Whether to return the path to the image. Defaults to False.
            extensions (Optional[List[str]], optional): Optional list of image extensions. Defaults to None. 
            is_binary (bool, optional): Whether the mask is binary. Defaults to False. If true, assumes >0 to be foreground and 0 to be background.
        """
        super().__init__(transform=transform, return_path=return_path, extensions=extensions, is_binary=is_binary)
        self.file_path = Path(file_path)
        self.image_col = image_col
        self.mask_col = mask_col
        self._load_samples()

    def _load_samples(self) -> None:
        df = load_csv_or_excel(self.file_path)

        # Check if the required columns are present in the DataFrame
        assert self.image_col in df.columns and self.mask_col in df.columns, f"File must contain '{self.image_col}' and '{self.mask_col}' columns."

        base_dir = self.file_path.parent
        samples: List[tuple[Path, Path]] = []

        # Iterate over the rows of the DataFrame
        for _, row in df.iterrows():
            img_path = (base_dir / str(row[self.image_col])).resolve()
            mask_path = (base_dir / str(row[self.mask_col])).resolve()
            
            # Check if the image and mask exist and have a valid extension
            if ((img_path.suffix.lower() in self.extensions and img_path.is_file()) and (img_path.suffix.lower() in self.extensions and img_path.is_file())):
                samples.append((img_path, mask_path))

        # Check if any valid samples were found
        assert len(samples) > 0, "No valid entries found in CSV/XLSX."
        self.samples = samples
    