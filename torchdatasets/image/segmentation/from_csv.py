from pathlib import Path
from typing import List, Optional, Callable, Union

from torchdatasets.image.segmentation.base import BaseImageSegmentationDataset
from torchdatasets._internal.io.common import load_csv_or_excel, require_columns


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
            image_col: str = "image",
            mask_col: str = "mask",
            return_path: bool = False,
            extensions: Optional[List[str]] = None,
            is_binary: bool = False
        ):
        """Initialize the dataset.

        Args:
            file_path (Union[str, Path]): Path to the CSV or Excel file. Relative image and mask paths
                in it are resolved from the file's own folder.
            transform (Optional[Callable], optional): Optional transform to be applied to the images. Defaults to None.
            image_col (str, optional): Column name for image paths. Defaults to "image".
            mask_col (str, optional): Column name for mask paths. Defaults to "mask".
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
        """Load image/mask pairs from the CSV or Excel file.

        Raises:
            ValueError: If the file lacks image_col or mask_col.
            FileNotFoundError: If no row points to an existing image and mask with allowed extensions.
        """
        df = load_csv_or_excel(self.file_path)

        # Check if the required columns are present in the DataFrame
        require_columns(df, [self.image_col, self.mask_col], self.file_path)

        base_dir = self.file_path.parent
        samples: List[tuple[Path, Path]] = []

        # Iterate over the rows of the DataFrame
        for _, row in df.iterrows():
            img_path = (base_dir / str(row[self.image_col])).resolve()
            mask_path = (base_dir / str(row[self.mask_col])).resolve()
            
            # Check if the image and mask exist and have a valid extension
            img_ok = img_path.suffix.lower() in self.extensions and img_path.is_file()
            mask_ok = mask_path.suffix.lower() in self.extensions and mask_path.is_file()
            if img_ok and mask_ok:
                samples.append((img_path, mask_path))

        # Check if any valid samples were found
        if not samples:
            raise FileNotFoundError(
                f"No valid entries in '{self.file_path}': none of its {len(df)} row(s) points to an existing "
                f"image and mask with extensions {sorted(self.extensions)}. Relative paths are resolved "
                f"from '{base_dir}'."
            )
        self.samples = samples
    