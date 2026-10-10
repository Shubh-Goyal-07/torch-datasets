from pathlib import Path
from typing import List, Optional, Callable, Tuple, Union

from torchdatasets.image.segmentation.base import BaseImageSegmentationDataset
from torchdatasets._internal.io.common import (
    cell_text,
    folder_of,
    is_blank,
    load_csv_or_excel,
    no_usable_rows,
    require_columns,
    resolve_path,
    row_name,
    warn_skipped_rows,
)


class ImageSegCSVDataset(BaseImageSegmentationDataset):
    """Image segmentation dataset from CSV or Excel file.
    
    Supports the following:
    1. Dynamic column mapping.
    2. Binary and multi-class segmentation.
    3. Both CSV and Excel file formats.
    """

    def __init__(
            self,
            file_path: Union[str, Path],
            *,
            transform: Optional[Callable] = None,
            image_col: str = "image",
            mask_col: str = "mask",
            return_path: bool = False,
            extensions: Optional[List[str]] = None,
            is_binary: bool = False
        ):
        """Initialize the dataset.

        Args:
            file_path (Union[str, Path]): Path to the CSV or Excel file. Image and mask paths in it can
                be absolute or relative to the file's own folder (not the working directory).
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

        Rows with an empty cell, or whose image or mask is missing or has another extension,
        are skipped with a warning.

        Raises:
            ValueError: If the file lacks image_col or mask_col.
            FileNotFoundError: If no row points to an existing image and mask with allowed extensions.
        """
        df = load_csv_or_excel(self.file_path, as_text=True)

        # Check if the required columns are present in the DataFrame
        require_columns(df, [self.image_col, self.mask_col], self.file_path)

        base_dir = folder_of(self.file_path)
        samples: List[Tuple[Path, Path]] = []
        skipped: List[str] = []

        for position, (image_cell, mask_cell) in enumerate(zip(df[self.image_col], df[self.mask_col])):
            if is_blank(image_cell) or is_blank(mask_cell):
                skipped.append(f"{row_name(position)}: empty cell")
                continue
            img_path = resolve_path(cell_text(image_cell), base_dir)
            mask_path = resolve_path(cell_text(mask_cell), base_dir)

            # Keep the row only if the image and mask exist and have a valid extension
            img_ok = img_path.suffix.lower() in self.extensions and img_path.is_file()
            mask_ok = mask_path.suffix.lower() in self.extensions and mask_path.is_file()
            if img_ok and mask_ok:
                samples.append((img_path, mask_path))
            else:
                skipped.append(f"{row_name(position)}: {mask_path if img_ok else img_path}")

        requirement = (
            f"each row needs an existing image and mask with an extension in {sorted(self.extensions)}; "
            f"relative paths start from the file's folder '{base_dir}'"
        )
        if not samples:
            raise no_usable_rows(self.file_path, skipped, len(df), requirement)
        warn_skipped_rows(self.file_path, skipped, len(df), requirement)
        self.samples = samples
