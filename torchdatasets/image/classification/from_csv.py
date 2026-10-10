from pathlib import Path
from typing import List, Optional, Callable, Union

from torchdatasets.image.classification.base import BaseImageClassificationDataset
from torchdatasets._internal.io.common import (
    folder_of,
    labelled_rows,
    load_csv_or_excel,
    no_usable_rows,
    require_columns,
    warn_skipped_rows,
)


class ImageCSVDataset(BaseImageClassificationDataset):
    """Image classification dataset from CSV or Excel file.
    
    Supports the following:
    1. Dynamic column mapping.
    2. Both single-label and multi-label classification.
    3. Both CSV and Excel file formats.
    """
    
    def __init__(
            self,
            file_path: Union[str, Path],
            *,
            transform: Optional[Callable] = None,
            return_path: bool = False,
            extensions: Optional[List[str]] = None,
            path_col: str = "path",
            label_col: str = "label",
            label_sep: Optional[str] = None
        ) -> None:
        """Initialize the dataset from a CSV or Excel file.

        Args:
            file_path (Union[str, Path]): Path to the CSV or Excel file. Image paths in it can be
                absolute or relative to the file's own folder (not the working directory).
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
        """Load samples from the CSV or Excel file.

        Rows with an empty cell, a missing file or another extension are skipped, with a warning.

        Raises:
            ValueError: If the file lacks path_col or label_col.
            FileNotFoundError: If no row has a label and an existing image file with an allowed extension.
        """

        df = load_csv_or_excel(self.file_path, as_text=True)

        # Check if the required columns are present in the DataFrame
        require_columns(df, [self.path_col, self.label_col], self.file_path)

        # Keep rows with a label and an existing image file; the rest are skipped with a warning
        base_dir = folder_of(self.file_path)
        samples, skipped = labelled_rows(
            df, self.path_col, self.label_col, self.label_sep, self.extensions, base_dir
        )
        requirement = (
            f"each row needs a label and an existing image file with an extension in {sorted(self.extensions)}; "
            f"relative paths start from the file's folder '{base_dir}'"
        )

        # Check if any valid samples were found
        if not samples:
            raise no_usable_rows(self.file_path, skipped, len(df), requirement)
        warn_skipped_rows(self.file_path, skipped, len(df), requirement)

        classes = {lbl for _, labels in samples for lbl in (labels if isinstance(labels, list) else [labels])}

        # Sort classes and create class_to_idx mapping
        classes = sorted(classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        self.samples = [(img_path, self.encode_labels(labels)) for img_path, labels in samples]
