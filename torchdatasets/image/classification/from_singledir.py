from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from torchdatasets._internal.filename_labels import samples_from_filenames
from torchdatasets.image.classification.base import BaseImageClassificationDataset


class ImageSingleDirDataset(BaseImageClassificationDataset):
    """Image classification dataset from one folder, with each image's class taken from its file name.

    Works for the following directory structure:
    root_dir/
        cat_001.jpg
        cat_002.jpg
        dog_001.png
        ...

    The class name is the part of the file name before the first ``delimiter``
    (``cat_001.jpg`` -> ``"cat"``); a name without the delimiter is used whole. Only files
    directly inside ``root_dir`` are read. Class indices follow the sorted class names,
    unless ``label_map`` gives them.
    """

    def __init__(
            self,
            root_dir: Union[str, Path],
            transform: Optional[Callable] = None,
            extensions: Optional[List[str]] = None,
            return_path: bool = False,
            delimiter: str = "_",
            label_map: Optional[Dict[str, int]] = None
        ) -> None:
        """Initialize the dataset from a single folder.

        Args:
            root_dir (Union[str, Path]): Path to the folder holding the images.
            transform (Optional[Callable], optional): Optional transform to be applied to the images. Defaults to None.
            extensions (Optional[List[str]], optional): Optional list of image extensions. Defaults to None.
            return_path (bool, optional): Whether to return the path to the image. Defaults to False.
            delimiter (str, optional): Separator between the class name and the rest of the file name. Defaults to "_".
            label_map (Optional[Dict[str, int]], optional): Class name -> class index. Must cover every class
                found in the folder. Defaults to None (sorted class names numbered 0..K-1).
        """
        super().__init__(transform=transform, return_path=return_path, extensions=extensions)
        self.root: Path = Path(root_dir)
        self.delimiter: str = delimiter
        self.label_map: Optional[Dict[str, int]] = label_map
        self._load_samples()
        self.create_metadata()

    def _load_samples(self) -> None:
        """Load samples from the file names in the folder."""
        self.samples, self.class_to_idx = samples_from_filenames(
            self.root, self.extensions, self.delimiter, self.label_map
        )
