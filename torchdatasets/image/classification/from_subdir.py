from pathlib import Path
from typing import List, Optional, Callable, Union

from torchdatasets.image.classification.base import BaseImageClassificationDataset
from torchdatasets._internal.subdir_labels import samples_from_subfolders


class ImageSubdirDataset(BaseImageClassificationDataset):
    """Image classification dataset from sub-directory structure. Similar to PyTorch's ImageFolder.
    
    Works for the following directory structure:
    root_dir/
        class1/
            image1.jpg
            image2.png
            ...
        class2/
            image1.jpg
            image2.png
            ...
        ...

    With ``label_sep`` set, a folder name can list several labels (multi-label classification):
    root_dir/
        cat/
        cat,dog/        (with label_sep=",", these images are both "cat" and "dog")
        dog/
    The classes are then the individual labels, and each label is a list of class indices.

    Hidden files and folders (names starting with ``.``, such as ``.ipynb_checkpoints`` or
    macOS ``._image1.jpg`` files) are skipped.
    """
    
    def __init__(
            self,
            root_dir: Union[str, Path],
            *,
            transform: Optional[Callable] = None,
            extensions: Optional[List[str]] = None,
            return_path: bool = False,
            label_sep: Optional[str] = None
        ) -> None:
        """Initialize the dataset from a sub-directory structure.

        Args:
            root_dir (Union[str, Path]): Path to the root directory, with one sub-folder per class.
            transform (Optional[Callable], optional): Optional transform to be applied to the images. Defaults to None.
            extensions (Optional[List[str]], optional): Optional list of image extensions. Defaults to None.
            return_path (bool, optional): Whether to return the path to the image. Defaults to False.
            label_sep (Optional[str], optional): Separator between several labels in one folder name
                (e.g. ``","``), for multi-label classification. Defaults to None (each folder is one class).
        """
        super().__init__(transform=transform, return_path=return_path, extensions=extensions)
        self.root: Path = Path(root_dir)
        self.label_sep: Optional[str] = label_sep
        self._load_samples()
        self.create_metadata()

    def _load_samples(self) -> None:
        """Load samples from the sub-directory structure.

        Raises:
            ValueError: If label_sep is an empty string, or a folder name has no label between the separators.
            FileNotFoundError: If root_dir does not exist, has no class folders, or no image files.
        """
        self.samples, self.class_to_idx = samples_from_subfolders(self.root, self.extensions, self.label_sep)
