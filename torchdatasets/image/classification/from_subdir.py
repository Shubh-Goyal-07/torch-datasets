from pathlib import Path
from typing import List, Optional, Callable, Union

from torchdatasets.image.classification.base import BaseImageClassificationDataset


# TODO: (Check viability) Add support for multi-label classification (comma/semicolon separated labels) in this dataset as well, similar to the CSV/XLSX version. 
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
    """
    
    def __init__(
            self,
            root_dir: Union[str, Path],
            transform: Optional[Callable] = None,
            extensions: Optional[List[str]] = None,
            return_path: bool = False
        ) -> None:
        """Initialize the dataset from a sub-directory structure.

        Args:
            root_dir (Union[str, Path]): Path to the root directory.
            transform (Optional[Callable], optional): Optional transform to be applied to the images. Defaults to None.
            extensions (Optional[List[str]], optional): Optional list of image extensions. Defaults to None.
            return_path (bool, optional): Whether to return the path to the image. Defaults to False.
        """
        super().__init__(transform=transform, return_path=return_path, extensions=extensions)
        self.root: Path = Path(root_dir)
        self._load_samples()
        self.create_metadata()

    def _load_samples(self) -> None:
        """Load samples from the sub-directory structure.

        Raises:
            FileNotFoundError: If root_dir does not exist, has no class folders, or no image files.
        """
        if not self.root.is_dir():
            raise FileNotFoundError(f"Directory not found: '{self.root}'")

        # Get all sub-directories as classes and create class_to_idx mapping
        classes: List[str] = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        if not classes:
            raise FileNotFoundError(f"No class folders found in '{self.root}' (expected one sub-folder per class).")
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        samples: List[tuple[Path, int]] = []

        # Iterate over the classes and their images (sorted, so the order is the same on every OS)
        for cls in classes:
            for img_path in sorted((self.root / cls).glob("*")):
                if img_path.is_file() and img_path.suffix.lower() in self.extensions:
                    samples.append((img_path, self.class_to_idx[cls]))

        if not samples:
            raise FileNotFoundError(
                f"No files with extensions {sorted(self.extensions)} found in the class folders of '{self.root}'."
            )
        self.samples = samples
