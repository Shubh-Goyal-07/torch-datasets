from pathlib import Path
from typing import List, Optional, Callable, Union

from torchdatasets.image.segmentation.base import BaseImageSegmentationDataset


class ImageSegSingleDirDataset(BaseImageSegmentationDataset):
    """Image segmentation dataset from a single directory.
    
    Works for the following directory structure:
    root_dir/
        image1.png
        image1{suffix}.png
        image2.png
        image2{suffix}.png
        ...
    """

    def __init__(
        self,
        src_dir: Union[str, Path],
        suffix: str,
        transform: Optional[Callable] = None,
        return_path: bool = False,
        extensions: Optional[List[str]] = None,
        is_binary: bool = False,
    ):
        """Initialize the dataset from a single directory.

        Args:
            src_dir (Union[str, Path]): Path to the root directory.
            suffix (str): Suffix to be appended to the image name to get the mask name.
            transform (Optional[Callable], optional): Optional transform to be applied to the images. Defaults to None.
            return_path (bool, optional): Whether to return the path to the image. Defaults to False.
            extensions (Optional[List[str]], optional): Optional list of image extensions. Defaults to None.
            is_binary (bool, optional): Whether the dataset is binary. Defaults to False.
        """
        super().__init__(transform=transform, return_path=return_path, extensions=extensions, is_binary=is_binary)
        self.src_dir = Path(src_dir)
        self.suffix = suffix
        self._load_samples()

    def _load_samples(self) -> None:
        """Load the samples from the directory."""

        # assert the source directory exists
        assert self.src_dir.exists(), "Source directory does not exist."

        samples: List[tuple[Path, Path]] = []

        # iterate over all files in the source directory
        for img_path in self.src_dir.glob("*"):

            # skip files with the suffix
            if self.suffix in img_path.stem:
                continue
            
            # skip files that are not images or do not have the correct extension
            if not img_path.is_file() or img_path.suffix.lower() not in self.extensions:
                continue
            
            name = img_path.stem
            ext = img_path.suffix
            
            mask_name = name + (self.suffix or "") + ext
            mask_path = self.src_dir / mask_name
            
            # check if the mask exists
            if mask_path.exists():
                samples.append((img_path, mask_path))
        
        # assert that at least one sample was found
        assert len(samples) > 0, "No valid samples found in the dataset."
        
        self.samples = samples
