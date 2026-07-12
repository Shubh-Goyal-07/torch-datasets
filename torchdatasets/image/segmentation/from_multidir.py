from pathlib import Path
from typing import List, Optional, Callable, Union

from torchdatasets.image.segmentation.base import BaseImageSegmentationDataset


class ImageSegMultidirDataset(BaseImageSegmentationDataset):
    """Image Segmentation Dataset from Two Directories.

    Works for the following directory structure:
    image_dir/
        image1.png
        image2.png
        ...
    mask_dir/
        image1{suffix}.png
        image2{suffix}.png
        ...
    """

    def __init__(
        self,
        image_dir: Union[str, Path],
        mask_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        return_path: bool = False,
        extensions: Optional[List[str]] = None,
        is_binary: bool = False,
        suffix: Optional[str] = None
    ):
        """Initialize the dataset from two directories.
        
        Args:
            image_dir (Union[str, Path]): Path to the image directory.
            mask_dir (Union[str, Path]): Path to the mask directory.
            transform (Optional[Callable], optional): Optional transform to be applied to the images. Defaults to None.
            return_path (bool, optional): Whether to return the path to the image. Defaults to False.
            extensions (Optional[List[str]], optional): Optional list of image extensions. Defaults to None.
            is_binary (bool, optional): Whether the dataset is binary. Defaults to False.
            suffix (Optional[str], optional): Optional suffix to be appended to the image name to get the mask name. Defaults to None.
        """
        super().__init__(transform=transform, return_path=return_path, extensions=extensions, is_binary=is_binary)
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.suffix = suffix
        self._load_samples()

    def _load_samples(self):
        """Load the samples from the directory."""

        # assert the source directory exists
        assert self.image_dir.exists(), "Image directory does not exist."
        assert self.mask_dir.exists(), "Mask directory does not exist."
        
        samples: List[tuple[Path, Path]] = []

        # iterate over all files in the image directory
        for img_path in self.image_dir.glob("*"):
            # skip files that are not images or do not have the correct extension
            if not img_path.is_file() or img_path.suffix.lower() not in self.extensions:
                continue
            
            name = img_path.stem
            ext = img_path.suffix
            mask_name = name + (self.suffix or "") + ext
            mask_path = self.mask_dir / mask_name
            
            # check if the mask exists
            if mask_path.exists():
                samples.append((img_path, mask_path))
        
        # assert that at least one sample was found
        assert len(samples) > 0, "No valid samples found in the dataset."

        self.samples = samples
