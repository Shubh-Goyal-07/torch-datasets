from pathlib import Path
from typing import List, Optional, Callable, Union

from torchdatasets.image.segmentation.base import BaseImageSegmentationDataset
from torchdatasets._internal.io.image import DEFAULT_IMAGE_EXTENSIONS


class ImageSegMultidirDataset(BaseImageSegmentationDataset):
    """Image Segmentation Dataset from Two Directories.

    Works for the following directory structure:
    image_dir/
        image1.jpg
        image2.jpg
        ...
    mask_dir/
        image1{suffix}.png
        image2{suffix}.png
        ...

    A mask is matched by name (image stem + suffix). Its extension may differ from the
    image's (e.g. image1.jpg + image1.png); if several masks share the name, the one with
    the image's own extension is used, otherwise the first in ``mask_extensions`` order.
    """

    def __init__(
        self,
        image_dir: Union[str, Path],
        mask_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        return_path: bool = False,
        extensions: Optional[List[str]] = None,
        is_binary: bool = False,
        suffix: Optional[str] = None,
        mask_extensions: Optional[List[str]] = None
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
            mask_extensions (Optional[List[str]], optional): Extensions a mask file may have, in order of preference.
                Defaults to None (the default image extensions).
        """
        super().__init__(transform=transform, return_path=return_path, extensions=extensions, is_binary=is_binary)
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.suffix = suffix
        self.mask_extensions = [ext.lower() for ext in (mask_extensions or DEFAULT_IMAGE_EXTENSIONS)]
        self._load_samples()

    def _load_samples(self):
        """Load the samples from the directory."""

        # assert the source directory exists
        assert self.image_dir.exists(), "Image directory does not exist."
        assert self.mask_dir.exists(), "Mask directory does not exist."
        
        samples: List[tuple[Path, Path]] = []

        # index the mask directory once: mask stem -> {extension: path}
        masks = self._index_masks(self.mask_dir, self.mask_extensions)

        # iterate over all files in the image directory
        for img_path in sorted(self.image_dir.glob("*")):
            # skip files that are not images or do not have the correct extension
            if not img_path.is_file() or img_path.suffix.lower() not in self.extensions:
                continue
            
            candidates = masks.get(img_path.stem + (self.suffix or ""))

            # skip images without a mask
            if not candidates:
                continue

            # prefer a mask with the image's own extension, otherwise follow mask_extensions order
            samples.append((img_path, self._pick_mask(img_path, candidates, self.mask_extensions)))
        
        # assert that at least one sample was found
        assert len(samples) > 0, "No valid samples found in the dataset."

        self.samples = samples
