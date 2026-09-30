from pathlib import Path
from typing import List, Optional, Callable, Union

from torchdatasets.image.segmentation.base import BaseImageSegmentationDataset
from torchdatasets._internal.io.image import DEFAULT_IMAGE_EXTENSIONS


class ImageSegSingleDirDataset(BaseImageSegmentationDataset):
    """Image segmentation dataset from a single directory.
    
    Works for the following directory structure:
    root_dir/
        image1.jpg
        image1{suffix}.png
        image2.jpg
        image2{suffix}.png
        ...

    A mask is matched by name (image stem + suffix). Its extension may differ from the
    image's (e.g. image1.jpg + image1_mask.png); if several masks share the name, the one
    with the image's own extension is used, otherwise the first in ``mask_extensions`` order.
    """

    def __init__(
        self,
        src_dir: Union[str, Path],
        suffix: str,
        transform: Optional[Callable] = None,
        return_path: bool = False,
        extensions: Optional[List[str]] = None,
        is_binary: bool = False,
        mask_extensions: Optional[List[str]] = None,
    ):
        """Initialize the dataset from a single directory.

        Args:
            src_dir (Union[str, Path]): Path to the root directory.
            suffix (str): Suffix to be appended to the image name to get the mask name.
            transform (Optional[Callable], optional): Optional transform to be applied to the images. Defaults to None.
            return_path (bool, optional): Whether to return the path to the image. Defaults to False.
            extensions (Optional[List[str]], optional): Optional list of image extensions. Defaults to None.
            is_binary (bool, optional): Whether the dataset is binary. Defaults to False.
            mask_extensions (Optional[List[str]], optional): Extensions a mask file may have, in order of preference.
                Defaults to None (the default image extensions).
        """
        super().__init__(transform=transform, return_path=return_path, extensions=extensions, is_binary=is_binary)
        self.src_dir = Path(src_dir)
        self.suffix = suffix
        self.mask_extensions = [ext.lower() for ext in (mask_extensions or DEFAULT_IMAGE_EXTENSIONS)]
        self._load_samples()

    def _load_samples(self) -> None:
        """Load the samples from the directory."""

        # assert the source directory exists
        assert self.src_dir.exists(), "Source directory does not exist."

        samples: List[tuple[Path, Path]] = []

        # index the directory's files by name: stem -> {extension: path}
        masks = self._index_masks(self.src_dir, self.mask_extensions)

        # iterate over all files in the source directory
        for img_path in sorted(self.src_dir.glob("*")):

            # skip mask files (stem ends with the suffix); a name that merely
            # contains the suffix elsewhere is still a valid image
            if img_path.stem.endswith(self.suffix):
                continue
            
            # skip files that are not images or do not have the correct extension
            if not img_path.is_file() or img_path.suffix.lower() not in self.extensions:
                continue
            
            candidates = masks.get(img_path.stem + self.suffix)

            # skip images without a mask
            if not candidates:
                continue

            # prefer a mask with the image's own extension, otherwise follow mask_extensions order
            samples.append((img_path, self._pick_mask(img_path, candidates, self.mask_extensions)))
        
        # assert that at least one sample was found
        assert len(samples) > 0, "No valid samples found in the dataset."
        
        self.samples = samples
