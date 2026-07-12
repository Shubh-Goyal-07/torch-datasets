import torch
import numpy as np
from PIL import Image
from torchvision import tv_tensors
from typing import Optional, Callable, List, Tuple, Union
from torch.utils.data import Dataset

from torchdatasets._internal.io.image import DEFAULT_IMAGE_EXTENSIONS, load_image, load_mask


# TODO: Add support for torch and albumentations transforms
class BaseImageSegmentationDataset(Dataset):
    """Base class to be used as parent class for image segmentation datasets."""

    def __init__(
            self,
            transform: Optional[Callable] = None,
            return_path: bool = False,
            extensions: Optional[List[str]] = None,
            is_binary: bool = False
        ) -> None:
        """Initialize the dataset.

        Args:
            transform (Optional[Callable], optional): Optional transform to be applied to the images. Defaults to None.
            return_path (bool, optional): Whether to return the path to the image. Defaults to False.
            extensions (Optional[List[str]], optional): Optional list of image extensions. Defaults to None.
            is_binary (bool, optional): Whether the mask is binary. Defaults to False.
        """
        self.transform = transform
        self.return_path = return_path
        self.is_binary = is_binary
        self.extensions = set(ext.lower() for ext in (extensions or DEFAULT_IMAGE_EXTENSIONS))
        self.samples: List[Tuple[str, int]] = []
        
    def __len__(self):
        """Return the number of samples in the dataset.
        
        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, str]]:
        """Get a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, str]]: Image as torch.Tensor and its mask as torch.Tensor. If return_path is True, returns the path to the image.
        """
        img_path, mask_path = self.samples[idx]

        image_np = load_image(img_path)
        mask_np = load_mask(mask_path, is_binary=self.is_binary)

        image = tv_tensors.Image(image_np).permute(2, 0, 1)
        mask = tv_tensors.Mask(mask_np).unsqueeze(0)

        if self.transform:
            image, mask = self.transform(image, mask)

        if self.return_path:
            return image, mask, str(img_path)
        
        return image, mask

    def _load_samples(self) -> None:
        """To be implemented in the children classes.

        Raises:
            NotImplementedError: If the method is not implemented in the children classes.
        """
        raise NotImplementedError("Subclasses must implement _load_samples method to populate self.samples")
