import torch
import numpy as np
from PIL import Image
from torchvision import tv_tensors
from typing import Optional, Callable, List, Tuple, Union
from torch.utils.data import Dataset

from torchdatasets._internal.io.image import DEFAULT_IMAGE_EXTENSIONS, load_image, load_mask


class BaseImageSegmentationDataset(Dataset):
    def __init__(
            self,
            transform: Optional[Callable] = None,
            return_path: bool = False,
            extensions: Optional[List[str]] = None,
            is_binary: bool = False
        ) -> None:
        self.transform = transform
        self.return_path = return_path
        self.is_binary = is_binary
        self.extensions = set(ext.lower() for ext in (extensions or DEFAULT_IMAGE_EXTENSIONS))
        self.samples: List[Tuple[str, int]] = []
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, str]]:
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
        raise NotImplementedError("Subclasses must implement _load_samples method to populate self.samples")
