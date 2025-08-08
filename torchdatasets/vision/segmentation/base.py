import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from typing import Callable

from torchdatasets.vision.constants import DEFAULT_IMAGE_EXTENSIONS


class BaseImageSegmentationDataset(Dataset):
    def __init__(self, transform=None, return_path=False, extensions=None, binary_mask=False):
        self._validate_transform(transform)
        
        self.transform = transform
        self.return_path = return_path
        self.binary_mask =binary_mask
        self.extensions = set(ext.lower() for ext in (extensions or DEFAULT_IMAGE_EXTENSIONS))
        
        self.samples = []

    def _validate_transform(self, transform):
        if transform is None:
            return
        elif not isinstance(transform, Callable):
            raise ValueError("Transform must be an albumentations callable")

        dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
        dummy_mask = np.zeros((10, 10, 3), dtype=np.uint8)
        out = transform(image=dummy_img, mask=dummy_mask)
        
        if not isinstance(out, dict) or "image" not in out or "mask" not in out:
            raise ValueError("Please check transform, it should return image and mask keys")
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        
        mask = Image.open(mask_path)
        mask = np.array(mask.covert("1" if self.binary_mask else "L"))

        if self.transform:
            out = self.transform(image=image, mask=mask)
            image = out["image"]
            mask = out["mask"]
        if self.return_path:
            return image, mask, str(img_path)
        return image, mask
