from pathlib import Path
from typing import List, Optional, Callable, Union

from torchdatasets.image.segmentation.base import BaseImageSegmentationDataset


class ImageSegMultidirDataset(BaseImageSegmentationDataset):
    def __init__(
        self,
        image_dir: Union[str, Path],
        mask_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        return_path: bool = False,
        extensions: Optional[List[str]] = None,
        binary_mask: bool = False,
        suffix: Optional[str] = None
    ):
        super().__init__(transform=transform, return_path=return_path, extensions=extensions, binary_mask=binary_mask)
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.suffix = suffix
        self.make_dataset()

    def make_dataset(self):
        samples = []

        for img_path in self.image_dir.glob("*"):
            if not img_path.is_file() or img_path.suffix.lower() not in self.extensions:
                continue
            
            name = img_path.stem
            ext = img_path.suffix
            mask_name = name + (self.suffix or "") + ext
            mask_path = self.mask_dir / mask_name
            
            if mask_path.exists():
                samples.append((img_path, mask_path))
        
        assert len(samples) > 0, "No valid samples found in the dataset."
        self.samples = samples
