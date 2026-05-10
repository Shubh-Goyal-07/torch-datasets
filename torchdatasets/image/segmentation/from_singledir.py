from pathlib import Path
from typing import List, Optional, Callable, Union

from torchdatasets.image.segmentation.base import BaseImageSegmentationDataset


class ImageSegSingleDirDataset(BaseImageSegmentationDataset):
    def __init__(
        self,
        src_dir: Union[str, Path],
        suffix: str,
        transform: Optional[Callable] = None,
        return_path: bool = False,
        extensions: Optional[List[str]] = None,
        binary_mask: bool = False,
    ):
        super().__init__(transform=transform, return_path=return_path, extensions=extensions, binary_mask=binary_mask)
        self.src_dir = Path(src_dir)
        self.suffix = suffix
        self.make_dataset()

    def make_dataset(self):
        samples = []

        for img_path in self.src_dir.glob("*"):
            if self.suffix in img_path.stem:
                continue
            
            if not img_path.is_file() or img_path.suffix.lower() not in self.extensions:
                continue
            
            name = img_path.stem
            ext = img_path.suffix
            
            mask_name = name + (self.suffix or "") + ext
            mask_path = self.src_dir / mask_name
            
            if mask_path.exists():
                samples.append((img_path, mask_path))
        
        assert len(samples) > 0, "No valid samples found in the dataset."
        self.samples = samples
