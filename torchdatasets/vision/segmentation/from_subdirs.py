from pathlib import Path

from .base import BaseImageSegmentationDataset


class ImageSubdirDataset(BaseImageSegmentationDataset):
    def __init__(self,
                 image_dir,
                 mask_dir,
                 transform=None,
                 return_path=False,
                 extensions=None,
                 binary_mask=False,
                 suffix=None
                ):
        super().__init__(transform=transform, extensions=extensions, return_path=return_path, binary_mask=binary_mask)
        
        print("*"*50)
        print(image_dir, mask_dir)
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.suffix = suffix
        self.make_dataset()
        # self.finalize()

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
