from pathlib import Path

from .base import BaseImageClassificationDataset

class ImageSubdirDataset(BaseImageClassificationDataset):
    def __init__(self, root, transform=None, extensions=None, return_path=False):
        super().__init__(transform=transform, extensions=extensions, return_path=return_path)
        self.root = Path(root)
        self.make_dataset()
        self.finalize()

    def make_dataset(self):
        classes = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        samples = []

        for cls in classes:
            for img_path in (self.root / cls).glob("*"):
                if img_path.is_file() and img_path.suffix.lower() in self.extensions:
                    samples.append((img_path, self.class_to_idx[cls]))

        self.samples = samples
        assert len(self.samples) > 0, "No valid samples found in the dataset."
