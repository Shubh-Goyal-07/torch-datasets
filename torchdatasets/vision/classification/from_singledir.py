from pathlib import Path

from .base import BaseImageClassificationDataset

class ImageSingleDirDataset(BaseImageClassificationDataset):
    def __init__(self, root, transform=None, label_map=None, delimiter="_"):
        super().__init__(transform=transform)
        self.root = Path(root)
        self.delimiter = delimiter
        self.make_dataset(label_map=label_map)
        self.finalize()

    def make_dataset(self, label_map=None):
        samples = []
        classes = set()

        for img_path in self.root.glob("*"):
            if img_path.is_file() and img_path.suffix.lower() in self.extensions:
                class_name = img_path.stem.split(self.delimiter)[0]
                classes.add(class_name)
                samples.append((img_path, class_name))

        classes = sorted(classes)
        if label_map is None:
            self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        else:
            self.class_to_idx = label_map
            assert all(cls in label_map for _, cls in samples), "Some classes missing from label_map"

        self.samples = [(p, self.class_to_idx[c]) for p, c in samples]
        assert len(self.samples) > 0, "No valid samples found in the dataset."
