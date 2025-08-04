# src/torchdatasets/vision/classification/from_singledir.py
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

class ImageSingleDirDataset(Dataset):
    """
    Classification dataset where all images are in one folder.
    Labels inferred via a user-provided mapping or filename prefix.

    Example filename convention: cat_001.jpg, dog_014.jpg
    Or provide label_map: {"cat": 0, "dog": 1}
    """
    def __init__(self, root, transform=None, label_map=None, delimiter="_"):
        self.root = Path(root)
        self.transform = transform
        self.delimiter = delimiter

        self.samples = []
        self.classes = set()

        for img_path in self.root.glob("*"):
            if img_path.is_file():
                # infer class from filename prefix
                class_name = img_path.stem.split(delimiter)[0]
                self.classes.add(class_name)
                self.samples.append((img_path, class_name))

        self.classes = sorted(list(self.classes))
        if label_map is None:
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        else:
            self.class_to_idx = label_map

        self.samples = [(p, self.class_to_idx[c]) for p, c in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
