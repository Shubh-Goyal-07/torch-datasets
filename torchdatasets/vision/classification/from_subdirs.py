# src/torchdatasets/vision/classification/from_subdirs.py
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

class ImageSubdirDataset(Dataset):
    """
    Classification dataset where each subdirectory is a class.
    
    root/
      class1/
        img1.jpg
        img2.jpg
      class2/
        img3.jpg
    """
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        self.classes = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        for cls in self.classes:
            for img_path in (self.root / cls).glob("*"):
                if img_path.is_file():
                    self.samples.append((img_path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
