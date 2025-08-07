from torch.utils.data import Dataset
from PIL import Image
from collections import defaultdict


class BaseImageClassificationDataset(Dataset):
    def __init__(self, transform=None, return_path=False):
        self.transform = transform
        self.return_path = return_path
        self.samples = []          # List of (path, label)
        self.class_to_idx = {}     # Dict[str, int]
        self.idx_to_class = {}     # Dict[int, str]
        self.class_count = {}      # Dict[int, int]

    def finalize(self):
        """
        Call after self.samples and self.class_to_idx are fully populated.
        Initializes idx_to_class and class_count.
        """
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        count = defaultdict(int)
        for _, label in self.samples:
            count[label] += 1
        self.class_count = dict(count)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.return_path:
            return image, label, str(path)
        return image, label
