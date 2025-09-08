from torch.utils.data import Dataset
from PIL import Image
from collections import defaultdict

from torchdatasets.vision.constants import DEFAULT_IMAGE_EXTENSIONS


class BaseImageClassificationDataset(Dataset):
    def __init__(self, transform=None, return_path=False, extensions=None):
        self.transform = transform
        self.return_path = return_path
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.class_count = {}
        self.extensions = set(ext.lower() for ext in (extensions or DEFAULT_IMAGE_EXTENSIONS))

    def finalize(self):
        """
        Call after self.samples and self.class_to_idx are fully populated.
        Initializes idx_to_class and class_count.
        """
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        count = defaultdict(int)
        for _, label in self.samples:
            if isinstance(label, list):
                for lbl in label:
                    count[lbl] += 1 
            else: 
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