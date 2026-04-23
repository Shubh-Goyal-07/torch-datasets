from torch.utils.data import Dataset
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Callable, Tuple, Any, Dict, Union

from torchdatasets._internal.io.image import DEFAULT_IMAGE_EXTENSIONS, load_image


class BaseImageClassificationDataset(Dataset):
    def __init__(
            self, 
            transform: Optional[Callable] = None,
            return_path: bool = False,
            extensions: Optional[List[str]] = None
        ) -> None:
        self.transform = transform
        self.return_path = return_path
        self.samples: List[Tuple[str, Union[int, List[int]]]] = []
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}
        self.class_count: Dict[str, int] = {}
        self.extensions = extensions if extensions is not None else DEFAULT_IMAGE_EXTENSIONS

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[Any, Union[int, List[int]]] | Tuple[Any, Union[int, List[int]], str]:
        path, label = self.samples[idx]
        img = load_image(path)
        if self.transform:
            img = self.transform(img)
        if self.return_path:
            return img, label, path
        return img, label
    
    def _load_samples(self) -> None:
        raise NotImplementedError("Subclasses must implement _load_samples method to populate self.samples")

    def create_metadata(self) -> None:
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        count = defaultdict(int)
        for _, label in self.samples:
            if isinstance(label, list):
                for lbl in label:
                    count[lbl] += 1
            else:
                count[label] += 1
        self.class_count = dict(count)
