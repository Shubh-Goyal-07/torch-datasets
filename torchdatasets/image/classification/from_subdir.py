from pathlib import Path
from typing import List, Optional, Callable, Union

from torchdatasets.image.classification.base import BaseImageClassificationDataset


# TODO: (Check viability) Add support for multi-label classification (comma/semicolon separated labels) in this dataset as well, similar to the CSV/XLSX version. 
class ImageSubdirDataset(BaseImageClassificationDataset):
    def __init__(
            self,
            root_dir: Union[str, Path],
            transform: Optional[Callable] = None,
            extensions: Optional[List[str]] = None,
            return_path: bool = False
        ) -> None:
        super().__init__(transform=transform, return_path=return_path, extensions=extensions)
        self.root: Path = Path(root_dir)
        self._load_samples()
        self.create_metadata()

    def _load_samples(self) -> None:
        classes: List[str] = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        samples: List[tuple[Path, int]] = []

        for cls in classes:
            for img_path in (self.root / cls).glob("*"):
                if img_path.is_file() and img_path.suffix.lower() in self.extensions:
                    samples.append((img_path, self.class_to_idx[cls]))

        self.samples = samples
        assert len(self.samples) > 0, "No valid samples found in the dataset."
