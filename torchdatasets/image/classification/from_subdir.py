from pathlib import Path
from typing import List, Optional, Callable, Union
from torchdatasets.image.classification.base import BaseImageClassificationDataset


# TO DO: (Check viability) Add support for multi-label classification (comma/semicolon separated labels) in this dataset as well, similar to the CSV/XLSX version. 

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



# MULTI-CLASS/LABEL SUPPORT ADDED - lint testing completed, runtime tests to be done 

class MultiClassImageSubdirDataset(BaseImageClassificationDataset):

    def __init__(
            self,
            root_dir: Union[str, Path],
            transform: Optional[Callable] = None,
            extensions: Optional[List[str]] = None,
            return_path: bool = False,
            label_sep: Optional[str] = None
        ) -> None:
        super().__init__(transform=transform, return_path=return_path, extensions=extensions)
        self.root: Path = Path(root_dir)
        self.label_sep: Optional[str] = label_sep
        self._load_samples()
        self.create_metadata()


    def _load_samples(self) -> None:
        class_dirs: List[str] = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        samples: List[tuple[Path, Union[str, List[str]]]] = []
        classes: set[str] = set()

        for cls in class_dirs:
            if self.label_sep is None:
                labels: List[str] = [cls]
            else:
                labels = [lbl.strip() for lbl in str(cls).split(self.label_sep)]

            for img_path in (self.root / cls).glob("*"):
                if img_path.is_file() and img_path.suffix.lower() in self.extensions:
                    samples.append((img_path, labels if self.label_sep else labels[0]))
                    classes.update(labels)

        classes = sorted(classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        def encode_labels(lbl: Union[str, List[str]]) -> Union[int, List[int]]:
            if isinstance(lbl, list):
                return [self.class_to_idx[x] for x in lbl]
            return self.class_to_idx[lbl]

        self.samples = [(img_path, encode_labels(labels)) for img_path, labels in samples]
        assert len(self.samples) > 0, "No valid samples found in the dataset."

