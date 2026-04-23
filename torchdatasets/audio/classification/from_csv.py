from pathlib import Path
import pandas as pd
from typing import List, Optional, Callable, Union
from torchdatasets.audio.classification.base import BaseAudioClassificationDataset
from torchdatasets._internal.io.common import load_csv_or_excel


class AudioCSVXLSXDataset(BaseAudioClassificationDataset):

    def __init__(
            self,
            file_path: Union[str, Path],
            transform: Optional[Callable] = None,
            return_path: bool = False,
            sample_rate: Optional[int] = None,
            extensions: Optional[List[str]] = None,
            path_col: str = "path",
            label_col: str = "label",
            label_sep: Optional[str] = None
        ) -> None:
        super().__init__(transform=transform, return_path=return_path, extensions=extensions, sample_rate=sample_rate)
        self.file_path: Path = Path(file_path)
        self.path_col: str = path_col
        self.label_col: str = label_col
        self.label_sep: Optional[str] = label_sep
        self._load_samples()
        self.create_metadata()


    def _load_samples(self) -> None:

        df = load_csv_or_excel(self.file_path)

        assert self.path_col in df.columns and self.label_col in df.columns, f"File must contain '{self.path_col}' and '{self.label_col}' columns."

        samples: List[tuple[Path, Union[str, List[str]]]] = []
        classes: set[str] = set()

        for _, row in df.iterrows():
            audio_path = Path(row[self.path_col])
            label_cell = row[self.label_col]

            if self.label_sep is None:
                labels: List[str] = [str(label_cell)]
            else:
                labels = [lbl.strip() for lbl in str(label_cell).split(self.label_sep)]

            if audio_path.suffix.lower() in self.extensions and audio_path.is_file():
                samples.append((audio_path, labels if self.label_sep else labels[0]))
                classes.update(labels)

        assert len(samples) > 0, "No valid audio entries found in CSV/XLSX."

        classes = sorted(classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        def encode_labels(lbl: Union[str, List[str]]) -> Union[int, List[int]]:
            if isinstance(lbl, list):
                return [self.class_to_idx[x] for x in lbl]
            return self.class_to_idx[lbl]

        self.samples = [(audio_path, encode_labels(labels)) for audio_path, labels in samples]
