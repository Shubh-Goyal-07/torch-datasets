from torch.utils.data import Dataset
from collections import defaultdict
import torchaudio
from typing import List, Optional, Callable, Tuple, Any, Dict, Union
from torchdatasets._internal.io.audio import DEFAULT_AUDIO_EXTENSIONS, load_audio


class BaseAudioClassificationDataset(Dataset):

    def __init__(
            self,
            transform: Optional[Callable] = None,
            return_path: bool = False,
            extensions: Optional[List[str]] = None,
            sample_rate: Optional[int] = None
        ) -> None:
        self.transform: Optional[Callable] = transform
        self.return_path: bool = return_path
        self.sample_rate: Optional[int] = sample_rate
        self.samples: List[Tuple[str, Union[int, List[int]]]] = []
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}
        self.class_count: Dict[str, int] = {}
        self.extensions: set[str] = set(ext.lower() for ext in (extensions or DEFAULT_AUDIO_EXTENSIONS))


    def __len__(self) -> int:
        return len(self.samples)


    def __getitem__(self, idx: int) -> Tuple[Any, Union[int, List[int]]] | Tuple[Any, Union[int, List[int]], str]:
        path, label = self.samples[idx]
        waveform, _ = load_audio(path, self.sample_rate)

        if self.transform:
            waveform = self.transform(waveform)

        if self.return_path:
            return waveform, label, str(path)
        return waveform, label
    

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
