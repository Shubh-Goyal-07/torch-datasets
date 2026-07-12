import torch
from torch.utils.data import Dataset
from collections import defaultdict
from typing import List, Optional, Callable, Tuple, Any, Dict, Union

from torchdatasets._internal.io.audio import DEFAULT_AUDIO_EXTENSIONS, load_audio


class BaseAudioClassificationDataset(Dataset):
    """Base class to be used as parent class for audio classification datasets"""

    def __init__(
            self,
            transform: Optional[Callable] = None,
            return_path: bool = False,
            extensions: Optional[List[str]] = None,
            sample_rate: Optional[int] = None
        ) -> None:
        """Initialize the dataset.
        
        Args:
            transform (Optional[Callable], optional): Optional transform to be applied to the audio samples. Defaults to None.
            return_path (bool, optional): Whether to return the path to the audio sample. Defaults to False.
            extensions (Optional[List[str]], optional): Optional list of audio extensions. Defaults to None.
            sample_rate (Optional[int], optional): Optional sample rate to be used for the audio samples. Defaults to None.
        """
        self.transform = transform
        self.return_path = return_path
        self.sample_rate = sample_rate
        self.samples: List[Tuple[str, Union[int, List[int]]]] = []
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}
        self.class_count: Dict[str, int] = {}
        self.extensions: set[str] = set(ext.lower() for ext in (extensions or DEFAULT_AUDIO_EXTENSIONS))

    def __len__(self) -> int:
        """Return the number of samples in the dataset.
        
        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[int, List[int]]] | Tuple[torch.Tensor, Union[int, List[int]], str]:
        """Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve.
            
        Returns:
            Tuple[torch.Tensor, Union[int, List[int]]] | Tuple[torch.Tensor, Union[int, List[int]], str]: Audio sample as torch.Tensor and its label. If return_path is True, returns the path to the audio sample.
        """
        path, label = self.samples[idx]
        waveform, sr = load_audio(path, self.sample_rate)

        if self.transform:
            waveform = self.transform(waveform)

        if self.return_path:
            return waveform, label, str(path)
        return waveform, label

    def encode_labels(self, lbl: Union[str, List[str]]) -> Union[int, List[int]]:
        """Encode labels to integer labels.
        
        Args:
            lbl (Union[str, List[str]]): Labels to encode.
            
        Returns:
            Union[int, List[int]]: Encoded labels.
        """
        if isinstance(lbl, list):
            return [self.class_to_idx[x] for x in lbl]
        return self.class_to_idx[lbl]
    
    def _load_samples(self) -> None:
        """To be implemented in the children classes.

        Raises:
            NotImplementedError: If the method is not implemented in the children classes.
        """
        raise NotImplementedError("Subclasses must implement _load_samples method to populate self.samples")

    def create_metadata(self) -> None:
        """Create metadata for the dataset, it includes:
        1. idx_to_class: Dictionary mapping class indices to class names.
        2. class_count: Dictionary mapping class names to the number of samples in each class.
        """
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        count = defaultdict(int)
        
        for _, label in self.samples:
            # If the label is a list, it means it's a multi-label classification dataset
            if isinstance(label, list):
                for lbl in label:
                    count[lbl] += 1
            else:
                count[label] += 1
        
        self.class_count = dict(count)

