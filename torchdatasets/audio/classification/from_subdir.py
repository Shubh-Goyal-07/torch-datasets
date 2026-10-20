from pathlib import Path
from typing import List, Optional, Callable, Union

from torchdatasets.audio.classification.base import BaseAudioClassificationDataset
from torchdatasets._internal.subdir_labels import samples_from_subfolders


class AudioSubdirDataset(BaseAudioClassificationDataset):
    """Audio classification dataset from sub-directory structure. Similar to PyTorch's ImageFolder.
    
    Works for the following directory structure:
    root_dir/
        class1/
            audio1.wav
            audio2.mp3
            ...
        class2/
            audio1.wav
            audio2.mp3
            ...
        ...

    With ``label_sep`` set, a folder name can list several labels (multi-label classification):
    root_dir/
        cat/
        cat,dog/        (with label_sep=",", these audio files are both "cat" and "dog")
        dog/
    The classes are then the individual labels, and each label is a list of class indices.

    Hidden files and folders (names starting with ``.``, such as ``.ipynb_checkpoints`` or
    macOS ``._audio1.wav`` files) are skipped.
    """

    def __init__(
            self,
            root_dir: Union[str, Path],
            *,
            transform: Optional[Callable] = None,
            extensions: Optional[List[str]] = None,
            sample_rate: Optional[int] = None,
            return_path: bool = False,
            label_sep: Optional[str] = None
        ) -> None:
        """Initialize the dataset from a sub-directory structure.

        Args:
            root_dir (Union[str, Path]): Path to the root directory, with one sub-folder per class.
            transform (Optional[Callable], optional): Optional transform to be applied to the waveforms. Defaults to None.
            extensions (Optional[List[str]], optional): Optional list of audio extensions. Defaults to None.
            sample_rate (Optional[int], optional): Resample every clip to this rate. Defaults to None (keep the file's rate).
            return_path (bool, optional): Whether to return the path to the audio file. Defaults to False.
            label_sep (Optional[str], optional): Separator between several labels in one folder name
                (e.g. ``","``), for multi-label classification. Defaults to None (each folder is one class).
        """
        super().__init__(transform=transform, return_path=return_path, extensions=extensions, sample_rate=sample_rate)
        self.root: Path = Path(root_dir)
        self.label_sep: Optional[str] = label_sep
        self._load_samples()
        self.create_metadata()

    def _load_samples(self) -> None:
        """Load samples from the sub-directory structure.

        Raises:
            ValueError: If label_sep is an empty string, or a folder name has no label between the separators.
            FileNotFoundError: If root_dir does not exist, has no class folders, or no audio files.
        """
        self.samples, self.class_to_idx = samples_from_subfolders(self.root, self.extensions, self.label_sep)
