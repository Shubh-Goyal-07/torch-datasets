from pathlib import Path
from .base import BaseAudioClassificationDataset

class AudioSingleDirDataset(BaseAudioClassificationDataset):
    def __init__(self, root, transform=None, sample_rate=None, label_map=None, delimiter="_"):
        super().__init__(transform=transform, sample_rate=sample_rate)
        self.root = Path(root)
        self.delimiter = delimiter
        self.make_dataset(label_map=label_map)
        self.finalize()

    def make_dataset(self, label_map=None):
        samples = []
        classes = set()

        for audio_path in self.root.glob("*"):
            if audio_path.is_file() and audio_path.suffix.lower() in self.extensions:
                class_name = audio_path.stem.split(self.delimiter)[0]
                classes.add(class_name)
                samples.append((audio_path, class_name))

        classes = sorted(classes)
        if label_map is None:
            self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        else:
            self.class_to_idx = label_map
            assert all(cls in label_map for _, cls in samples), "Some classes missing from label_map"

        self.samples = [(p, self.class_to_idx[c]) for p, c in samples]
        assert len(self.samples) > 0, "No valid audio files found."
