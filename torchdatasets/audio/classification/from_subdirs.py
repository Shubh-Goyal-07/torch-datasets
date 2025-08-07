from pathlib import Path
from .base import BaseAudioClassificationDataset

class AudioSubdirDataset(BaseAudioClassificationDataset):
    def __init__(self, root, transform=None, sample_rate=None):
        super().__init__(transform=transform, sample_rate=sample_rate)
        self.root = Path(root)
        self.make_dataset()
        self.finalize()

    def make_dataset(self):
        classes = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        samples = []

        for cls in classes:
            for audio_path in (self.root / cls).glob("*"):
                if audio_path.is_file() and audio_path.suffix.lower() in self.extensions:
                    samples.append((audio_path, self.class_to_idx[cls]))

        assert len(samples) > 0, "No valid audio samples found."
        self.samples = samples
