from torch.utils.data import Dataset
from collections import defaultdict
import torchaudio

from torchdatasets.audio.constants import DEFAULT_AUDIO_EXTENSIONS


class BaseAudioClassificationDataset(Dataset):
    def __init__(self, transform=None, return_path=False, extensions=None, sample_rate=None):
        self.transform = transform
        self.return_path = return_path
        self.sample_rate = sample_rate  # New: user-specified sample rate
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.class_count = {}
        self.extensions = set(ext.lower() for ext in (extensions or DEFAULT_AUDIO_EXTENSIONS))

    def finalize(self):
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
        waveform, orig_sr = torchaudio.load(path)

        if self.sample_rate and orig_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sr, self.sample_rate)
            waveform = resampler(waveform)

        if self.transform:
            waveform = self.transform(waveform)

        if self.return_path:
            return waveform, label, str(path)
        return waveform, label
