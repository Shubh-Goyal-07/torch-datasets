# Batching notes: when you need your own `collate_fn`

PyTorch's default `DataLoader` batching (`default_collate`) stacks every sample into one
tensor, so all samples in a batch must have the **same shape**. The datasets below can
return samples of different shapes. They don't pad, crop or reshape anything for you,
because the right choice depends on your model, so handle it with a transform or a
custom `collate_fn`.

| Dataset | What can differ between samples | When | What to do |
|---|---|---|---|
| `ImageCSVXLSXDataset` | The label: a list of class indices of different lengths | Multi-label (`label_sep` set) | Turn the lists into multi-hot vectors in a `collate_fn` ([example](#multi-label-labels)) |
| `AudioCSVXLSXDataset` | The label, as above | Multi-label (`label_sep` set) | Same as above |
| `AudioSubdirDataset`, `AudioCSVXLSXDataset` | Waveform length (and channel count, mono vs stereo) | Clips of different durations or channel counts | Pad/crop in a `collate_fn` or a fixed-length transform ([example](#audio-of-different-lengths)) |
| All image datasets (`ImageSubdirDataset`, `ImageCSVXLSXDataset`, `ImageSeg*Dataset`) | Image (and mask) height/width | Images of different sizes | Add a resize/crop transform ([example](#images-of-different-sizes)) |

Tabular datasets always return fixed-size tensors, so the default `DataLoader` works as is.

`return_path=True` adds the file path as a string; the default `collate_fn` turns those
into a list of strings, which is fine. The examples below assume `return_path=False`.

## Image transforms receive uint8 tensors

Image datasets load images as `tv_tensors.Image` tensors: `uint8`, shape `[3, H, W]`
(masks are `tv_tensors.Mask`, shape `[1, H, W]`). Use `torchvision.transforms.v2`:

```python
import torch
from torchvision.transforms import v2

transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToDtype(torch.float32, scale=True),   # replaces ToTensor(): uint8 [0, 255] -> float [0, 1]
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

The old `transforms.ToTensor()` expects a PIL image or numpy array and raises a
`TypeError` on these tensors, so leave it out.

## Images of different sizes

Resize or crop to a fixed size in the transform:

```python
from torchdatasets.image.classification.from_subdir import ImageSubdirDataset

ds = ImageSubdirDataset("data/train", transform=v2.Resize((224, 224)))
```

For segmentation, the transform is called as `transform(image, mask)`, and torchvision v2
applies the same geometry to both (masks are resized with nearest-neighbour, so class ids
stay intact):

```python
from torchdatasets.image.segmentation.from_multidir import ImageSegMultidirDataset

ds = ImageSegMultidirDataset(
    "data/images", "data/masks",
    transform=v2.Compose([v2.RandomResizedCrop((256, 256)), v2.RandomHorizontalFlip()]),
)
```

## Multi-label labels

With `label_sep`, each label is a list such as `[0, 3]`. Convert the lists to multi-hot
vectors (e.g. for `BCEWithLogitsLoss`):

```python
import torch
from torch.utils.data import DataLoader
from torchdatasets.image.classification.from_csv import ImageCSVXLSXDataset


def multi_hot_collate(num_classes):
    def collate(batch):
        inputs = torch.stack([sample for sample, _ in batch])
        targets = torch.zeros(len(batch), num_classes)
        for row, (_, label) in enumerate(batch):
            targets[row, label] = 1.0
        return inputs, targets
    return collate


ds = ImageCSVXLSXDataset("labels.csv", label_sep=",", transform=v2.Resize((224, 224)))
loader = DataLoader(ds, batch_size=32, collate_fn=multi_hot_collate(len(ds.class_to_idx)))
```

For multi-label audio, combine this with the padding shown below.

## Audio of different lengths

Waveforms are `float32` tensors of shape `[channels, samples]`. Either pad each batch to
its longest clip (and keep the true lengths, e.g. for masking):

```python
import torch
import torch.nn.functional as F
from torchdatasets.audio.classification.from_csv import AudioCSVXLSXDataset


def pad_collate(batch):
    waveforms = [waveform for waveform, _ in batch]
    lengths = torch.tensor([w.shape[-1] for w in waveforms])
    max_len = int(lengths.max())
    padded = torch.stack([F.pad(w, (0, max_len - w.shape[-1])) for w in waveforms])
    labels = torch.tensor([label for _, label in batch])
    return padded, labels, lengths


ds = AudioCSVXLSXDataset("clips.csv", sample_rate=16000)
loader = DataLoader(ds, batch_size=16, collate_fn=pad_collate)
```

or make every clip the same length (and channel count) with a transform, after which the
default `DataLoader` works:

```python
import torch.nn.functional as F
from torchdatasets.audio.classification.from_subdir import AudioSubdirDataset


def fixed_length_mono(num_samples):
    def transform(waveform):
        waveform = waveform.mean(dim=0, keepdim=True)          # stereo -> mono
        if waveform.shape[-1] >= num_samples:
            return waveform[..., :num_samples]                 # crop
        return F.pad(waveform, (0, num_samples - waveform.shape[-1]))  # pad with silence
    return transform


ds = AudioSubdirDataset("data/audio", sample_rate=16000, transform=fixed_length_mono(16000 * 5))
loader = DataLoader(ds, batch_size=16)
```

If your clips mix mono and stereo files, downmix (or duplicate channels) before padding;
padding only evens out the length, not the channel count.
