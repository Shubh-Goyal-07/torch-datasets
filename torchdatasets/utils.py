from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import numpy as np

def train_val_split(
    dataset,
    val_ratio=0.2,
    train_transform = None,
    val_transform = None,
    stratify: bool =False,
    random_state=42):

    indices = np.arange(len(dataset))

    if stratify:
        try:
            labels = [label for _, label in dataset.samples]
        except:
            raise ValueError("Dataset must have labels to use stratify")
        
        train_indices, val_indices = train_test_split(
            indices,
            test_size=val_ratio,
            stratify=labels,
            random_state=random_state,
        )
    else:
        train_indices, val_indices = train_test_split(
            indices,
            test_size=val_ratio,
            random_state=random_state,
        )
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    if train_transform:
        train_dataset.transform = train_transform
    if val_transform:
        val_dataset.transform = val_transform

    return train_dataset, val_dataset
