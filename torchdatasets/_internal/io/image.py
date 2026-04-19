import cv2
import numpy as np
from pathlib import Path

DEFAULT_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Failed to load image from path: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
