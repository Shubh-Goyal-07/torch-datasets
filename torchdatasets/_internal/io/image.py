import cv2
import numpy as np
from pathlib import Path

# Default image extensions
DEFAULT_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]


def load_image(path: Path) -> np.ndarray:
    """Load image file.

    Args:
        path (Path): Path of the image file.

    Returns:
        np.ndarray: Image array.
    """
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Failed to load image from path: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_mask(path: Path, is_binary: bool = False) -> np.ndarray:
    """Load mask file.

    Args:
        path (Path): Path of the mask file.
        is_binary (bool, optional): Whether the mask is binary. Defaults to False.

    Returns:
        np.ndarray: Mask array.
    """
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to load mask from path: {path}")
    
    if is_binary:
        mask = (mask > 0).astype(np.uint8)        
    
    return mask
