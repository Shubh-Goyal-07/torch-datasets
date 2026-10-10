import cv2
import numpy as np
from pathlib import Path

# Default image extensions. For masks, the order is the preference when several files share a
# name (see the segmentation datasets), so new extensions go at the end.
DEFAULT_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"]


def _decode(path: Path, flags: int, what: str) -> np.ndarray:
    """Read a file's bytes and decode them with OpenCV.

    ``cv2.imread`` can't open paths with non-ASCII characters on Windows; reading the bytes
    with numpy and decoding them in memory works for any path (and decodes identically,
    EXIF orientation included).

    Args:
        path (Path): Path of the file.
        flags (int): OpenCV read flags (e.g. ``cv2.IMREAD_COLOR``).
        what (str): "image" or "mask", for the error message.

    Raises:
        ValueError: If the file can't be read or decoded.

    Returns:
        np.ndarray: The decoded array.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError as err:
        raise ValueError(f"Failed to load {what} from path: {path}") from err
    decoded = cv2.imdecode(data, flags) if data.size else None
    if decoded is None:
        raise ValueError(f"Failed to load {what} from path: {path}")
    return decoded


def load_image(path: Path) -> np.ndarray:
    """Load image file.

    Args:
        path (Path): Path of the image file.

    Raises:
        ValueError: If the file is missing or isn't a readable image.

    Returns:
        np.ndarray: RGB image array (H, W, 3).
    """
    img = _decode(path, cv2.IMREAD_COLOR, "image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path: Path, is_binary: bool = False) -> np.ndarray:
    """Load mask file.

    Args:
        path (Path): Path of the mask file.
        is_binary (bool, optional): Whether the mask is binary. Defaults to False.

    Raises:
        ValueError: If the file is missing or isn't a readable image.

    Returns:
        np.ndarray: Mask array (H, W).
    """
    mask = _decode(path, cv2.IMREAD_GRAYSCALE, "mask")
    if is_binary:
        mask = (mask > 0).astype(np.uint8)
    return mask
