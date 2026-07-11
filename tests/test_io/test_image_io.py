"""Tests for torchdatasets._internal.io.image — image and mask loading."""

import numpy as np
import pytest

from torchdatasets._internal.io.image import DEFAULT_IMAGE_EXTENSIONS, load_image, load_mask


class TestLoadImage:
    """Tests for load_image."""

    def test_returns_rgb_ndarray(self, tmp_image_factory):
        path = tmp_image_factory("test.png", size=(16, 16), color=(100, 150, 200))
        img = load_image(path)
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3
        assert img.shape[2] == 3  # RGB channels

    def test_pixel_values_are_correct(self, tmp_image_factory):
        color = (100, 150, 200)
        path = tmp_image_factory("test.png", size=(4, 4), color=color)
        img = load_image(path)
        # Check the center pixel matches expected RGB
        np.testing.assert_array_equal(img[2, 2], list(color))

    def test_spatial_dimensions_match(self, tmp_image_factory):
        w, h = 32, 24
        path = tmp_image_factory("test.png", size=(w, h))
        img = load_image(path)
        assert img.shape[:2] == (h, w)  # cv2 returns (H, W, C)

    def test_missing_image_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Failed to load image"):
            load_image(tmp_path / "nonexistent.png")

    def test_supports_jpg(self, tmp_image_factory):
        path = tmp_image_factory("test.jpg")
        img = load_image(path)
        assert isinstance(img, np.ndarray)


class TestLoadMask:
    """Tests for load_mask."""

    def test_returns_grayscale_ndarray(self, tmp_mask_factory):
        path = tmp_mask_factory("mask.png", value=100)
        mask = load_mask(path)
        assert isinstance(mask, np.ndarray)
        assert mask.ndim == 2  # Grayscale = 2D

    def test_preserves_pixel_values(self, tmp_mask_factory):
        path = tmp_mask_factory("mask.png", size=(4, 4), value=200)
        mask = load_mask(path)
        assert mask[0, 0] == 200

    def test_binary_mode_binarizes(self, tmp_mask_factory):
        path = tmp_mask_factory("mask.png", size=(4, 4), value=128)
        mask = load_mask(path, is_binary=True)
        assert set(np.unique(mask)).issubset({0, 1})
        assert mask[0, 0] == 1  # 128 > 0 → 1

    def test_binary_mode_zero_stays_zero(self, tmp_mask_factory):
        path = tmp_mask_factory("mask.png", size=(4, 4), value=0)
        mask = load_mask(path, is_binary=True)
        assert np.all(mask == 0)

    def test_missing_mask_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Failed to load mask"):
            load_mask(tmp_path / "nonexistent.png")


class TestDefaultExtensions:
    """Tests for the extension constants."""

    def test_default_extensions_are_lowercase(self):
        for ext in DEFAULT_IMAGE_EXTENSIONS:
            assert ext == ext.lower()
            assert ext.startswith(".")
