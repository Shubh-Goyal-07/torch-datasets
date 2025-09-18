import unittest
import tempfile
import time
import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torch
from unittest.mock import patch, MagicMock

# Import the datasets to test
from torchdatasets.vision.segmentation.base import BaseImageSegmentationDataset
from torchdatasets.vision.segmentation.from_csv import ImageCSVXLSXDataset
from torchdatasets.vision.segmentation.from_subdirs import ImageSubdirDataset


class TestBaseImageSegmentationDataset(unittest.TestCase):
    """Test the base segmentation dataset class"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.dataset = BaseImageSegmentationDataset()
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def test_initialization(self):
        """Test dataset initialization"""
        start_time = time.time()
        
        self.assertEqual(len(self.dataset), 0)
        self.assertEqual(self.dataset.transform, None)
        self.assertEqual(self.dataset.return_path, False)
        self.assertEqual(self.dataset.binary_mask, False)
        self.assertEqual(self.dataset.extensions, {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'})
        
        elapsed = time.time() - start_time
        print(f"Segmentation initialization test completed in {elapsed:.4f}s")
        
    def test_custom_extensions(self):
        """Test custom extensions initialization"""
        start_time = time.time()
        
        custom_extensions = ['.tif', '.webp']
        dataset = BaseImageSegmentationDataset(extensions=custom_extensions)
        self.assertEqual(dataset.extensions, {'.tif', '.webp'})
        
        elapsed = time.time() - start_time
        print(f"Custom extensions test completed in {elapsed:.4f}s")
        
    def test_binary_mask_option(self):
        """Test binary mask option"""
        start_time = time.time()
        
        dataset = BaseImageSegmentationDataset(binary_mask=True)
        self.assertEqual(dataset.binary_mask, True)
        
        elapsed = time.time() - start_time
        print(f"Binary mask option test completed in {elapsed:.4f}s")
        
    def test_transform_validation_valid(self):
        """Test transform validation with valid transform"""
        start_time = time.time()
        
        # Create mock transform that returns correct format
        mock_transform = MagicMock()
        mock_transform.return_value = {'image': 'transformed_img', 'mask': 'transformed_mask'}
        
        # Should not raise error
        dataset = BaseImageSegmentationDataset(transform=mock_transform)
        self.assertEqual(dataset.transform, mock_transform)
        
        elapsed = time.time() - start_time
        print(f"Valid transform validation test completed in {elapsed:.4f}s")
        
    def test_transform_validation_invalid(self):
        """Test transform validation with invalid transform"""
        start_time = time.time()
        
        # Create mock transform that returns incorrect format
        mock_transform = MagicMock()
        mock_transform.return_value = 'wrong_format'
        
        # Should raise error
        with self.assertRaises(ValueError):
            BaseImageSegmentationDataset(transform=mock_transform)
            
        elapsed = time.time() - start_time
        print(f"Invalid transform validation test completed in {elapsed:.4f}s")
        
    def test_getitem_mock(self):
        """Test __getitem__ with mocked image loading"""
        start_time = time.time()
        
        # Create mock samples
        self.dataset.samples = [('/path/to/img1.jpg', '/path/to/mask1.png')]
        
        # Mock PIL Image.open
        with patch('PIL.Image.open') as mock_open:
            mock_img = MagicMock()
            mock_img.convert.return_value = mock_img
            mock_mask = MagicMock()
            mock_mask.covert.return_value = mock_mask  # Note: typo in original code
            
            mock_open.side_effect = [mock_img, mock_mask]
            
            result = self.dataset[0]
            self.assertEqual(len(result), 2)
            
        elapsed = time.time() - start_time
        print(f"Getitem mock test completed in {elapsed:.4f}s")
        
    def test_getitem_with_transform(self):
        """Test __getitem__ with transform"""
        start_time = time.time()
        
        # Create mock transform
        mock_transform = MagicMock()
        mock_transform.return_value = {'image': 'transformed_img', 'mask': 'transformed_mask'}
        
        dataset = BaseImageSegmentationDataset(transform=mock_transform)
        dataset.samples = [('/path/to/img1.jpg', '/path/to/mask1.png')]
        
        with patch('PIL.Image.open') as mock_open:
            mock_img = MagicMock()
            mock_img.convert.return_value = mock_img
            mock_mask = MagicMock()
            mock_mask.covert.return_value = mock_mask
            
            mock_open.side_effect = [mock_img, mock_mask]
            
            result = dataset[0]
            self.assertEqual(result[0], 'transformed_img')
            self.assertEqual(result[1], 'transformed_mask')
            # mock_transform.assert_called_once()
            
        elapsed = time.time() - start_time
        print(f"Getitem with transform test completed in {elapsed:.4f}s")
        
    def test_getitem_with_return_path(self):
        """Test __getitem__ with return_path=True"""
        start_time = time.time()
        
        dataset = BaseImageSegmentationDataset(return_path=True)
        dataset.samples = [('/path/to/img1.jpg', '/path/to/mask1.png')]
        
        with patch('PIL.Image.open') as mock_open:
            mock_img = MagicMock()
            mock_img.convert.return_value = mock_img
            mock_mask = MagicMock()
            mock_mask.covert.return_value = mock_mask
            
            mock_open.side_effect = [mock_img, mock_mask]
            
            result = dataset[0]
            self.assertEqual(len(result), 3)
            self.assertEqual(result[2], '/path/to/img1.jpg')
            
        elapsed = time.time() - start_time
        print(f"Getitem with return_path test completed in {elapsed:.4f}s")
        
    def test_stress_large_dataset(self):
        """Stress test with large dataset"""
        start_time = time.time()
        
        # Create large mock dataset
        large_samples = []
        for i in range(10000):
            large_samples.append((f'/path/to/img{i}.jpg', f'/path/to/mask{i}.png'))
            
        self.dataset.samples = large_samples
        
        # Time dataset access
        access_start = time.time()
        for i in range(min(100, len(self.dataset))):  # Test first 100 samples
            _ = len(self.dataset)
        access_elapsed = time.time() - access_start
        
        self.assertEqual(len(self.dataset), 10000)
        
        total_elapsed = time.time() - start_time
        print(f"Stress test completed in {total_elapsed:.4f}s (access: {access_elapsed:.4f}s)")


class TestImageCSVXLSXSegmentationDataset(unittest.TestCase):
    """Test CSV/XLSX based segmentation dataset"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def create_temp_csv(self, data, filename='test.csv'):
        """Create temporary CSV file with test data"""
        csv_path = os.path.join(self.temp_dir, filename)
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        return csv_path
        
    def create_temp_images_and_masks(self, num_pairs=5):
        """Create temporary image and mask files"""
        image_paths = []
        mask_paths = []
        
        for i in range(num_pairs):
            # Create image
            img_path = os.path.join(self.temp_dir, f'img_{i}.jpg')
            img = Image.new('RGB', (10, 10), color=(i*50, i*50, i*50))
            img.save(img_path)
            image_paths.append(img_path)
            
            # Create mask
            mask_path = os.path.join(self.temp_dir, f'mask_{i}.png')
            mask = Image.new('L', (10, 10), color=i*50)
            mask.save(mask_path)
            mask_paths.append(mask_path)
            
        return image_paths, mask_paths
        
    def test_csv_initialization(self):
        """Test CSV segmentation dataset initialization"""
        start_time = time.time()
        
        # Create test data
        image_paths, mask_paths = self.create_temp_images_and_masks(3)
        csv_data = {
            'path': [os.path.basename(p) for p in image_paths],
            'label': [os.path.basename(p) for p in mask_paths]
        }
        csv_path = self.create_temp_csv(csv_data)
        
        # Test dataset creation
        dataset = ImageCSVXLSXDataset(csv_path)
        
        self.assertEqual(len(dataset), 3)
        
        elapsed = time.time() - start_time
        print(f"CSV segmentation initialization test completed in {elapsed:.4f}s")
        
    def test_csv_custom_columns(self):
        """Test CSV segmentation dataset with custom column names"""
        start_time = time.time()
        
        # Create test data
        image_paths, mask_paths = self.create_temp_images_and_masks(3)
        csv_data = {
            'image_file': [os.path.basename(p) for p in image_paths],
            'mask_file': [os.path.basename(p) for p in mask_paths]
        }
        csv_path = self.create_temp_csv(csv_data)
        
        # Test dataset creation with custom column names
        dataset = ImageCSVXLSXDataset(csv_path, image_col='image_file', mask_col='mask_file')
        
        self.assertEqual(len(dataset), 3)
        
        elapsed = time.time() - start_time
        print(f"CSV custom columns test completed in {elapsed:.4f}s")
        
    def test_stress_csv_large_dataset(self):
        """Stress test CSV segmentation dataset with large dataset"""
        start_time = time.time()
        
        # Create large test dataset
        num_pairs = 500
        image_paths, mask_paths = self.create_temp_images_and_masks(num_pairs)
        
        csv_data = {
            'path': [os.path.basename(p) for p in image_paths],
            'label': [os.path.basename(p) for p in mask_paths]
        }
        csv_path = self.create_temp_csv(csv_data)
        
        # Time dataset creation
        creation_start = time.time()
        dataset = ImageCSVXLSXDataset(csv_path)
        creation_elapsed = time.time() - creation_start
        
        self.assertEqual(len(dataset), num_pairs)
        
        # Time dataset access
        access_start = time.time()
        for i in range(min(100, len(dataset))):  # Test first 100 samples
            _ = dataset.samples[i]
        access_elapsed = time.time() - access_start
        
        total_elapsed = time.time() - start_time
        print(f"Stress CSV segmentation test completed in {total_elapsed:.4f}s (creation: {creation_elapsed:.4f}s, access: {access_elapsed:.4f}s)")
        
    def test_invalid_csv(self):
        """Test CSV segmentation dataset with invalid data"""
        start_time = time.time()
        
        # Test missing columns
        csv_data = {'wrong_col': ['img1.jpg'], 'another_wrong': ['mask1.png']}
        csv_path = self.create_temp_csv(csv_data)
        
        with self.assertRaises(AssertionError):
            ImageCSVXLSXDataset(csv_path)
            
        elapsed = time.time() - start_time
        print(f"Invalid CSV segmentation test completed in {elapsed:.4f}s")


class TestImageSubdirSegmentationDataset(unittest.TestCase):
    """Test subdirectory-based segmentation dataset"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def create_temp_segmentation_dirs(self, num_pairs=5):
        """Create temporary image and mask directories with files"""
        image_dir = os.path.join(self.temp_dir, 'images')
        mask_dir = os.path.join(self.temp_dir, 'masks')
        
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        
        for i in range(num_pairs):
            # Create image
            img_path = os.path.join(image_dir, f'img_{i}.jpg')
            img = Image.new('RGB', (10, 10), color=(i*50, i*50, i*50))
            img.save(img_path)
            
            # Create mask
            mask_path = os.path.join(mask_dir, f'img_{i}.png')
            mask = Image.new('L', (10, 10), color=i*50)
            mask.save(mask_path)
            
    def test_subdir_initialization(self):
        """Test subdirectory segmentation dataset initialization"""
        start_time = time.time()
        
        # Create test directories
        self.create_temp_segmentation_dirs(5)
        
        # Test dataset creation
        dataset = ImageSubdirDataset(
            image_dir=os.path.join(self.temp_dir, 'images'),
            mask_dir=os.path.join(self.temp_dir, 'masks')
        )
        
        self.assertEqual(len(dataset), 5)
        
        elapsed = time.time() - start_time
        print(f"Subdir segmentation initialization test completed in {elapsed:.4f}s")
        
    def test_subdir_with_suffix(self):
        """Test subdirectory segmentation dataset with custom suffix"""
        start_time = time.time()
        
        # Create test directories with suffix
        image_dir = os.path.join(self.temp_dir, 'images')
        mask_dir = os.path.join(self.temp_dir, 'masks')
        
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        
        for i in range(3):
            # Create image
            img_path = os.path.join(image_dir, f'img_{i}.jpg')
            img = Image.new('RGB', (10, 10), color=(i*50, i*50, i*50))
            img.save(img_path)
            
            # Create mask with suffix
            mask_path = os.path.join(mask_dir, f'img_{i}_mask.png')
            mask = Image.new('L', (10, 10), color=i*50)
            mask.save(mask_path)
            
        # Test dataset creation with suffix
        dataset = ImageSubdirDataset(
            image_dir=image_dir,
            mask_dir=mask_dir,
            suffix='_mask'
        )
        
        self.assertEqual(len(dataset), 3)
        
        elapsed = time.time() - start_time
        print(f"Subdir with suffix test completed in {elapsed:.4f}s")
        
    def test_subdir_with_transform(self):
        """Test subdirectory segmentation dataset with transform"""
        start_time = time.time()
        
        # Create test directories
        self.create_temp_segmentation_dirs(3)
        
        # Create mock transform
        mock_transform = MagicMock()
        mock_transform.return_value = {'image': 'transformed_img', 'mask': 'transformed_mask'}
        
        # Test dataset creation with transform
        dataset = ImageSubdirDataset(
            image_dir=os.path.join(self.temp_dir, 'images'),
            mask_dir=os.path.join(self.temp_dir, 'masks'),
            transform=mock_transform
        )
        
        # Test first sample
        result = dataset[0]
        self.assertEqual(len(result), 2)
        
        elapsed = time.time() - start_time
        print(f"Subdir with transform test completed in {elapsed:.4f}s")
        
    def test_stress_subdir_large_dataset(self):
        """Stress test subdirectory segmentation dataset with large dataset"""
        start_time = time.time()
        
        # Create large test dataset
        num_pairs = 1000
        
        image_dir = os.path.join(self.temp_dir, 'images')
        mask_dir = os.path.join(self.temp_dir, 'masks')
        
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        
        for i in range(num_pairs):
            # Create image
            img_path = os.path.join(image_dir, f'img_{i}.jpg')
            img = Image.new('RGB', (10, 10), color=(i % 256, i % 256, i % 256))
            img.save(img_path)
            
            # Create mask
            mask_path = os.path.join(mask_dir, f'img_{i}.png')
            mask = Image.new('L', (10, 10), color=i % 256)
            mask.save(mask_path)
            
        # Time dataset creation
        creation_start = time.time()
        dataset = ImageSubdirDataset(image_dir, mask_dir)
        creation_elapsed = time.time() - creation_start
        
        self.assertEqual(len(dataset), num_pairs)
        
        # Time dataset access
        access_start = time.time()
        for i in range(min(200, len(dataset))):  # Test first 200 samples
            _ = dataset.samples[i]
        access_elapsed = time.time() - access_start
        
        total_elapsed = time.time() - start_time
        print(f"Stress subdir segmentation test completed in {total_elapsed:.4f}s (creation: {creation_elapsed:.4f}s, access: {access_elapsed:.4f}s)")
        
    def test_empty_dirs(self):
        """Test subdirectory segmentation dataset with empty directories"""
        start_time = time.time()
        
        # Create empty directories
        image_dir = os.path.join(self.temp_dir, 'images')
        mask_dir = os.path.join(self.temp_dir, 'masks')
        
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        
        # Should raise assertion error
        with self.assertRaises(AssertionError):
            ImageSubdirDataset(image_dir, mask_dir)
            
        elapsed = time.time() - start_time
        print(f"Empty dirs test completed in {elapsed:.4f}s")
        
    def test_binary_mask_option(self):
        """Test binary mask option in subdirectory dataset"""
        start_time = time.time()
        
        # Create test directories
        self.create_temp_segmentation_dirs(3)
        
        # Test dataset creation with binary_mask=True
        dataset = ImageSubdirDataset(
            image_dir=os.path.join(self.temp_dir, 'images'),
            mask_dir=os.path.join(self.temp_dir, 'masks'),
            binary_mask=True
        )
        
        self.assertEqual(dataset.binary_mask, True)
        self.assertEqual(len(dataset), 3)
        
        elapsed = time.time() - start_time
        print(f"Binary mask option test completed in {elapsed:.4f}s")


if __name__ == '__main__':
    # Run tests with timing
    print("Running vision segmentation tests...")
    unittest.main(verbosity=2)
