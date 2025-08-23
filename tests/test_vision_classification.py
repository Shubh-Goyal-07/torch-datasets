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
from torchdatasets.vision.classification.base import BaseImageClassificationDataset
from torchdatasets.vision.classification.from_csv import ImageCSVXLSXDataset
from torchdatasets.vision.classification.from_singledir import ImageSingleDirDataset
from torchdatasets.vision.classification.from_subdirs import ImageSubdirDataset


class TestBaseImageClassificationDataset(unittest.TestCase):
    """Test the base classification dataset class"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.dataset = BaseImageClassificationDataset()
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def test_initialization(self):
        """Test dataset initialization"""
        start_time = time.time()
        
        self.assertEqual(len(self.dataset), 0)
        self.assertEqual(self.dataset.transform, None)
        self.assertEqual(self.dataset.return_path, False)
        self.assertEqual(self.dataset.extensions, {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'})
        
        elapsed = time.time() - start_time
        print(f"Initialization test completed in {elapsed:.4f}s")
        
    def test_custom_extensions(self):
        """Test custom extensions initialization"""
        start_time = time.time()
        
        custom_extensions = ['.tif', '.webp']
        dataset = BaseImageClassificationDataset(extensions=custom_extensions)
        self.assertEqual(dataset.extensions, {'.tif', '.webp'})
        
        elapsed = time.time() - start_time
        print(f"Custom extensions test completed in {elapsed:.4f}s")
        
    def test_finalize(self):
        """Test finalize method"""
        start_time = time.time()
        
        # Create mock samples
        self.dataset.samples = [
            ('/path/to/img1.jpg', 0),
            ('/path/to/img2.jpg', 1),
            ('/path/to/img3.jpg', 0),
            ('/path/to/img4.jpg', 2)
        ]
        self.dataset.class_to_idx = {'cat': 0, 'dog': 1, 'bird': 2}
        
        self.dataset.finalize()
        
        self.assertEqual(self.dataset.idx_to_class, {0: 'cat', 1: 'dog', 2: 'bird'})
        self.assertEqual(self.dataset.class_count, {0: 2, 1: 1, 2: 1})
        
        elapsed = time.time() - start_time
        print(f"Finalize test completed in {elapsed:.4f}s")
        
    def test_stress_finalize(self):
        """Stress test finalize method with large dataset"""
        start_time = time.time()
        
        # Create large mock dataset
        large_samples = []
        large_class_to_idx = {}
        
        for i in range(10000):
            large_samples.append((f'/path/to/img{i}.jpg', i % 100))
            if i % 100 == 0:
                large_class_to_idx[f'class_{i//100}'] = i % 100
                
        self.dataset.samples = large_samples
        self.dataset.class_to_idx = large_class_to_idx
        
        # Time the finalize operation
        finalize_start = time.time()
        self.dataset.finalize()
        finalize_elapsed = time.time() - finalize_start
        
        self.assertEqual(len(self.dataset.idx_to_class), 100)
        self.assertEqual(len(self.dataset.class_count), 100)
        
        total_elapsed = time.time() - start_time
        print(f"Stress finalize test completed in {total_elapsed:.4f}s (finalize: {finalize_elapsed:.4f}s)")
        
    def test_getitem_mock(self):
        """Test __getitem__ with mocked image loading"""
        start_time = time.time()
        
        # Create mock samples
        self.dataset.samples = [('/path/to/img1.jpg', 0)]
        self.dataset.class_to_idx = {'cat': 0}
        self.dataset.finalize()
        
        # Mock PIL Image.open
        with patch('PIL.Image.open') as mock_open:
            mock_img = MagicMock()
            mock_img.convert.return_value = mock_img
            mock_open.return_value = mock_img
            
            result = self.dataset[0]
            self.assertEqual(len(result), 2)
            self.assertEqual(result[1], 0)
            
        elapsed = time.time() - start_time
        print(f"Getitem mock test completed in {elapsed:.4f}s")
        
    def test_getitem_with_transform(self):
        """Test __getitem__ with transform"""
        start_time = time.time()
        
        # Create mock transform
        mock_transform = MagicMock()
        mock_transform.return_value = 'transformed_image'
        
        dataset = BaseImageClassificationDataset(transform=mock_transform)
        dataset.samples = [('/path/to/img1.jpg', 0)]
        dataset.class_to_idx = {'cat': 0}
        dataset.finalize()
        
        with patch('PIL.Image.open') as mock_open:
            mock_img = MagicMock()
            mock_img.convert.return_value = mock_img
            mock_open.return_value = mock_img
            
            result = dataset[0]
            self.assertEqual(result[0], 'transformed_image')
            mock_transform.assert_called_once()
            
        elapsed = time.time() - start_time
        print(f"Getitem with transform test completed in {elapsed:.4f}s")
        
    def test_getitem_with_return_path(self):
        """Test __getitem__ with return_path=True"""
        start_time = time.time()
        
        dataset = BaseImageClassificationDataset(return_path=True)
        dataset.samples = [('/path/to/img1.jpg', 0)]
        dataset.class_to_idx = {'cat': 0}
        dataset.finalize()
        
        with patch('PIL.Image.open') as mock_open:
            mock_img = MagicMock()
            mock_img.convert.return_value = mock_img
            mock_open.return_value = mock_img
            
            result = dataset[0]
            self.assertEqual(len(result), 3)
            self.assertEqual(result[2], '/path/to/img1.jpg')
            
        elapsed = time.time() - start_time
        print(f"Getitem with return_path test completed in {elapsed:.4f}s")


class TestImageCSVXLSXDataset(unittest.TestCase):
    """Test CSV/XLSX based classification dataset"""
    
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
        
    def create_temp_images(self, num_images=5):
        """Create temporary image files"""
        image_paths = []
        for i in range(num_images):
            img_path = os.path.join(self.temp_dir, f'img_{i}.jpg')
            # Create a small test image
            img = Image.new('RGB', (10, 10), color=(i*50, i*50, i*50))
            img.save(img_path)
            image_paths.append(img_path)
        return image_paths
        
    def test_csv_initialization(self):
        """Test CSV dataset initialization"""
        start_time = time.time()
        
        # Create test data
        image_paths = self.create_temp_images(3)
        csv_data = {
            'path': [os.path.basename(p) for p in image_paths],
            'label': ['cat', 'dog', 'cat']
        }
        csv_path = self.create_temp_csv(csv_data)
        
        # Test dataset creation
        dataset = ImageCSVXLSXDataset(csv_path)
        
        self.assertEqual(len(dataset), 3)
        self.assertEqual(len(dataset.class_to_idx), 2)
        self.assertEqual(dataset.class_to_idx['cat'], 0)
        self.assertEqual(dataset.class_to_idx['dog'], 1)
        
        elapsed = time.time() - start_time
        print(f"CSV initialization test completed in {elapsed:.4f}s")
        
    def test_csv_multi_label(self):
        """Test CSV dataset with multi-label support"""
        start_time = time.time()
        
        # Create test data
        image_paths = self.create_temp_images(3)
        csv_data = {
            'path': [os.path.basename(p) for p in image_paths],
            'label': ['cat,dog', 'dog', 'cat,bird']
        }
        csv_path = self.create_temp_csv(csv_data)
        
        # Test dataset creation with separator
        dataset = ImageCSVXLSXDataset(csv_path, sep=',')
        
        self.assertEqual(len(dataset), 3)
        self.assertEqual(len(dataset.class_to_idx), 3)
        
        # Check first sample (should have two labels)
        first_sample = dataset.samples[0]
        self.assertEqual(len(first_sample[1]), 2)  # Two labels
        
        elapsed = time.time() - start_time
        print(f"CSV multi-label test completed in {elapsed:.4f}s")
        
    def test_stress_csv_large_dataset(self):
        """Stress test CSV dataset with large dataset"""
        start_time = time.time()
        
        # Create large test dataset
        num_images = 1000
        image_paths = self.create_temp_images(num_images)
        
        # Create varied labels
        labels = ['cat', 'dog', 'bird', 'fish', 'horse'] * (num_images // 5)
        labels = labels[:num_images]
        
        csv_data = {
            'path': [os.path.basename(p) for p in image_paths],
            'label': labels
        }
        csv_path = self.create_temp_csv(csv_data)
        
        # Time dataset creation
        creation_start = time.time()
        dataset = ImageCSVXLSXDataset(csv_path)
        creation_elapsed = time.time() - creation_start
        
        self.assertEqual(len(dataset), num_images)
        self.assertEqual(len(dataset.class_to_idx), 5)
        
        # Time dataset access
        access_start = time.time()
        for i in range(min(100, len(dataset))):  # Test first 100 samples
            _ = dataset[i]
        access_elapsed = time.time() - access_start
        
        total_elapsed = time.time() - start_time
        print(f"Stress CSV test completed in {total_elapsed:.4f}s (creation: {creation_elapsed:.4f}s, access: {access_elapsed:.4f}s)")
        
    def test_invalid_csv(self):
        """Test CSV dataset with invalid data"""
        start_time = time.time()
        
        # Test missing columns
        csv_data = {'wrong_col': ['img1.jpg'], 'another_wrong': ['cat']}
        csv_path = self.create_temp_csv(csv_data)
        
        with self.assertRaises(AssertionError):
            ImageCSVXLSXDataset(csv_path)
            
        elapsed = time.time() - start_time
        print(f"Invalid CSV test completed in {elapsed:.4f}s")


class TestImageSingleDirDataset(unittest.TestCase):
    """Test single directory classification dataset"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def create_temp_images_with_labels(self, num_images=5):
        """Create temporary image files with label prefixes"""
        image_paths = []
        labels = ['cat', 'dog', 'bird', 'cat', 'dog']
        
        for i in range(num_images):
            img_path = os.path.join(self.temp_dir, f'{labels[i]}_img_{i}.jpg')
            img = Image.new('RGB', (10, 10), color=(i*50, i*50, i*50))
            img.save(img_path)
            image_paths.append(img_path)
        return image_paths, labels
        
    def test_single_dir_initialization(self):
        """Test single directory dataset initialization"""
        start_time = time.time()
        
        # Create test images
        image_paths, labels = self.create_temp_images_with_labels(5)
        
        # Test dataset creation
        dataset = ImageSingleDirDataset(self.temp_dir)
        
        self.assertEqual(len(dataset), 5)
        self.assertEqual(len(dataset.class_to_idx), 3)  # cat, dog, bird
        self.assertEqual(dataset.class_to_idx['cat'], 0)
        self.assertEqual(dataset.class_to_idx['dog'], 1)
        self.assertEqual(dataset.class_to_idx['bird'], 2)
        
        elapsed = time.time() - start_time
        print(f"Single dir initialization test completed in {elapsed:.4f}s")
        
    def test_custom_delimiter(self):
        """Test single directory dataset with custom delimiter"""
        start_time = time.time()
        
        # Create test images with custom delimiter
        for i in range(3):
            img_path = os.path.join(self.temp_dir, f'class{i}.img{i}.jpg')
            img = Image.new('RGB', (10, 10), color=(i*50, i*50, i*50))
            img.save(img_path)
            
        # Test dataset creation with custom delimiter
        dataset = ImageSingleDirDataset(self.temp_dir, delimiter='.')
        
        self.assertEqual(len(dataset), 3)
        self.assertEqual(len(dataset.class_to_idx), 3)
        
        elapsed = time.time() - start_time
        print(f"Custom delimiter test completed in {elapsed:.4f}s")
        
    def test_custom_label_map(self):
        """Test single directory dataset with custom label mapping"""
        start_time = time.time()
        
        # Create test images
        image_paths, labels = self.create_temp_images_with_labels(3)
        
        # Create custom label map
        label_map = {'cat': 10, 'dog': 20, 'bird': 30}
        
        # Test dataset creation with custom label map
        dataset = ImageSingleDirDataset(self.temp_dir, label_map=label_map)
        
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.class_to_idx, label_map)
        
        elapsed = time.time() - start_time
        print(f"Custom label map test completed in {elapsed:.4f}s")
        
    def test_stress_single_dir_large_dataset(self):
        """Stress test single directory dataset with large dataset"""
        start_time = time.time()
        
        # Create large test dataset
        num_images = 500
        labels = ['class_' + str(i % 50) for i in range(num_images)]
        
        for i in range(num_images):
            img_path = os.path.join(self.temp_dir, f'{labels[i]}_img_{i}.jpg')
            img = Image.new('RGB', (10, 10), color=(i % 256, i % 256, i % 256))
            img.save(img_path)
            
        # Time dataset creation
        creation_start = time.time()
        dataset = ImageSingleDirDataset(self.temp_dir)
        creation_elapsed = time.time() - creation_start
        
        self.assertEqual(len(dataset), num_images)
        self.assertEqual(len(dataset.class_to_idx), 50)
        
        # Time dataset access
        access_start = time.time()
        for i in range(min(100, len(dataset))):  # Test first 100 samples
            _ = dataset[i]
        access_elapsed = time.time() - access_start
        
        total_elapsed = time.time() - start_time
        print(f"Stress single dir test completed in {total_elapsed:.4f}s (creation: {creation_elapsed:.4f}s, access: {access_elapsed:.4f}s)")


class TestImageSubdirDataset(unittest.TestCase):
    """Test subdirectory-based classification dataset"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def create_temp_subdirs(self, num_classes=3, images_per_class=5):
        """Create temporary subdirectories with images"""
        for class_idx in range(num_classes):
            class_name = f'class_{class_idx}'
            class_dir = os.path.join(self.temp_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            for img_idx in range(images_per_class):
                img_path = os.path.join(class_dir, f'img_{img_idx}.jpg')
                img = Image.new('RGB', (10, 10), color=(class_idx*50, img_idx*50, 100))
                img.save(img_path)
                
    def test_subdir_initialization(self):
        """Test subdirectory dataset initialization"""
        start_time = time.time()
        
        # Create test subdirectories
        self.create_temp_subdirs(3, 4)
        
        # Test dataset creation
        dataset = ImageSubdirDataset(self.temp_dir)
        
        self.assertEqual(len(dataset), 12)  # 3 classes * 4 images
        self.assertEqual(len(dataset.class_to_idx), 3)
        self.assertEqual(dataset.class_to_idx['class_0'], 0)
        self.assertEqual(dataset.class_to_idx['class_1'], 1)
        self.assertEqual(dataset.class_to_idx['class_2'], 2)
        
        elapsed = time.time() - start_time
        print(f"Subdir initialization test completed in {elapsed:.4f}s")
        
    def test_subdir_with_transform(self):
        """Test subdirectory dataset with transform"""
        start_time = time.time()
        
        # Create test subdirectories
        self.create_temp_subdirs(2, 3)
        
        # Create mock transform
        mock_transform = MagicMock()
        mock_transform.return_value = 'transformed_image'
        
        # Test dataset creation with transform
        dataset = ImageSubdirDataset(self.temp_dir, transform=mock_transform)
        
        # Test first sample
        result = dataset[0]
        self.assertEqual(result[0], 'transformed_image')
        mock_transform.assert_called()
        
        elapsed = time.time() - start_time
        print(f"Subdir with transform test completed in {elapsed:.4f}s")
        
    def test_stress_subdir_large_dataset(self):
        """Stress test subdirectory dataset with large dataset"""
        start_time = time.time()
        
        # Create large test dataset
        num_classes = 100
        images_per_class = 20
        
        self.create_temp_subdirs(num_classes, images_per_class)
        
        # Time dataset creation
        creation_start = time.time()
        dataset = ImageSubdirDataset(self.temp_dir)
        creation_elapsed = time.time() - creation_start
        
        self.assertEqual(len(dataset), num_classes * images_per_class)
        self.assertEqual(len(dataset.class_to_idx), num_classes)
        
        # Time dataset access
        access_start = time.time()
        for i in range(min(200, len(dataset))):  # Test first 200 samples
            _ = dataset[i]
        access_elapsed = time.time() - access_start
        
        total_elapsed = time.time() - start_time
        print(f"Stress subdir test completed in {total_elapsed:.4f}s (creation: {creation_elapsed:.4f}s, access: {access_elapsed:.4f}s)")
        
    def test_empty_subdirs(self):
        """Test subdirectory dataset with empty subdirectories"""
        start_time = time.time()
        
        # Create empty subdirectories
        for i in range(3):
            class_dir = os.path.join(self.temp_dir, f'class_{i}')
            os.makedirs(class_dir, exist_ok=True)
            
        # Should raise assertion error
        with self.assertRaises(AssertionError):
            ImageSubdirDataset(self.temp_dir)
            
        elapsed = time.time() - start_time
        print(f"Empty subdirs test completed in {elapsed:.4f}s")


if __name__ == '__main__':
    # Run tests with timing
    print("Running vision classification tests...")
    unittest.main(verbosity=2)
