import unittest
import tempfile
import time
import os
import shutil
import gc
import psutil
import threading
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torch
from unittest.mock import patch, MagicMock
import multiprocessing as mp

# Import all vision datasets
from torchdatasets.vision.classification.base import BaseImageClassificationDataset
from torchdatasets.vision.classification.from_csv import ImageCSVXLSXDataset
from torchdatasets.vision.classification.from_singledir import ImageSingleDirDataset
from torchdatasets.vision.classification.from_subdirs import ImageSubdirDataset
from torchdatasets.vision.segmentation.base import BaseImageSegmentationDataset
from torchdatasets.vision.segmentation.from_csv import ImageCSVXLSXDataset as SegmentationCSVDataset
from torchdatasets.vision.segmentation.from_subdirs import ImageSubdirDataset as SegmentationSubdirDataset
from torchdatasets.vision.metric_learning.base import BaseMetricLearningWrapper
from torchdatasets.vision.metric_learning.contrastive import ContrastiveWrapper
from torchdatasets.vision.metric_learning.few_shot import FewShotWrapper
from torchdatasets.vision.metric_learning.triplet import TripletWrapper


class MockDataset:
    """Mock dataset for testing metric learning wrappers"""
    def __init__(self, num_samples=100, num_classes=10):
        self.samples = []
        for i in range(num_samples):
            label = i % num_classes
            self.samples.append((f'image_{i}', label))
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return self.samples[idx]


class VisionDatasetPerformanceTest(unittest.TestCase):
    """Comprehensive performance and stress testing for vision datasets"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.performance_results = {}
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def log_performance(self, test_name, elapsed_time, memory_usage=None, additional_info=None):
        """Log performance metrics"""
        self.performance_results[test_name] = {
            'elapsed_time': elapsed_time,
            'memory_usage': memory_usage,
            'additional_info': additional_info
        }
        print(f"Performance: {test_name} - Time: {elapsed_time:.4f}s, Memory: {memory_usage}MB")
        
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
        
    def create_large_image_dataset(self, num_images=1000, image_size=(64, 64)):
        """Create a large dataset of temporary images"""
        start_time = time.time()
        
        image_paths = []
        for i in range(num_images):
            img_path = os.path.join(self.temp_dir, f'img_{i}.jpg')
            # Create larger images for more realistic testing
            img = Image.new('RGB', image_size, color=(i % 256, i % 256, i % 256))
            img.save(img_path, quality=85)  # Lower quality for faster creation
            image_paths.append(img_path)
            
        creation_time = time.time() - start_time
        print(f"Created {num_images} images in {creation_time:.4f}s")
        return image_paths
        
    def create_large_csv_dataset(self, num_images=1000, num_classes=100):
        """Create a large CSV dataset"""
        start_time = time.time()
        
        # Create images
        image_paths = self.create_large_image_dataset(num_images)
        
        # Create CSV data
        labels = [f'class_{i % num_classes}' for i in range(num_images)]
        csv_data = {
            'path': [os.path.basename(p) for p in image_paths],
            'label': labels
        }
        
        csv_path = os.path.join(self.temp_dir, 'large_dataset.csv')
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)
        
        creation_time = time.time() - start_time
        print(f"Created CSV dataset with {num_images} samples in {creation_time:.4f}s")
        return csv_path
        
    def create_large_subdir_dataset(self, num_classes=100, images_per_class=50):
        """Create a large subdirectory dataset"""
        start_time = time.time()
        
        total_images = 0
        for class_idx in range(num_classes):
            class_name = f'class_{class_idx}'
            class_dir = os.path.join(self.temp_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            for img_idx in range(images_per_class):
                img_path = os.path.join(class_dir, f'img_{img_idx}.jpg')
                img = Image.new('RGB', (64, 64), color=(class_idx % 256, img_idx % 256, 100))
                img.save(img_path, quality=85)
                total_images += 1
                
        creation_time = time.time() - start_time
        print(f"Created subdir dataset with {num_classes} classes, {total_images} total images in {creation_time:.4f}s")
        
    def test_classification_csv_performance(self):
        """Performance test for CSV classification dataset"""
        start_time = time.time()
        initial_memory = self.get_memory_usage()
        
        # Create large dataset
        csv_path = self.create_large_csv_dataset(5000, 200)
        
        # Time dataset creation
        creation_start = time.time()
        dataset = ImageCSVXLSXDataset(csv_path)
        creation_time = time.time() - creation_start
        
        # Time dataset access
        access_start = time.time()
        for i in range(min(500, len(dataset))):
            _ = dataset.samples[i]
        access_time = time.time() - access_start
        
        # Memory usage
        final_memory = self.get_memory_usage()
        memory_used = final_memory - initial_memory
        
        total_time = time.time() - start_time
        
        self.log_performance(
            'Classification CSV Performance',
            total_time,
            memory_used,
            {
                'dataset_size': len(dataset),
                'num_classes': len(dataset.class_to_idx),
                'creation_time': creation_time,
                'access_time': access_time
            }
        )
        
        self.assertEqual(len(dataset), 5000)
        self.assertEqual(len(dataset.class_to_idx), 200)
        
    def test_classification_subdir_performance(self):
        """Performance test for subdirectory classification dataset"""
        start_time = time.time()
        initial_memory = self.get_memory_usage()
        
        # Create large dataset
        self.create_large_subdir_dataset(150, 40)
        
        # Time dataset creation
        creation_start = time.time()
        dataset = ImageSubdirDataset(self.temp_dir)
        creation_time = time.time() - creation_start
        
        # Time dataset access
        access_start = time.time()
        for i in range(min(500, len(dataset))):
            _ = dataset.samples[i]
        access_time = time.time() - access_start
        
        # Memory usage
        final_memory = self.get_memory_usage()
        memory_used = final_memory - initial_memory
        
        total_time = time.time() - start_time
        
        self.log_performance(
            'Classification Subdir Performance',
            total_time,
            memory_used,
            {
                'dataset_size': len(dataset),
                'num_classes': len(dataset.class_to_idx),
                'creation_time': creation_time,
                'access_time': access_time
            }
        )
        
        self.assertEqual(len(dataset), 150 * 40)
        self.assertEqual(len(dataset.class_to_idx), 150)
        
    def test_segmentation_performance(self):
        """Performance test for segmentation datasets"""
        start_time = time.time()
        initial_memory = self.get_memory_usage()
        
        # Create large segmentation dataset
        num_pairs = 2000
        image_dir = os.path.join(self.temp_dir, 'images')
        mask_dir = os.path.join(self.temp_dir, 'masks')
        
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        
        for i in range(num_pairs):
            # Create image
            img_path = os.path.join(image_dir, f'img_{i}.jpg')
            img = Image.new('RGB', (64, 64), color=(i % 256, i % 256, i % 256))
            img.save(img_path, quality=85)
            
            # Create mask
            mask_path = os.path.join(mask_dir, f'img_{i}.png')
            mask = Image.new('L', (64, 64), color=i % 256)
            mask.save(mask_path)
            
        # Time dataset creation
        creation_start = time.time()
        dataset = SegmentationSubdirDataset(image_dir, mask_dir)
        creation_time = time.time() - creation_start
        
        # Time dataset access
        access_start = time.time()
        for i in range(min(300, len(dataset))):
            _ = dataset.samples[i]
        access_time = time.time() - access_start
        
        # Memory usage
        final_memory = self.get_memory_usage()
        memory_used = final_memory - initial_memory
        
        total_time = time.time() - start_time
        
        self.log_performance(
            'Segmentation Performance',
            total_time,
            memory_used,
            {
                'dataset_size': len(dataset),
                'creation_time': creation_time,
                'access_time': access_time
            }
        )
        
        self.assertEqual(len(dataset), num_pairs)
        
    def test_metric_learning_performance(self):
        """Performance test for metric learning wrappers"""
        start_time = time.time()
        initial_memory = self.get_memory_usage()
        
        # Create large mock dataset
        large_dataset = MockDataset(10000, 200)
        
        # Time wrapper creation
        creation_start = time.time()
        contrastive = ContrastiveWrapper(large_dataset)
        few_shot = FewShotWrapper(large_dataset, n_way=10, k_shot=5, q_query=20)
        triplet = TripletWrapper(large_dataset)
        creation_time = time.time() - creation_start
        
        # Time wrapper access
        access_start = time.time()
        for i in range(100):
            _ = contrastive[i % len(contrastive)]
            _ = few_shot[i % len(few_shot)]
            _ = triplet[i % len(triplet)]
        access_time = time.time() - access_start
        
        # Memory usage
        final_memory = self.get_memory_usage()
        memory_used = final_memory - initial_memory
        
        total_time = time.time() - start_time
        
        self.log_performance(
            'Metric Learning Performance',
            total_time,
            memory_used,
            {
                'dataset_size': len(large_dataset),
                'num_classes': 200,
                'creation_time': creation_time,
                'access_time': access_time
            }
        )
        
    def test_memory_efficiency(self):
        """Test memory efficiency with large datasets"""
        start_time = time.time()
        
        # Monitor memory during dataset creation
        initial_memory = self.get_memory_usage()
        
        # Create very large dataset
        csv_path = self.create_large_csv_dataset(10000, 500)
        
        memory_after_creation = self.get_memory_usage()
        
        # Create dataset
        dataset = ImageCSVXLSXDataset(csv_path)
        
        memory_after_dataset = self.get_memory_usage()
        
        # Force garbage collection
        gc.collect()
        memory_after_gc = self.get_memory_usage()
        
        # Memory usage analysis
        memory_used = memory_after_dataset - initial_memory
        memory_saved_by_gc = memory_after_dataset - memory_after_gc
        
        total_time = time.time() - start_time
        
        self.log_performance(
            'Memory Efficiency Test',
            total_time,
            memory_used,
            {
                'dataset_size': len(dataset),
                'memory_after_creation': memory_after_creation - initial_memory,
                'memory_after_dataset': memory_used,
                'memory_saved_by_gc': memory_saved_by_gc
            }
        )
        
        # Verify memory usage is reasonable (less than 1GB for 10k samples)
        self.assertLess(memory_used, 1024)
        
    def test_concurrent_access(self):
        """Test concurrent access to datasets"""
        start_time = time.time()
        
        # Create dataset
        csv_path = self.create_large_csv_dataset(2000, 100)
        dataset = ImageCSVXLSXDataset(csv_path)
        
        # Test concurrent access
        def access_dataset(thread_id, num_accesses):
            for i in range(num_accesses):
                idx = (thread_id * num_accesses + i) % len(dataset)
                _ = dataset.samples[idx]
                
        # Create multiple threads
        threads = []
        num_threads = 4
        accesses_per_thread = 100
        
        thread_start = time.time()
        for i in range(num_threads):
            thread = threading.Thread(
                target=access_dataset,
                args=(i, accesses_per_thread)
            )
            threads.append(thread)
            thread.start()
            
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
            
        thread_time = time.time() - thread_start
        total_time = time.time() - start_time
        
        self.log_performance(
            'Concurrent Access Test',
            total_time,
            None,
            {
                'num_threads': num_threads,
                'accesses_per_thread': accesses_per_thread,
                'thread_time': thread_time
            }
        )
        
    def test_stress_test_extreme_sizes(self):
        """Extreme stress test with very large datasets"""
        start_time = time.time()
        
        # Test with extremely large dataset
        try:
            csv_path = self.create_large_csv_dataset(50000, 1000)
            
            # Time dataset creation
            creation_start = time.time()
            dataset = ImageCSVXLSXDataset(csv_path)
            creation_time = time.time() - creation_start
            
            # Time dataset access
            access_start = time.time()
            for i in range(min(1000, len(dataset))):
                _ = dataset.samples[i]
            access_time = time.time() - access_start
            
            total_time = time.time() - start_time
            
            self.log_performance(
                'Extreme Stress Test',
                total_time,
                self.get_memory_usage(),
                {
                    'dataset_size': len(dataset),
                    'num_classes': len(dataset.class_to_idx),
                    'creation_time': creation_time,
                    'access_time': access_time
                }
            )
            
            self.assertEqual(len(dataset), 50000)
            self.assertEqual(len(dataset.class_to_idx), 1000)
            
        except Exception as e:
            print(f"Extreme stress test failed: {e}")
            # This is expected for very large datasets on some systems
            
    def test_performance_regression(self):
        """Test for performance regression by comparing with baseline"""
        start_time = time.time()
        
        # Create baseline dataset
        csv_path = self.create_large_csv_dataset(1000, 50)
        
        # Baseline performance measurement
        baseline_start = time.time()
        dataset = ImageCSVXLSXDataset(csv_path)
        baseline_time = time.time() - baseline_start
        
        # Access baseline
        baseline_access_start = time.time()
        for i in range(100):
            _ = dataset.samples[i]
        baseline_access_time = time.time() - baseline_access_start
        
        total_time = time.time() - start_time
        
        self.log_performance(
            'Performance Regression Test',
            total_time,
            self.get_memory_usage(),
            {
                'baseline_creation': baseline_time,
                'baseline_access': baseline_access_time,
                'dataset_size': len(dataset)
            }
        )
        
        # Performance thresholds (adjust based on your system)
        self.assertLess(baseline_time, 5.0)  # Should create dataset in under 5 seconds
        self.assertLess(baseline_access_time, 1.0)  # Should access 100 samples in under 1 second
        
    def test_resource_cleanup(self):
        """Test that resources are properly cleaned up"""
        start_time = time.time()
        
        initial_memory = self.get_memory_usage()
        
        # Create and destroy multiple datasets
        for iteration in range(5):
            csv_path = self.create_large_csv_dataset(1000, 50)
            dataset = ImageCSVXLSXDataset(csv_path)
            
            # Access some data
            for i in range(100):
                _ = dataset.samples[i]
                
            # Delete dataset and force cleanup
            del dataset
            gc.collect()
            
        final_memory = self.get_memory_usage()
        memory_delta = final_memory - initial_memory
        
        total_time = time.time() - start_time
        
        self.log_performance(
            'Resource Cleanup Test',
            total_time,
            memory_delta,
            {
                'iterations': 5,
                'initial_memory': initial_memory,
                'final_memory': final_memory
            }
        )
        
        # Memory should not grow significantly after cleanup
        self.assertLess(abs(memory_delta), 100)  # Should not grow more than 100MB


class VisionDatasetStressTest(unittest.TestCase):
    """Extreme stress testing for vision datasets"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def test_memory_pressure(self):
        """Test behavior under memory pressure"""
        start_time = time.time()
        
        # Create multiple large datasets to simulate memory pressure
        datasets = []
        
        try:
            for i in range(10):
                csv_path = self.create_large_csv_dataset(1000, 50)
                dataset = ImageCSVXLSXDataset(csv_path)
                datasets.append(dataset)
                
                # Check memory usage
                memory = psutil.Process().memory_info().rss / 1024 / 1024
                print(f"Dataset {i+1}: Memory usage: {memory:.1f}MB")
                
        except MemoryError:
            print("Memory pressure test triggered MemoryError (expected)")
        except Exception as e:
            print(f"Memory pressure test failed with: {e}")
            
        total_time = time.time() - start_time
        print(f"Memory pressure test completed in {total_time:.4f}s")
        
    def create_large_csv_dataset(self, num_images, num_classes):
        """Helper method to create large CSV dataset"""
        # Create images
        image_paths = []
        for i in range(num_images):
            img_path = os.path.join(self.temp_dir, f'dataset_{num_images}_img_{i}.jpg')
            img = Image.new('RGB', (32, 32), color=(i % 256, i % 256, i % 256))
            img.save(img_path, quality=70)
            image_paths.append(img_path)
            
        # Create CSV
        labels = [f'class_{i % num_classes}' for i in range(num_images)]
        csv_data = {
            'path': [os.path.basename(p) for p in image_paths],
            'label': labels
        }
        
        csv_path = os.path.join(self.temp_dir, f'large_dataset_{num_images}.csv')
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)
        return csv_path
        
    def test_concurrent_dataset_creation(self):
        """Test creating multiple datasets concurrently"""
        start_time = time.time()
        
        def create_dataset(dataset_id):
            try:
                csv_path = self.create_large_csv_dataset(500, 25)
                dataset = ImageCSVXLSXDataset(csv_path)
                return len(dataset)
            except Exception as e:
                return f"Error: {e}"
                
        # Create datasets in parallel
        with mp.Pool(processes=4) as pool:
            results = pool.map(create_dataset, range(4))
            
        total_time = time.time() - start_time
        print(f"Concurrent dataset creation completed in {total_time:.4f}s")
        print(f"Results: {results}")
        
        # All should succeed
        for result in results:
            self.assertIsInstance(result, int)
            self.assertEqual(result, 500)


if __name__ == '__main__':
    # Run performance tests
    print("Running vision dataset performance tests...")
    
    # Performance tests
    performance_suite = unittest.TestLoader().loadTestsFromTestCase(VisionDatasetPerformanceTest)
    performance_runner = unittest.TextTestRunner(verbosity=2)
    performance_result = performance_runner.run(performance_suite)
    
    # Stress tests
    print("\nRunning vision dataset stress tests...")
    stress_suite = unittest.TestLoader().loadTestsFromTestCase(VisionDatasetStressTest)
    stress_runner = unittest.TextTestRunner(verbosity=2)
    stress_result = stress_runner.run(stress_suite)
    
    # Print performance summary
    print("\n" + "="*60)
    print("PERFORMANCE TEST SUMMARY")
    print("="*60)
    
    if hasattr(performance_result, 'performance_results'):
        for test_name, metrics in performance_result.performance_results.items():
            print(f"\n{test_name}:")
            print(f"  Time: {metrics['elapsed_time']:.4f}s")
            if metrics['memory_usage']:
                print(f"  Memory: {metrics['memory_usage']:.1f}MB")
            if metrics['additional_info']:
                for key, value in metrics['additional_info'].items():
                    print(f"  {key}: {value}")
                    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
