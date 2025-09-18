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
from torchdatasets.vision.metric_learning.base import BaseMetricLearningWrapper
from torchdatasets.vision.metric_learning.contrastive import ContrastiveWrapper, make_contrastive
from torchdatasets.vision.metric_learning.few_shot import FewShotWrapper, make_few_shot
from torchdatasets.vision.metric_learning.triplet import TripletWrapper, make_triplet


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


class TestBaseMetricLearningWrapper(unittest.TestCase):
    """Test the base metric learning wrapper class"""
    
    def setUp(self):
        self.mock_dataset = MockDataset(100, 10)
        self.wrapper = BaseMetricLearningWrapper(self.mock_dataset)
        
    def test_initialization(self):
        """Test wrapper initialization"""
        start_time = time.time()
        
        self.assertEqual(len(self.wrapper), 100)
        self.assertEqual(len(self.wrapper.labels), 10)
        self.assertEqual(len(self.wrapper.class_to_indices), 10)
        
        # Check that class_to_indices is properly populated
        for label in range(10):
            self.assertIn(label, self.wrapper.class_to_indices)
            self.assertGreater(len(self.wrapper.class_to_indices[label]), 0)
            
        elapsed = time.time() - start_time
        print(f"Base wrapper initialization test completed in {elapsed:.4f}s")
        
    def test_stress_large_dataset(self):
        """Stress test with large dataset"""
        start_time = time.time()
        
        # Create large mock dataset
        large_dataset = MockDataset(10000, 100)
        
        # Time wrapper creation
        creation_start = time.time()
        large_wrapper = BaseMetricLearningWrapper(large_dataset)
        creation_elapsed = time.time() - creation_start
        
        self.assertEqual(len(large_wrapper), 10000)
        self.assertEqual(len(large_wrapper.labels), 100)
        self.assertEqual(len(large_wrapper.class_to_indices), 100)
        
        # Verify class distribution
        total_samples = sum(len(indices) for indices in large_wrapper.class_to_indices.values())
        self.assertEqual(total_samples, 10000)
        
        total_elapsed = time.time() - start_time
        print(f"Stress base wrapper test completed in {total_elapsed:.4f}s (creation: {creation_elapsed:.4f}s)")
        
    def test_class_distribution(self):
        """Test class distribution in wrapper"""
        start_time = time.time()
        
        # Create dataset with known distribution
        dataset = MockDataset(1000, 5)  # 5 classes, 200 samples each
        wrapper = BaseMetricLearningWrapper(dataset)
        
        # Check that each class has exactly 200 samples
        for label in range(5):
            self.assertEqual(len(wrapper.class_to_indices[label]), 200)
            
        elapsed = time.time() - start_time
        print(f"Class distribution test completed in {elapsed:.4f}s")


class TestContrastiveWrapper(unittest.TestCase):
    """Test contrastive learning wrapper"""
    
    def setUp(self):
        self.mock_dataset = MockDataset(100, 10)
        self.wrapper = ContrastiveWrapper(self.mock_dataset)
        
    def test_initialization(self):
        """Test contrastive wrapper initialization"""
        start_time = time.time()
        
        self.assertEqual(len(self.wrapper), 100)
        self.assertEqual(self.wrapper.same_class_prob, 0.5)
        self.assertEqual(self.wrapper.transform_pair, None)
        
        elapsed = time.time() - start_time
        print(f"Contrastive initialization test completed in {elapsed:.4f}s")
        
    def test_getitem_same_class(self):
        """Test __getitem__ with same class probability"""
        start_time = time.time()
        
        # Force same class by setting probability to 1.0
        wrapper = ContrastiveWrapper(self.mock_dataset, same_class_prob=1.0)
        
        # Test multiple samples
        for _ in range(10):
            result = wrapper[0]
            self.assertEqual(len(result), 2)
            self.assertEqual(result[1], 1)  # Should always be positive pair
            
        elapsed = time.time() - start_time
        print(f"Same class getitem test completed in {elapsed:.4f}s")
        
    def test_getitem_different_class(self):
        """Test __getitem__ with different class probability"""
        start_time = time.time()
        
        # Force different class by setting probability to 0.0
        wrapper = ContrastiveWrapper(self.mock_dataset, same_class_prob=0.0)
        
        # Test multiple samples
        for _ in range(10):
            result = wrapper[0]
            self.assertEqual(len(result), 2)
            self.assertEqual(result[1], 0)  # Should always be negative pair
            
        elapsed = time.time() - start_time
        print(f"Different class getitem test completed in {elapsed:.4f}s")
        
    def test_getitem_with_transform(self):
        """Test __getitem__ with transform_pair"""
        start_time = time.time()
        
        # Create mock transform
        mock_transform = MagicMock()
        mock_transform.return_value = 'transformed_image'
        
        wrapper = ContrastiveWrapper(self.mock_dataset, transform_pair=mock_transform)
        
        # Test that transform is called
        result = wrapper[0]
        mock_transform.assert_called()
        
        elapsed = time.time() - start_time
        print(f"Transform pair test completed in {elapsed:.4f}s")
        
    def test_stress_large_dataset(self):
        """Stress test contrastive wrapper with large dataset"""
        start_time = time.time()
        
        # Create large dataset
        large_dataset = MockDataset(5000, 50)
        wrapper = ContrastiveWrapper(large_dataset)
        
        # Time multiple accesses
        access_start = time.time()
        for i in range(100):
            result = wrapper[i % len(wrapper)]
            self.assertEqual(len(result), 2)
            self.assertIn(result[1], [0, 1])  # Should be 0 or 1
        access_elapsed = time.time() - access_start
        
        total_elapsed = time.time() - start_time
        print(f"Stress contrastive test completed in {total_elapsed:.4f}s (access: {access_elapsed:.4f}s)")
        
    def test_make_contrastive_function(self):
        """Test make_contrastive factory function"""
        start_time = time.time()
        
        wrapper = make_contrastive(self.mock_dataset, same_class_prob=0.7)
        self.assertIsInstance(wrapper, ContrastiveWrapper)
        self.assertEqual(wrapper.same_class_prob, 0.7)
        
        elapsed = time.time() - start_time
        print(f"Make contrastive function test completed in {elapsed:.4f}s")


class TestFewShotWrapper(unittest.TestCase):
    """Test few-shot learning wrapper"""
    
    def setUp(self):
        self.mock_dataset = MockDataset(100, 10)
        self.wrapper = FewShotWrapper(self.mock_dataset)
        
    def test_initialization(self):
        """Test few-shot wrapper initialization"""
        start_time = time.time()
        
        self.assertEqual(len(self.wrapper), 100)
        self.assertEqual(self.wrapper.n_way, 5)
        self.assertEqual(self.wrapper.k_shot, 1)
        self.assertEqual(self.wrapper.q_query, 15)
        self.assertEqual(self.wrapper.transform_support, None)
        self.assertEqual(self.wrapper.transform_query, None)
        
        elapsed = time.time() - start_time
        print(f"Few-shot initialization test completed in {elapsed:.4f}s")
        
    def test_getitem_structure(self):
        """Test __getitem__ returns correct structure"""
        start_time = time.time()
        
        result = self.wrapper[0]
        self.assertEqual(len(result), 2)
        
        support_set, query_set = result
        self.assertIsInstance(support_set, list)
        self.assertIsInstance(query_set, list)
        
        # Check support set
        self.assertEqual(len(support_set), 5)  # n_way * k_shot
        for img, label in support_set:
            self.assertIsInstance(img, str)
            self.assertIsInstance(label, int)
            
        # Check query set
        self.assertEqual(len(query_set), 75)  # n_way * q_query
        for img, label in query_set:
            self.assertIsInstance(img, str)
            self.assertIsInstance(label, int)
            
        elapsed = time.time() - start_time
        print(f"Getitem structure test completed in {elapsed:.4f}s")
        
    def test_getitem_class_distribution(self):
        """Test that __getitem__ samples from different classes"""
        start_time = time.time()
        
        # Get multiple episodes
        episodes = []
        for _ in range(10):
            result = self.wrapper[0]
            support_set, query_set = result
            
            # Collect all labels from this episode
            episode_labels = set()
            for img, label in support_set + query_set:
                episode_labels.add(label)
                
            episodes.append(episode_labels)
            
        # Check that we get different class combinations
        unique_episodes = set(tuple(sorted(episode)) for episode in episodes)
        self.assertGreater(len(unique_episodes), 1)
        
        elapsed = time.time() - start_time
        print(f"Class distribution test completed in {elapsed:.4f}s")
        
    def test_getitem_with_transforms(self):
        """Test __getitem__ with transforms"""
        start_time = time.time()
        
        # Create mock transforms
        mock_support_transform = MagicMock()
        mock_support_transform.return_value = 'transformed_support'
        mock_query_transform = MagicMock()
        mock_query_transform.return_value = 'transformed_query'
        
        wrapper = FewShotWrapper(
            self.mock_dataset,
            transform_support=mock_support_transform,
            transform_query=mock_query_transform
        )
        
        # Test that transforms are called
        result = wrapper[0]
        support_set, query_set = result
        
        # Check support set transformations
        for img, label in support_set:
            self.assertEqual(img, 'transformed_support')
            
        # Check query set transformations
        for img, label in query_set:
            self.assertEqual(img, 'transformed_query')
            
        elapsed = time.time() - start_time
        print(f"Transforms test completed in {elapsed:.4f}s")
        
    def test_stress_large_dataset(self):
        """Stress test few-shot wrapper with large dataset"""
        start_time = time.time()
        
        # Create large dataset
        large_dataset = MockDataset(10000, 100)
        wrapper = FewShotWrapper(large_dataset, n_way=10, k_shot=5, q_query=20)
        
        # Time multiple episodes
        episode_start = time.time()
        for i in range(50):
            result = wrapper[i % len(wrapper)]
            support_set, query_set = result
            
            # Verify episode structure
            self.assertEqual(len(support_set), 50)  # n_way * k_shot
            self.assertEqual(len(query_set), 200)   # n_way * q_query
            
        episode_elapsed = time.time() - episode_start
        
        total_elapsed = time.time() - start_time
        print(f"Stress few-shot test completed in {total_elapsed:.4f}s (episodes: {episode_elapsed:.4f}s)")
        
    def test_make_few_shot_function(self):
        """Test make_few_shot factory function"""
        start_time = time.time()
        
        wrapper = make_few_shot(
            self.mock_dataset,
            n_way=3,
            k_shot=2,
            q_query=10,
            transform_support='support_transform',
            transform_query='query_transform'
        )
        
        self.assertIsInstance(wrapper, FewShotWrapper)
        self.assertEqual(wrapper.n_way, 3)
        self.assertEqual(wrapper.k_shot, 2)
        self.assertEqual(wrapper.q_query, 10)
        self.assertEqual(wrapper.transform_support, 'support_transform')
        self.assertEqual(wrapper.transform_query, 'query_transform')
        
        elapsed = time.time() - start_time
        print(f"Make few-shot function test completed in {elapsed:.4f}s")


class TestTripletWrapper(unittest.TestCase):
    """Test triplet learning wrapper"""
    
    def setUp(self):
        self.mock_dataset = MockDataset(100, 10)
        self.wrapper = TripletWrapper(self.mock_dataset)
        
    def test_initialization(self):
        """Test triplet wrapper initialization"""
        start_time = time.time()
        
        self.assertEqual(len(self.wrapper), 100)
        self.assertEqual(self.wrapper.transform_pos, None)
        self.assertEqual(self.wrapper.transform_neg, None)
        
        elapsed = time.time() - start_time
        print(f"Triplet initialization test completed in {elapsed:.4f}s")
        
    def test_getitem_structure(self):
        """Test __getitem__ returns correct structure"""
        start_time = time.time()
        
        result = self.wrapper[0]
        self.assertEqual(len(result), 3)
        
        anchor, positive, negative = result
        self.assertIsInstance(anchor, str)
        self.assertIsInstance(positive, str)
        self.assertIsInstance(negative, str)
        
        elapsed = time.time() - start_time
        print(f"Getitem structure test completed in {elapsed:.4f}s")
        
    def test_getitem_class_relationships(self):
        """Test that positive and negative samples have correct class relationships"""
        start_time = time.time()
        
        # Get the original label for the anchor
        anchor_idx = 0
        anchor_img, anchor_label = self.mock_dataset[anchor_idx]
        
        # Get triplet
        anchor, positive, negative = self.wrapper[anchor_idx]
        
        # Find the labels for positive and negative samples
        positive_idx = None
        negative_idx = None
        
        for i, (img, label) in enumerate(self.mock_dataset.samples):
            if img == positive:
                positive_idx = i
            elif img == negative:
                negative_idx = i
                
        self.assertIsNotNone(positive_idx)
        self.assertIsNotNone(negative_idx)
        
        # Check that positive has same label as anchor
        positive_img, positive_label = self.mock_dataset[positive_idx]
        self.assertEqual(positive_label, anchor_label)
        
        # Check that negative has different label than anchor
        negative_img, negative_label = self.mock_dataset[negative_idx]
        self.assertNotEqual(negative_label, anchor_label)
        
        elapsed = time.time() - start_time
        print(f"Class relationships test completed in {elapsed:.4f}s")
        
    def test_getitem_with_transforms(self):
        """Test __getitem__ with transforms"""
        start_time = time.time()
        
        # Create mock transforms
        mock_pos_transform = MagicMock()
        mock_pos_transform.return_value = 'transformed_positive'
        mock_neg_transform = MagicMock()
        mock_neg_transform.return_value = 'transformed_negative'
        
        wrapper = TripletWrapper(
            self.mock_dataset,
            transform_pos=mock_pos_transform,
            transform_neg=mock_neg_transform
        )
        
        # Test that transforms are called
        result = wrapper[0]
        anchor, positive, negative = result
        
        # Check that positive and negative are transformed
        self.assertEqual(positive, 'transformed_positive')
        self.assertEqual(negative, 'transformed_negative')
        
        # Anchor should remain unchanged
        self.assertNotEqual(anchor, 'transformed_positive')
        self.assertNotEqual(anchor, 'transformed_negative')
        
        elapsed = time.time() - start_time
        print(f"Transforms test completed in {elapsed:.4f}s")
        
    def test_stress_large_dataset(self):
        """Stress test triplet wrapper with large dataset"""
        start_time = time.time()
        
        # Create large dataset
        large_dataset = MockDataset(5000, 50)
        wrapper = TripletWrapper(large_dataset)
        
        # Time multiple accesses
        access_start = time.time()
        for i in range(100):
            result = wrapper[i % len(wrapper)]
            self.assertEqual(len(result), 3)
            
            # Verify all elements are different
            anchor, positive, negative = result
            self.assertNotEqual(anchor, positive)
            self.assertNotEqual(anchor, negative)
            self.assertNotEqual(positive, negative)
            
        access_elapsed = time.time() - access_start
        
        total_elapsed = time.time() - start_time
        print(f"Stress triplet test completed in {total_elapsed:.4f}s (access: {access_elapsed:.4f}s)")
        
    def test_make_triplet_function(self):
        """Test make_triplet factory function"""
        start_time = time.time()
        
        wrapper = make_triplet(
            self.mock_dataset,
            transform_pos='pos_transform',
            transform_neg='neg_transform'
        )
        
        self.assertIsInstance(wrapper, TripletWrapper)
        self.assertEqual(wrapper.transform_pos, 'pos_transform')
        self.assertEqual(wrapper.transform_neg, 'neg_transform')
        
        elapsed = time.time() - start_time
        print(f"Make triplet function test completed in {elapsed:.4f}s")


class TestMetricLearningIntegration(unittest.TestCase):
    """Integration tests for metric learning components"""
    
    def setUp(self):
        self.mock_dataset = MockDataset(1000, 20)
        
    def test_wrapper_chain(self):
        """Test chaining multiple wrappers together"""
        start_time = time.time()
        
        # Create contrastive wrapper
        contrastive = ContrastiveWrapper(self.mock_dataset)
        
        # Create few-shot wrapper on top
        few_shot = FewShotWrapper(contrastive)
        
        # Create triplet wrapper on top
        triplet = TripletWrapper(few_shot)
        
        # Test that the chain works
        self.assertEqual(len(triplet), 1000)
        
        # Test that we can still access the underlying dataset structure
        self.assertEqual(len(triplet.labels), 20)
        
        elapsed = time.time() - start_time
        print(f"Wrapper chain test completed in {elapsed:.4f}s")
        
    def test_stress_wrapper_chain(self):
        """Stress test wrapper chain with large dataset"""
        start_time = time.time()
        
        # Create large dataset
        large_dataset = MockDataset(10000, 100)
        
        # Time wrapper chain creation
        chain_start = time.time()
        contrastive = ContrastiveWrapper(large_dataset)
        few_shot = FewShotWrapper(contrastive, n_way=5, k_shot=3, q_query=15)
        triplet = TripletWrapper(few_shot)
        chain_elapsed = time.time() - chain_start
        
        # Test access through the chain
        access_start = time.time()
        for i in range(50):
            result = triplet[i % len(triplet)]
            self.assertEqual(len(result), 3)
        access_elapsed = time.time() - access_start
        
        total_elapsed = time.time() - start_time
        print(f"Stress wrapper chain test completed in {total_elapsed:.4f}s (chain: {chain_elapsed:.4f}s, access: {access_elapsed:.4f}s)")


if __name__ == '__main__':
    # Run tests with timing
    print("Running vision metric learning tests...")
    unittest.main(verbosity=2)
