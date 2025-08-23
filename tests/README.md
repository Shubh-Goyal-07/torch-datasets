# Vision Datasets Test Suite

This directory contains comprehensive tests for the torch-datasets vision modules, including timing, performance monitoring, and stress testing.

## 🚀 Features

- **Complete Coverage**: Tests for all vision dataset types (classification, segmentation, metric learning)
- **Performance Testing**: Timing and memory usage monitoring for all operations
- **Stress Testing**: Extreme load testing with large datasets
- **Temporary Data**: No physical data storage - uses temporary files and mocks
- **Comprehensive Reporting**: Detailed timing and performance metrics
- **Concurrent Testing**: Multi-threaded and multi-process testing capabilities

## 📁 Test Structure

```
tests/
├── __init__.py                           # Test package initialization
├── test_vision_classification.py         # Classification dataset tests
├── test_vision_segmentation.py          # Segmentation dataset tests
├── test_vision_metric_learning.py       # Metric learning wrapper tests
├── test_vision_performance.py           # Performance and stress tests
├── run_tests.py                         # Comprehensive test runner
├── requirements.txt                      # Test dependencies
└── README.md                            # This file
```

## 🧪 Test Categories

### 1. Classification Tests (`test_vision_classification.py`)
- **BaseImageClassificationDataset**: Core functionality, transforms, path handling
- **ImageCSVXLSXDataset**: CSV/Excel file loading, multi-label support
- **ImageSingleDirDataset**: Single directory with filename-based labeling
- **ImageSubdirDataset**: Subdirectory-based class organization

### 2. Segmentation Tests (`test_vision_segmentation.py`)
- **BaseImageSegmentationDataset**: Core segmentation functionality
- **ImageCSVXLSXDataset**: CSV-based image-mask pairs
- **ImageSubdirDataset**: Directory-based image-mask organization

### 3. Metric Learning Tests (`test_vision_metric_learning.py`)
- **BaseMetricLearningWrapper**: Base wrapper functionality
- **ContrastiveWrapper**: Contrastive learning pairs
- **FewShotWrapper**: Few-shot learning episodes
- **TripletWrapper**: Triplet learning samples

### 4. Performance Tests (`test_vision_performance.py`)
- **Performance Testing**: Dataset creation and access timing
- **Memory Monitoring**: Memory usage tracking
- **Stress Testing**: Extreme load scenarios
- **Concurrent Access**: Multi-threaded dataset access
- **Resource Cleanup**: Memory leak detection

## 🚀 Quick Start

### Install Dependencies
```bash
pip install -r tests/requirements.txt
```

### Run All Tests
```bash
python tests/run_tests.py --all
```

### Run Specific Test Categories
```bash
# Classification tests only
python tests/run_tests.py --file tests/test_vision_classification.py

# Performance tests only
python tests/run_tests.py --performance

# Stress tests only
python tests/run_tests.py --stress
```

### Run Individual Tests
```bash
# Run specific test method
python tests/run_tests.py --test TestBaseImageClassificationDataset.test_initialization

# Run with verbose output
python tests/run_tests.py --test TestBaseImageClassificationDataset.test_initialization --verbose
```

## 📊 Performance Testing

The test suite includes comprehensive performance monitoring:

### Metrics Tracked
- **Execution Time**: Individual test and suite timing
- **Memory Usage**: Memory consumption and cleanup efficiency
- **Dataset Creation**: Time to create datasets of various sizes
- **Data Access**: Time to access samples from datasets
- **Resource Cleanup**: Memory leak detection

### Stress Test Scenarios
- **Large Datasets**: Up to 50,000 samples with 1,000 classes
- **Memory Pressure**: Multiple large datasets simultaneously
- **Concurrent Access**: Multi-threaded dataset operations
- **Extreme Sizes**: Testing system limits

## 🔧 Test Runner Options

```bash
python tests/run_tests.py [OPTIONS]

Options:
  --test, -t TEXT          Run specific test by name
  --file, -f TEXT          Run tests from specific file
  --performance, -p        Run performance tests
  --stress, -s             Run stress tests
  --all, -a                Run all tests including performance and stress
  --verbose, -v            Verbose output
  --no-timing              Disable timing information
  --benchmark, -b          Run performance benchmark only
```

## 📈 Performance Benchmarks

### Expected Performance (on modern hardware)
- **Small Dataset (1K samples)**: < 1 second creation, < 0.1 second access
- **Medium Dataset (10K samples)**: < 5 seconds creation, < 0.5 second access
- **Large Dataset (50K samples)**: < 20 seconds creation, < 2 seconds access

### Memory Usage Guidelines
- **Small Dataset**: < 50MB
- **Medium Dataset**: < 200MB
- **Large Dataset**: < 1GB

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the project root directory
2. **Memory Errors**: Reduce dataset sizes in stress tests
3. **Permission Errors**: Check file permissions for temporary directories
4. **Performance Issues**: Monitor system resources during testing

### Debug Mode
```bash
# Run with maximum verbosity
python tests/run_tests.py --verbose --all

# Run specific failing test
python tests/run_tests.py --test TestName.test_method --verbose
```

## 🔍 Test Coverage

The test suite covers:

- ✅ **Functionality**: All public methods and properties
- ✅ **Edge Cases**: Invalid inputs, empty datasets, missing files
- ✅ **Performance**: Timing and memory usage
- ✅ **Stress**: Large datasets and concurrent access
- ✅ **Integration**: Dataset wrapper combinations
- ✅ **Error Handling**: Exception cases and error messages

## 📝 Adding New Tests

### Test Structure
```python
class TestNewFeature(unittest.TestCase):
    def setUp(self):
        # Setup test environment
        pass
        
    def tearDown(self):
        # Cleanup test environment
        pass
        
    def test_feature_functionality(self):
        # Test the feature
        start_time = time.time()
        
        # Your test code here
        
        elapsed = time.time() - start_time
        print(f"Feature test completed in {elapsed:.4f}s")
```

### Performance Testing
```python
def test_performance(self):
    start_time = time.time()
    initial_memory = self.get_memory_usage()
    
    # Your performance test here
    
    total_time = time.time() - start_time
    memory_used = self.get_memory_usage() - initial_memory
    
    self.log_performance(
        'Test Name',
        total_time,
        memory_used,
        {'additional_info': 'value'}
    )
```

## 🤝 Contributing

When adding new tests:

1. Follow the existing naming conventions
2. Include timing measurements
3. Add stress tests for new features
4. Update this README if adding new test categories
5. Ensure tests work without external data dependencies

## 📊 Continuous Integration

The test suite is designed to work with CI/CD systems:

- **Fast Tests**: Core functionality tests run quickly
- **Performance Tests**: Optional performance monitoring
- **Stress Tests**: Optional stress testing for release validation
- **Exit Codes**: Proper exit codes for CI integration

## 📚 Additional Resources

- [PyTorch Testing Guide](https://pytorch.org/docs/stable/testing.html)
- [Python unittest Documentation](https://docs.python.org/3/library/unittest.html)
- [Performance Testing Best Practices](https://docs.pytest.org/en/stable/usage.html)
