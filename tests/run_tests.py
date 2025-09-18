#!/usr/bin/env python3
"""
Comprehensive test runner for torch-datasets vision modules.
Runs all tests with timing, performance monitoring, and detailed reporting.
"""

import os
import sys
import time
import unittest
import argparse
import subprocess
import multiprocessing as mp
from pathlib import Path
import psutil
import gc

# Add the parent directory to the path so we can import torchdatasets
sys.path.insert(0, str(Path(__file__).parent.parent))

def get_system_info():
    """Get system information for test reporting"""
    info = {
        'cpu_count': mp.cpu_count(),
        'memory_total': psutil.virtual_memory().total / (1024**3),  # GB
        'memory_available': psutil.virtual_memory().available / (1024**3),  # GB
        'python_version': sys.version,
        'platform': sys.platform
    }
    return info

def run_test_suite(test_file, verbose=True, timing=True):
    """Run a specific test suite and return results"""
    print(f"\n{'='*60}")
    print(f"Running tests from: {test_file}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Load and run the test suite
    loader = unittest.TestLoader()
    suite = loader.discover(os.path.dirname(test_file), pattern=os.path.basename(test_file))
    
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)
    
    elapsed_time = time.time() - start_time
    
    if timing:
        print(f"\nTest suite completed in {elapsed_time:.4f} seconds")
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  {test}: {traceback}")
                
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  {test}: {traceback}")
    
    return result, elapsed_time

def run_all_tests(verbose=True, timing=True, performance=True, stress=True):
    """Run all test suites"""
    print("Starting comprehensive test suite for torch-datasets vision modules...")
    print(f"System Info: {get_system_info()}")
    
    # Test files to run
    test_files = [
        'tests/test_vision_classification.py',
        'tests/test_vision_segmentation.py',
        'tests/test_vision_metric_learning.py'
    ]
    
    if performance:
        test_files.append('tests/test_vision_performance.py')
    
    if stress:
        test_files.append('tests/test_vision_performance.py')  # Contains stress tests too
    
    results = {}
    total_start_time = time.time()
    
    # Run each test suite
    for test_file in test_files:
        if os.path.exists(test_file):
            try:
                result, elapsed = run_test_suite(test_file, verbose, timing)
                results[test_file] = {
                    'result': result,
                    'elapsed_time': elapsed,
                    'tests_run': result.testsRun,
                    'failures': len(result.failures),
                    'errors': len(result.errors)
                }
            except Exception as e:
                print(f"Error running {test_file}: {e}")
                results[test_file] = {
                    'result': None,
                    'elapsed_time': 0,
                    'tests_run': 0,
                    'failures': 0,
                    'errors': 1,
                    'error': str(e)
                }
        else:
            print(f"Test file not found: {test_file}")
    
    total_time = time.time() - total_start_time
    
    # Print summary
    print_summary(results, total_time)
    
    return results

def print_summary(results, total_time):
    """Print comprehensive test summary"""
    print("\n" + "="*80)
    print("COMPREHENSIVE TEST SUMMARY")
    print("="*80)
    
    total_tests = 0
    total_failures = 0
    total_errors = 0
    
    for test_file, result_info in results.items():
        print(f"\n{test_file}:")
        print(f"  Time: {result_info['elapsed_time']:.4f}s")
        print(f"  Tests: {result_info['tests_run']}")
        print(f"  Failures: {result_info['failures']}")
        print(f"  Errors: {result_info['errors']}")
        
        if 'error' in result_info:
            print(f"  Error: {result_info['error']}")
            
        total_tests += result_info['tests_run']
        total_failures += result_info['failures']
        total_errors += result_info['errors']
    
    print(f"\n{'='*80}")
    print(f"TOTAL SUMMARY:")
    print(f"  Total Time: {total_time:.4f}s")
    print(f"  Total Tests: {total_tests}")
    print(f"  Total Failures: {total_failures}")
    print(f"  Total Errors: {total_errors}")
    print(f"  Success Rate: {((total_tests - total_failures - total_errors) / total_tests * 100):.1f}%" if total_tests > 0 else "N/A")
    print(f"{'='*80}")
    
    # Overall result
    if total_failures == 0 and total_errors == 0:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        return True
    else:
        print(f"\n❌ {total_failures + total_errors} TESTS FAILED ❌")
        return False

def run_individual_test(test_name, verbose=True):
    """Run a specific test by name"""
    print(f"Running individual test: {test_name}")
    
    # Discover and run the specific test
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(test_name)
    
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)
    
    return result

def run_performance_benchmark():
    """Run performance benchmark tests"""
    print("\nRunning performance benchmarks...")
    
    try:
        # Import and run performance tests
        from tests.vision.performance import VisionDatasetPerformanceTest
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(VisionDatasetPerformanceTest)
        
        # Run with custom result collector
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result
        
    except ImportError as e:
        print(f"Could not import performance tests: {e}")
        return None

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Run torch-datasets vision tests')
    parser.add_argument('--test', '-t', help='Run specific test by name')
    parser.add_argument('--file', '-f', help='Run tests from specific file')
    parser.add_argument('--performance', '-p', action='store_true', help='Run performance tests')
    parser.add_argument('--stress', '-s', action='store_true', help='Run stress tests')
    parser.add_argument('--all', '-a', action='store_true', help='Run all tests including performance and stress')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--no-timing', action='store_true', help='Disable timing information')
    parser.add_argument('--benchmark', '-b', action='store_true', help='Run performance benchmark only')
    
    args = parser.parse_args()
    
    # Set defaults
    verbose = args.verbose
    timing = not args.no_timing
    performance = args.performance or args.all
    stress = args.stress or args.all
    
    if args.benchmark:
        print("Running performance benchmark...")
        result = run_performance_benchmark()
        return 0 if result and not result.failures and not result.errors else 1
    
    if args.test:
        print(f"Running specific test: {args.test}")
        result = run_individual_test(args.test, verbose)
        return 0 if not result.failures and not result.errors else 1
    
    if args.file:
        if os.path.exists(args.file):
            print(f"Running tests from file: {args.file}")
            result, elapsed = run_test_suite(args.file, verbose, timing)
            return 0 if not result.failures and not result.errors else 1
        else:
            print(f"Test file not found: {args.file}")
            return 1
    
    # Run all tests
    print("Running comprehensive test suite...")
    results = run_all_tests(verbose, timing, performance, stress)
    
    # Return exit code
    total_failures = sum(r['failures'] for r in results.values())
    total_errors = sum(r['errors'] for r in results.values())
    
    return 0 if total_failures == 0 and total_errors == 0 else 1

if __name__ == '__main__':
    # Set multiprocessing start method for compatibility
    if sys.platform.startswith('win'):
        mp.set_start_method('spawn', force=True)
    
    exit_code = main()
    sys.exit(exit_code)
