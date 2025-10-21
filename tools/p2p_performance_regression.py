#!/usr/bin/env python3
"""
P2P Performance Regression Testing Tool

This tool provides comprehensive performance regression testing for the P2P GPU communication implementation,
including baseline comparison, performance trend analysis, and regression detection.
"""

import os
import sys
import time
import json
import argparse
import tempfile
import shutil
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import pandas as pd

# Add the exllamav3 directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import torch
    from exllamav3.model.model_tp_p2p import P2PTopology
    from exllamav3.model.model_tp_backend import TPBackendP2P, TPBackendNative
    from exllamav3.util.p2p_monitor import P2PMonitor
    from exllamav3.util.p2p_profiler import P2PProfiler
    from exllamav3.ext import exllamav3_ext as ext
    P2P_AVAILABLE = True
except ImportError as e:
    print(f"P2P modules not available: {e}")
    P2P_AVAILABLE = False


@dataclass
class PerformanceBenchmark:
    """Data class for performance benchmark results."""
    timestamp: float
    operation_type: str
    algorithm: str
    tensor_size: int
    num_devices: int
    bandwidth_gbps: float
    latency_us: float
    throughput_ops_per_sec: float
    memory_usage_mb: float
    success_rate: float
    device_info: Dict[str, Any]
    test_metadata: Dict[str, Any]


class P2PPerformanceRegressionTester:
    """Performance regression tester for P2P operations."""
    
    def __init__(self, active_devices: List[int] = None, 
                 baseline_file: str = None, output_dir: str = "./p2p_regression"):
        """
        Initialize the regression tester.
        
        Args:
            active_devices: List of active GPU device IDs
            baseline_file: Path to baseline performance file
            output_dir: Directory to store regression test results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine active devices
        if active_devices is None:
            if torch.cuda.is_available():
                self.active_devices = list(range(torch.cuda.device_count()))
            else:
                self.active_devices = []
        else:
            self.active_devices = active_devices
        
        self.num_devices = len(self.active_devices)
        
        # Baseline data
        self.baseline_file = baseline_file
        self.baseline_data = {}
        if baseline_file and os.path.exists(baseline_file):
            self._load_baseline()
        
        # Current test results
        self.current_results = []
        
        # Regression thresholds
        self.regression_thresholds = {
            "bandwidth_degradation": 0.1,  # 10% degradation
            "latency_increase": 0.1,       # 10% increase
            "throughput_degradation": 0.1, # 10% degradation
            "success_rate_drop": 0.05      # 5% drop
        }
        
        # Initialize components
        self.topology = None
        self.backends = {}
        self.monitor = None
        self.profiler = None
        
        if P2P_AVAILABLE and self.num_devices > 0:
            try:
                self.topology = P2PTopology(self.active_devices)
                self.monitor = P2PMonitor(active_devices=self.active_devices, output_dir=output_dir)
                self.profiler = P2PProfiler(output_dir=output_dir)
            except Exception as e:
                print(f"Failed to initialize P2P components: {e}")
    
    def _load_baseline(self):
        """Load baseline performance data."""
        try:
            with open(self.baseline_file, 'r') as f:
                baseline_data = json.load(f)
            
            # Convert to PerformanceBenchmark objects
            for item in baseline_data.get("benchmarks", []):
                benchmark = PerformanceBenchmark(**item)
                key = self._get_benchmark_key(benchmark)
                self.baseline_data[key] = benchmark
            
            print(f"Loaded {len(self.baseline_data)} baseline benchmarks")
            
        except Exception as e:
            print(f"Failed to load baseline data: {e}")
    
    def _get_benchmark_key(self, benchmark: PerformanceBenchmark) -> str:
        """Get unique key for a benchmark."""
        return f"{benchmark.operation_type}_{benchmark.algorithm}_{benchmark.tensor_size}_{benchmark.num_devices}"
    
    def _initialize_backends(self) -> bool:
        """Initialize P2P backends."""
        try:
            for device in self.active_devices:
                backend = TPBackendP2P(
                    device=device,
                    active_devices=self.active_devices,
                    output_device=self.active_devices[0],
                    init_method="tcp://localhost:12345",
                    master=(device == self.active_devices[0]),
                    uuid="p2p_regression"
                )
                self.backends[device] = backend
            return True
        except Exception as e:
            print(f"Failed to initialize backends: {e}")
            return False
    
    def _cleanup_backends(self):
        """Clean up backends."""
        for device, backend in self.backends.items():
            try:
                backend.close()
            except Exception:
                pass
        self.backends.clear()
    
    def run_regression_tests(self, test_config: Dict = None) -> Dict:
        """
        Run comprehensive regression tests.
        
        Args:
            test_config: Configuration for regression tests
            
        Returns:
            Dictionary with regression test results
        """
        print("Starting P2P Performance Regression Tests...")
        print(f"Active devices: {self.active_devices}")
        print(f"Output directory: {self.output_dir}")
        print()
        
        # Initialize backends
        if not self._initialize_backends():
            return {"error": "Failed to initialize backends"}
        
        try:
            # Run regression test categories
            regression_results = {
                "timestamp": time.time(),
                "num_devices": self.num_devices,
                "active_devices": self.active_devices,
                "test_categories": {},
                "regression_detected": False,
                "summary": {}
            }
            
            # Test basic operations
            category_result = self._test_basic_operations_regression()
            regression_results["test_categories"]["basic_operations"] = category_result
            
            # Test scalability
            category_result = self._test_scalability_regression()
            regression_results["test_categories"]["scalability"] = category_result
            
            # Test algorithm performance
            category_result = self._test_algorithm_regression()
            regression_results["test_categories"]["algorithms"] = category_result
            
            # Test memory efficiency
            category_result = self._test_memory_regression()
            regression_results["test_categories"]["memory"] = category_result
            
            # Test stress performance
            category_result = self._test_stress_regression()
            regression_results["test_categories"]["stress"] = category_result
            
            # Analyze regression results
            regression_results["regression_detected"] = self._analyze_regression(regression_results)
            
            # Generate summary
            regression_results["summary"] = self._generate_regression_summary(regression_results)
            
            # Save results
            self._save_regression_results(regression_results)
            
            # Generate plots
            self._generate_regression_plots(regression_results)
            
            return regression_results
        
        finally:
            self._cleanup_backends()
    
    def _test_basic_operations_regression(self) -> Dict:
        """Test basic operations for regression."""
        print("Testing basic operations regression...")
        
        category_result = {
            "status": "unknown",
            "benchmarks": [],
            "regressions": [],
            "improvements": []
        }
        
        try:
            # Test configurations
            operations = ["broadcast", "all_reduce", "gather"]
            tensor_sizes = [1024, 1024*1024, 4*1024*1024]
            
            device = self.active_devices[0]
            backend = self.backends[device]
            
            for operation in operations:
                for tensor_size in tensor_sizes:
                    # Run benchmark
                    benchmark = self._run_operation_benchmark(backend, operation, tensor_size)
                    category_result["benchmarks"].append(benchmark)
                    
                    # Check regression against baseline
                    regression = self._check_regression(benchmark)
                    if regression:
                        category_result["regressions"].append(regression)
                    
                    # Check for improvements
                    improvement = self._check_improvement(benchmark)
                    if improvement:
                        category_result["improvements"].append(improvement)
            
            # Determine status
            if category_result["regressions"]:
                category_result["status"] = "regression_detected"
            elif category_result["improvements"]:
                category_result["status"] = "improvements_detected"
            else:
                category_result["status"] = "no_regression"
        
        except Exception as e:
            category_result["status"] = "error"
            category_result["error"] = str(e)
        
        print(f"Basic operations regression: {category_result['status']}")
        return category_result
    
    def _test_scalability_regression(self) -> Dict:
        """Test scalability for regression."""
        print("Testing scalability regression...")
        
        category_result = {
            "status": "unknown",
            "benchmarks": [],
            "regressions": [],
            "improvements": []
        }
        
        try:
            # Test with different device counts
            max_devices = min(self.num_devices, 8)
            
            for num_devices in range(2, max_devices + 1):
                # Use subset of devices
                test_devices = self.active_devices[:num_devices]
                
                # Create temporary topology and backends
                temp_backends = {}
                try:
                    for device in test_devices:
                        backend = TPBackendP2P(
                            device=device,
                            active_devices=test_devices,
                            output_device=test_devices[0],
                            init_method="tcp://localhost:12347",
                            master=(device == test_devices[0]),
                            uuid=f"scalability_test_{num_devices}"
                        )
                        temp_backends[device] = backend
                    
                    # Test scalability
                    tensor_size = 1024*1024 // num_devices  # Keep total data constant
                    device = test_devices[0]
                    backend = temp_backends[device]
                    
                    benchmark = self._run_operation_benchmark(backend, "all_reduce", tensor_size, num_devices)
                    benchmark.test_metadata["scalability_test"] = True
                    category_result["benchmarks"].append(benchmark)
                    
                    # Check regression
                    regression = self._check_regression(benchmark)
                    if regression:
                        category_result["regressions"].append(regression)
                    
                    # Check improvements
                    improvement = self._check_improvement(benchmark)
                    if improvement:
                        category_result["improvements"].append(improvement)
                
                finally:
                    # Clean up temporary backends
                    for backend in temp_backends.values():
                        try:
                            backend.close()
                        except Exception:
                            pass
            
            # Determine status
            if category_result["regressions"]:
                category_result["status"] = "regression_detected"
            elif category_result["improvements"]:
                category_result["status"] = "improvements_detected"
            else:
                category_result["status"] = "no_regression"
        
        except Exception as e:
            category_result["status"] = "error"
            category_result["error"] = str(e)
        
        print(f"Scalability regression: {category_result['status']}")
        return category_result
    
    def _test_algorithm_regression(self) -> Dict:
        """Test algorithm performance for regression."""
        print("Testing algorithm regression...")
        
        category_result = {
            "status": "unknown",
            "benchmarks": [],
            "regressions": [],
            "improvements": []
        }
        
        try:
            # Test different algorithms
            algorithms = ["ring", "binary_tree", "kary_tree", "balanced_tree"]
            tensor_size = 1024*1024
            
            device = self.active_devices[0]
            backend = self.backends[device]
            
            for algorithm in algorithms:
                # Run benchmark with specific algorithm
                benchmark = self._run_algorithm_benchmark(backend, "all_reduce", tensor_size, algorithm)
                category_result["benchmarks"].append(benchmark)
                
                # Check regression
                regression = self._check_regression(benchmark)
                if regression:
                    category_result["regressions"].append(regression)
                
                # Check improvements
                improvement = self._check_improvement(benchmark)
                if improvement:
                    category_result["improvements"].append(improvement)
            
            # Determine status
            if category_result["regressions"]:
                category_result["status"] = "regression_detected"
            elif category_result["improvements"]:
                category_result["status"] = "improvements_detected"
            else:
                category_result["status"] = "no_regression"
        
        except Exception as e:
            category_result["status"] = "error"
            category_result["error"] = str(e)
        
        print(f"Algorithm regression: {category_result['status']}")
        return category_result
    
    def _test_memory_regression(self) -> Dict:
        """Test memory efficiency for regression."""
        print("Testing memory regression...")
        
        category_result = {
            "status": "unknown",
            "benchmarks": [],
            "regressions": [],
            "improvements": []
        }
        
        try:
            # Test different tensor sizes
            tensor_sizes = [1024, 1024*1024, 4*1024*1024, 16*1024*1024]
            
            device = self.active_devices[0]
            backend = self.backends[device]
            
            for tensor_size in tensor_sizes:
                # Run memory benchmark
                benchmark = self._run_memory_benchmark(backend, "all_reduce", tensor_size)
                category_result["benchmarks"].append(benchmark)
                
                # Check regression
                regression = self._check_regression(benchmark)
                if regression:
                    category_result["regressions"].append(regression)
                
                # Check improvements
                improvement = self._check_improvement(benchmark)
                if improvement:
                    category_result["improvements"].append(improvement)
            
            # Determine status
            if category_result["regressions"]:
                category_result["status"] = "regression_detected"
            elif category_result["improvements"]:
                category_result["status"] = "improvements_detected"
            else:
                category_result["status"] = "no_regression"
        
        except Exception as e:
            category_result["status"] = "error"
            category_result["error"] = str(e)
        
        print(f"Memory regression: {category_result['status']}")
        return category_result
    
    def _test_stress_regression(self) -> Dict:
        """Test stress performance for regression."""
        print("Testing stress regression...")
        
        category_result = {
            "status": "unknown",
            "benchmarks": [],
            "regressions": [],
            "improvements": []
        }
        
        try:
            # Test concurrent operations
            num_threads = 4
            ops_per_thread = 10
            
            device = self.active_devices[0]
            backend = self.backends[device]
            
            # Run stress benchmark
            benchmark = self._run_stress_benchmark(backend, num_threads, ops_per_thread)
            category_result["benchmarks"].append(benchmark)
            
            # Check regression
            regression = self._check_regression(benchmark)
            if regression:
                category_result["regressions"].append(regression)
            
            # Check improvements
            improvement = self._check_improvement(benchmark)
            if improvement:
                category_result["improvements"].append(improvement)
            
            # Determine status
            if category_result["regressions"]:
                category_result["status"] = "regression_detected"
            elif category_result["improvements"]:
                category_result["status"] = "improvements_detected"
            else:
                category_result["status"] = "no_regression"
        
        except Exception as e:
            category_result["status"] = "error"
            category_result["error"] = str(e)
        
        print(f"Stress regression: {category_result['status']}")
        return category_result
    
    def _run_operation_benchmark(self, backend, operation_type: str, 
                               tensor_size: int, num_devices: int = None) -> PerformanceBenchmark:
        """Run a single operation benchmark."""
        if num_devices is None:
            num_devices = self.num_devices
        
        device = backend.device
        
        # Create test tensor
        tensor = torch.randn(tensor_size, dtype=torch.float32, device=device)
        
        # Get initial memory usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated(device)
        
        # Warm up
        for _ in range(3):
            try:
                if operation_type == "broadcast":
                    backend.broadcast(tensor, device)
                elif operation_type == "all_reduce":
                    test_tensor = tensor.clone()
                    backend.all_reduce(test_tensor)
                elif operation_type == "gather":
                    out_tensor = torch.zeros(tensor_size * num_devices, dtype=torch.float32, device=device)
                    gather_devices = torch.tensor(self.active_devices[:num_devices], dtype=torch.int)
                    ldims = [tensor_size] * num_devices
                    backend.gather(tensor, out_tensor, gather_devices, device, ldims)
            except Exception:
                pass
        
        # Benchmark
        num_iterations = 10
        successful_ops = 0
        total_time = 0.0
        
        for i in range(num_iterations):
            try:
                start_time = time.time()
                
                if operation_type == "broadcast":
                    backend.broadcast(tensor, device)
                elif operation_type == "all_reduce":
                    test_tensor = tensor.clone()
                    backend.all_reduce(test_tensor)
                elif operation_type == "gather":
                    out_tensor = torch.zeros(tensor_size * num_devices, dtype=torch.float32, device=device)
                    gather_devices = torch.tensor(self.active_devices[:num_devices], dtype=torch.int)
                    ldims = [tensor_size] * num_devices
                    backend.gather(tensor, out_tensor, gather_devices, device, ldims)
                
                end_time = time.time()
                total_time += (end_time - start_time)
                successful_ops += 1
                
            except Exception as e:
                print(f"Benchmark iteration {i} failed: {e}")
        
        # Get final memory usage
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated(device)
            memory_usage_mb = (final_memory - initial_memory) / (1024**2)
        else:
            memory_usage_mb = 0.0
        
        # Calculate metrics
        avg_time = total_time / max(successful_ops, 1)
        throughput_ops_per_sec = successful_ops / total_time if total_time > 0 else 0.0
        
        # Calculate bandwidth
        tensor_bytes = tensor_size * 4  # float32 = 4 bytes
        if operation_type in ["broadcast", "gather"]:
            total_data_bytes = tensor_bytes * num_devices
        else:
            total_data_bytes = tensor_bytes * (num_devices - 1)
        
        bandwidth_gbps = (total_data_bytes / avg_time) / (1024**3) if avg_time > 0 else 0.0
        
        # Calculate latency
        latency_us = (avg_time * 1e6) / num_devices if num_devices > 0 else 0.0
        
        # Get device info
        device_info = self._get_device_info(device)
        
        # Create benchmark
        benchmark = PerformanceBenchmark(
            timestamp=time.time(),
            operation_type=operation_type,
            algorithm="default",
            tensor_size=tensor_size,
            num_devices=num_devices,
            bandwidth_gbps=bandwidth_gbps,
            latency_us=latency_us,
            throughput_ops_per_sec=throughput_ops_per_sec,
            memory_usage_mb=memory_usage_mb,
            success_rate=successful_ops / num_iterations,
            device_info=device_info,
            test_metadata={
                "num_iterations": num_iterations,
                "successful_ops": successful_ops,
                "total_time": total_time
            }
        )
        
        self.current_results.append(benchmark)
        return benchmark
    
    def _run_algorithm_benchmark(self, backend, operation_type: str,
                              tensor_size: int, algorithm: str) -> PerformanceBenchmark:
        """Run a benchmark with specific algorithm."""
        device = backend.device
        
        # Create test tensor
        tensor = torch.randn(tensor_size, dtype=torch.float32, device=device)
        
        # Get initial memory usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated(device)
        
        # Warm up
        for _ in range(3):
            try:
                if operation_type == "all_reduce":
                    test_tensor = tensor.clone()
                    backend.all_reduce(test_tensor)
            except Exception:
                pass
        
        # Benchmark
        num_iterations = 10
        successful_ops = 0
        total_time = 0.0
        
        for i in range(num_iterations):
            try:
                start_time = time.time()
                
                if operation_type == "all_reduce":
                    test_tensor = tensor.clone()
                    backend.all_reduce(test_tensor)
                
                end_time = time.time()
                total_time += (end_time - start_time)
                successful_ops += 1
                
            except Exception as e:
                print(f"Algorithm benchmark iteration {i} failed: {e}")
        
        # Get final memory usage
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated(device)
            memory_usage_mb = (final_memory - initial_memory) / (1024**2)
        else:
            memory_usage_mb = 0.0
        
        # Calculate metrics
        avg_time = total_time / max(successful_ops, 1)
        throughput_ops_per_sec = successful_ops / total_time if total_time > 0 else 0.0
        
        # Calculate bandwidth
        tensor_bytes = tensor_size * 4  # float32 = 4 bytes
        total_data_bytes = tensor_bytes * (self.num_devices - 1)
        bandwidth_gbps = (total_data_bytes / avg_time) / (1024**3) if avg_time > 0 else 0.0
        
        # Calculate latency
        latency_us = (avg_time * 1e6) / self.num_devices if self.num_devices > 0 else 0.0
        
        # Get device info
        device_info = self._get_device_info(device)
        
        # Create benchmark
        benchmark = PerformanceBenchmark(
            timestamp=time.time(),
            operation_type=operation_type,
            algorithm=algorithm,
            tensor_size=tensor_size,
            num_devices=self.num_devices,
            bandwidth_gbps=bandwidth_gbps,
            latency_us=latency_us,
            throughput_ops_per_sec=throughput_ops_per_sec,
            memory_usage_mb=memory_usage_mb,
            success_rate=successful_ops / num_iterations,
            device_info=device_info,
            test_metadata={
                "num_iterations": num_iterations,
                "successful_ops": successful_ops,
                "total_time": total_time,
                "algorithm_test": True
            }
        )
        
        self.current_results.append(benchmark)
        return benchmark
    
    def _run_memory_benchmark(self, backend, operation_type: str, tensor_size: int) -> PerformanceBenchmark:
        """Run a memory-focused benchmark."""
        device = backend.device
        
        # Create test tensor
        tensor = torch.randn(tensor_size, dtype=torch.float32, device=device)
        
        # Get initial memory usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated(device)
        
        # Warm up
        for _ in range(3):
            try:
                if operation_type == "all_reduce":
                    test_tensor = tensor.clone()
                    backend.all_reduce(test_tensor)
            except Exception:
                pass
        
        # Benchmark with memory tracking
        num_iterations = 5
        successful_ops = 0
        total_time = 0.0
        peak_memory = initial_memory
        
        for i in range(num_iterations):
            try:
                start_time = time.time()
                
                if operation_type == "all_reduce":
                    test_tensor = tensor.clone()
                    backend.all_reduce(test_tensor)
                
                end_time = time.time()
                total_time += (end_time - start_time)
                successful_ops += 1
                
                # Track peak memory
                if torch.cuda.is_available():
                    current_memory = torch.cuda.memory_allocated(device)
                    peak_memory = max(peak_memory, current_memory)
                
            except Exception as e:
                print(f"Memory benchmark iteration {i} failed: {e}")
        
        # Get final memory usage
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated(device)
            memory_usage_mb = (final_memory - initial_memory) / (1024**2)
            peak_memory_mb = (peak_memory - initial_memory) / (1024**2)
        else:
            memory_usage_mb = 0.0
            peak_memory_mb = 0.0
        
        # Calculate metrics
        avg_time = total_time / max(successful_ops, 1)
        throughput_ops_per_sec = successful_ops / total_time if total_time > 0 else 0.0
        
        # Calculate bandwidth
        tensor_bytes = tensor_size * 4  # float32 = 4 bytes
        total_data_bytes = tensor_bytes * (self.num_devices - 1)
        bandwidth_gbps = (total_data_bytes / avg_time) / (1024**3) if avg_time > 0 else 0.0
        
        # Calculate latency
        latency_us = (avg_time * 1e6) / self.num_devices if self.num_devices > 0 else 0.0
        
        # Get device info
        device_info = self._get_device_info(device)
        
        # Create benchmark
        benchmark = PerformanceBenchmark(
            timestamp=time.time(),
            operation_type=operation_type,
            algorithm="default",
            tensor_size=tensor_size,
            num_devices=self.num_devices,
            bandwidth_gbps=bandwidth_gbps,
            latency_us=latency_us,
            throughput_ops_per_sec=throughput_ops_per_sec,
            memory_usage_mb=memory_usage_mb,
            success_rate=successful_ops / num_iterations,
            device_info=device_info,
            test_metadata={
                "num_iterations": num_iterations,
                "successful_ops": successful_ops,
                "total_time": total_time,
                "peak_memory_mb": peak_memory_mb,
                "memory_test": True
            }
        )
        
        self.current_results.append(benchmark)
        return benchmark
    
    def _run_stress_benchmark(self, backend, num_threads: int, ops_per_thread: int) -> PerformanceBenchmark:
        """Run a stress benchmark."""
        device = backend.device
        
        # Get initial memory usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated(device)
        
        # Stress test parameters
        tensor_size = 1024*1024  # 1M elements
        total_operations = num_threads * ops_per_thread
        
        # Track operations
        successful_ops = 0
        failed_ops = 0
        total_time = 0.0
        
        start_time = time.time()
        
        # Simulate concurrent operations (simplified)
        for thread_id in range(num_threads):
            for op_id in range(ops_per_thread):
                try:
                    tensor = torch.randn(tensor_size, dtype=torch.float32, device=device)
                    
                    op_start = time.time()
                    backend.all_reduce(tensor)
                    op_end = time.time()
                    
                    total_time += (op_end - op_start)
                    successful_ops += 1
                    
                except Exception as e:
                    failed_ops += 1
        
        end_time = time.time()
        total_test_time = end_time - start_time
        
        # Get final memory usage
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated(device)
            memory_usage_mb = (final_memory - initial_memory) / (1024**2)
        else:
            memory_usage_mb = 0.0
        
        # Calculate metrics
        avg_time = total_time / max(successful_ops, 1) if successful_ops > 0 else 0.0
        throughput_ops_per_sec = successful_ops / total_test_time if total_test_time > 0 else 0.0
        
        # Calculate bandwidth (simplified)
        tensor_bytes = tensor_size * 4  # float32 = 4 bytes
        total_data_bytes = tensor_bytes * successful_ops * (self.num_devices - 1)
        bandwidth_gbps = (total_data_bytes / total_test_time) / (1024**3) if total_test_time > 0 else 0.0
        
        # Calculate latency
        latency_us = (avg_time * 1e6) if avg_time > 0 else 0.0
        
        # Get device info
        device_info = self._get_device_info(device)
        
        # Create benchmark
        benchmark = PerformanceBenchmark(
            timestamp=time.time(),
            operation_type="stress_all_reduce",
            algorithm="concurrent",
            tensor_size=tensor_size,
            num_devices=self.num_devices,
            bandwidth_gbps=bandwidth_gbps,
            latency_us=latency_us,
            throughput_ops_per_sec=throughput_ops_per_sec,
            memory_usage_mb=memory_usage_mb,
            success_rate=successful_ops / total_operations,
            device_info=device_info,
            test_metadata={
                "num_threads": num_threads,
                "ops_per_thread": ops_per_thread,
                "total_operations": total_operations,
                "successful_ops": successful_ops,
                "failed_ops": failed_ops,
                "total_test_time": total_test_time,
                "stress_test": True
            }
        )
        
        self.current_results.append(benchmark)
        return benchmark
    
    def _get_device_info(self, device: int) -> Dict[str, Any]:
        """Get device information."""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        try:
            props = torch.cuda.get_device_properties(device)
            return {
                "name": props.name,
                "major": props.major,
                "minor": props.minor,
                "total_memory": props.total_memory,
                "multiprocessor_count": props.multiprocessor_count
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _check_regression(self, benchmark: PerformanceBenchmark) -> Optional[Dict]:
        """Check if benchmark shows regression compared to baseline."""
        key = self._get_benchmark_key(benchmark)
        
        if key not in self.baseline_data:
            return None
        
        baseline = self.baseline_data[key]
        
        regressions = []
        
        # Check bandwidth regression
        if baseline.bandwidth_gbps > 0:
            bandwidth_change = (baseline.bandwidth_gbps - benchmark.bandwidth_gbps) / baseline.bandwidth_gbps
            if bandwidth_change > self.regression_thresholds["bandwidth_degradation"]:
                regressions.append({
                    "metric": "bandwidth",
                    "baseline": baseline.bandwidth_gbps,
                    "current": benchmark.bandwidth_gbps,
                    "change": bandwidth_change,
                    "threshold": self.regression_thresholds["bandwidth_degradation"]
                })
        
        # Check latency regression
        if baseline.latency_us > 0:
            latency_change = (benchmark.latency_us - baseline.latency_us) / baseline.latency_us
            if latency_change > self.regression_thresholds["latency_increase"]:
                regressions.append({
                    "metric": "latency",
                    "baseline": baseline.latency_us,
                    "current": benchmark.latency_us,
                    "change": latency_change,
                    "threshold": self.regression_thresholds["latency_increase"]
                })
        
        # Check throughput regression
        if baseline.throughput_ops_per_sec > 0:
            throughput_change = (baseline.throughput_ops_per_sec - benchmark.throughput_ops_per_sec) / baseline.throughput_ops_per_sec
            if throughput_change > self.regression_thresholds["throughput_degradation"]:
                regressions.append({
                    "metric": "throughput",
                    "baseline": baseline.throughput_ops_per_sec,
                    "current": benchmark.throughput_ops_per_sec,
                    "change": throughput_change,
                    "threshold": self.regression_thresholds["throughput_degradation"]
                })
        
        # Check success rate regression
        success_rate_change = (baseline.success_rate - benchmark.success_rate)
        if success_rate_change > self.regression_thresholds["success_rate_drop"]:
            regressions.append({
                "metric": "success_rate",
                "baseline": baseline.success_rate,
                "current": benchmark.success_rate,
                "change": success_rate_change,
                "threshold": self.regression_thresholds["success_rate_drop"]
            })
        
        if regressions:
            return {
                "benchmark_key": key,
                "operation": benchmark.operation_type,
                "algorithm": benchmark.algorithm,
                "tensor_size": benchmark.tensor_size,
                "num_devices": benchmark.num_devices,
                "regressions": regressions
            }
        
        return None
    
    def _check_improvement(self, benchmark: PerformanceBenchmark) -> Optional[Dict]:
        """Check if benchmark shows improvement compared to baseline."""
        key = self._get_benchmark_key(benchmark)
        
        if key not in self.baseline_data:
            return None
        
        baseline = self.baseline_data[key]
        
        improvements = []
        
        # Check bandwidth improvement
        if baseline.bandwidth_gbps > 0:
            bandwidth_change = (benchmark.bandwidth_gbps - baseline.bandwidth_gbps) / baseline.bandwidth_gbps
            if bandwidth_change > self.regression_thresholds["bandwidth_degradation"]:
                improvements.append({
                    "metric": "bandwidth",
                    "baseline": baseline.bandwidth_gbps,
                    "current": benchmark.bandwidth_gbps,
                    "change": bandwidth_change
                })
        
        # Check latency improvement
        if baseline.latency_us > 0:
            latency_change = (baseline.latency_us - benchmark.latency_us) / baseline.latency_us
            if latency_change > self.regression_thresholds["latency_increase"]:
                improvements.append({
                    "metric": "latency",
                    "baseline": baseline.latency_us,
                    "current": benchmark.latency_us,
                    "change": latency_change
                })
        
        # Check throughput improvement
        if baseline.throughput_ops_per_sec > 0:
            throughput_change = (benchmark.throughput_ops_per_sec - baseline.throughput_ops_per_sec) / baseline.throughput_ops_per_sec
            if throughput_change > self.regression_thresholds["throughput_degradation"]:
                improvements.append({
                    "metric": "throughput",
                    "baseline": baseline.throughput_ops_per_sec,
                    "current": benchmark.throughput_ops_per_sec,
                    "change": throughput_change
                })
        
        if improvements:
            return {
                "benchmark_key": key,
                "operation": benchmark.operation_type,
                "algorithm": benchmark.algorithm,
                "tensor_size": benchmark.tensor_size,
                "num_devices": benchmark.num_devices,
                "improvements": improvements
            }
        
        return None
    
    def _analyze_regression(self, regression_results: Dict) -> bool:
        """Analyze regression results and determine if regression is detected."""
        total_regressions = 0
        total_improvements = 0
        
        for category_name, category_result in regression_results["test_categories"].items():
            total_regressions += len(category_result.get("regressions", []))
            total_improvements += len(category_result.get("improvements", []))
        
        return total_regressions > 0
    
    def _generate_regression_summary(self, regression_results: Dict) -> Dict:
        """Generate regression summary."""
        total_regressions = 0
        total_improvements = 0
        category_stats = {}
        
        for category_name, category_result in regression_results["test_categories"].items():
            regressions = len(category_result.get("regressions", []))
            improvements = len(category_result.get("improvements", []))
            
            total_regressions += regressions
            total_improvements += improvements
            
            category_stats[category_name] = {
                "status": category_result.get("status", "unknown"),
                "regressions": regressions,
                "improvements": improvements,
                "total_benchmarks": len(category_result.get("benchmarks", []))
            }
        
        return {
            "total_regressions": total_regressions,
            "total_improvements": total_improvements,
            "regression_detected": regression_results["regression_detected"],
            "category_stats": category_stats
        }
    
    def _save_regression_results(self, regression_results: Dict):
        """Save regression results to file."""
        results_file = os.path.join(self.output_dir, "regression_results.json")
        
        # Convert benchmarks to serializable format
        serializable_results = regression_results.copy()
        
        for category_name, category_result in serializable_results["test_categories"].items():
            if "benchmarks" in category_result:
                category_result["benchmarks"] = [asdict(b) for b in category_result["benchmarks"]]
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Regression results saved to: {results_file}")
        
        # Save current results as new baseline
        if self.current_results:
            baseline_file = os.path.join(self.output_dir, "baseline.json")
            baseline_data = {
                "timestamp": time.time(),
                "num_devices": self.num_devices,
                "active_devices": self.active_devices,
                "benchmarks": [asdict(b) for b in self.current_results]
            }
            
            with open(baseline_file, 'w') as f:
                json.dump(baseline_data, f, indent=2)
            
            print(f"New baseline saved to: {baseline_file}")
    
    def _generate_regression_plots(self, regression_results: Dict):
        """Generate regression plots."""
        try:
            # Collect all benchmarks
            all_benchmarks = []
            for category_result in regression_results["test_categories"].values():
                all_benchmarks.extend(category_result.get("benchmarks", []))
            
            if not all_benchmarks:
                return
            
            # Create DataFrame for easier plotting
            df_data = []
            for benchmark in all_benchmarks:
                df_data.append({
                    "operation": benchmark.operation_type,
                    "algorithm": benchmark.algorithm,
                    "tensor_size": benchmark.tensor_size,
                    "num_devices": benchmark.num_devices,
                    "bandwidth_gbps": benchmark.bandwidth_gbps,
                    "latency_us": benchmark.latency_us,
                    "throughput_ops_per_sec": benchmark.throughput_ops_per_sec,
                    "memory_usage_mb": benchmark.memory_usage_mb,
                    "success_rate": benchmark.success_rate
                })
            
            df = pd.DataFrame(df_data)
            
            # Plot bandwidth vs tensor size
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            for operation in df["operation"].unique():
                op_data = df[df["operation"] == operation]
                plt.plot(op_data["tensor_size"], op_data["bandwidth_gbps"], 'o-', label=operation)
            plt.xlabel("Tensor Size")
            plt.ylabel("Bandwidth (GB/s)")
            plt.title("Bandwidth vs Tensor Size")
            plt.legend()
            plt.grid(True)
            
            # Plot latency vs tensor size
            plt.subplot(2, 2, 2)
            for operation in df["operation"].unique():
                op_data = df[df["operation"] == operation]
                plt.plot(op_data["tensor_size"], op_data["latency_us"], 'o-', label=operation)
            plt.xlabel("Tensor Size")
            plt.ylabel("Latency (μs)")
            plt.title("Latency vs Tensor Size")
            plt.legend()
            plt.grid(True)
            
            # Plot throughput vs tensor size
            plt.subplot(2, 2, 3)
            for operation in df["operation"].unique():
                op_data = df[df["operation"] == operation]
                plt.plot(op_data["tensor_size"], op_data["throughput_ops_per_sec"], 'o-', label=operation)
            plt.xlabel("Tensor Size")
            plt.ylabel("Throughput (ops/sec)")
            plt.title("Throughput vs Tensor Size")
            plt.legend()
            plt.grid(True)
            
            # Plot memory usage vs tensor size
            plt.subplot(2, 2, 4)
            for operation in df["operation"].unique():
                op_data = df[df["operation"] == operation]
                plt.plot(op_data["tensor_size"], op_data["memory_usage_mb"], 'o-', label=operation)
            plt.xlabel("Tensor Size")
            plt.ylabel("Memory Usage (MB)")
            plt.title("Memory Usage vs Tensor Size")
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plot_file = os.path.join(self.output_dir, "regression_plots.png")
            plt.savefig(plot_file)
            plt.close()
            
            print(f"Regression plots saved to: {plot_file}")
            
        except Exception as e:
            print(f"Failed to generate regression plots: {e}")
    
    def set_regression_thresholds(self, thresholds: Dict[str, float]):
        """Set regression thresholds."""
        self.regression_thresholds.update(thresholds)
    
    def generate_baseline(self, test_config: Dict = None) -> Dict:
        """Generate baseline performance data."""
        print("Generating baseline performance data...")
        
        # Run regression tests to generate baseline
        regression_results = self.run_regression_tests(test_config)
        
        # Save as baseline
        baseline_file = os.path.join(self.output_dir, "baseline.json")
        baseline_data = {
            "timestamp": time.time(),
            "num_devices": self.num_devices,
            "active_devices": self.active_devices,
            "benchmarks": [asdict(b) for b in self.current_results],
            "regression_thresholds": self.regression_thresholds
        }
        
        with open(baseline_file, 'w') as f:
            json.dump(baseline_data, f, indent=2)
        
        print(f"Baseline saved to: {baseline_file}")
        
        return baseline_data


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="P2P Performance Regression Testing Tool")
    parser.add_argument("--devices", type=str, help="Comma-separated list of GPU devices to use")
    parser.add_argument("--baseline", type=str, help="Path to baseline performance file")
    parser.add_argument("--output-dir", type=str, default="./p2p_regression", 
                       help="Output directory for regression test results")
    parser.add_argument("--generate-baseline", action="store_true",
                       help="Generate baseline performance data")
    parser.add_argument("--thresholds", type=str, 
                       help="JSON string with regression thresholds")
    
    args = parser.parse_args()
    
    # Parse devices
    active_devices = None
    if args.devices:
        try:
            active_devices = [int(d.strip()) for d in args.devices.split(",")]
        except ValueError:
            print("Invalid device list")
            return 1
    
    # Parse thresholds
    thresholds = None
    if args.thresholds:
        try:
            thresholds = json.loads(args.thresholds)
        except ValueError:
            print("Invalid thresholds JSON")
            return 1
    
    # Create regression tester
    tester = P2PPerformanceRegressionTester(
        active_devices=active_devices,
        baseline_file=args.baseline,
        output_dir=args.output_dir
    )
    
    # Set thresholds if provided
    if thresholds:
        tester.set_regression_thresholds(thresholds)
    
    try:
        if args.generate_baseline:
            # Generate baseline
            baseline_data = tester.generate_baseline()
            print("Baseline generation completed successfully")
            return 0
        else:
            # Run regression tests
            regression_results = tester.run_regression_tests()
            
            # Print summary
            summary = regression_results["summary"]
            print("\n" + "=" * 50)
            print("REGRESSION TEST SUMMARY")
            print("=" * 50)
            print(f"Regression Detected: {summary['regression_detected']}")
            print(f"Total Regressions: {summary['total_regressions']}")
            print(f"Total Improvements: {summary['total_improvements']}")
            
            print("\nCategory Results:")
            for category, stats in summary["category_stats"].items():
                print(f"  {category}: {stats['status']} "
                      f"(Regressions: {stats['regressions']}, "
                      f"Improvements: {stats['improvements']})")
            
            # Return appropriate exit code
            if summary["regression_detected"]:
                return 1  # Regression detected
            else:
                return 0  # No regression
    
    except Exception as e:
        print(f"Regression testing failed: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())