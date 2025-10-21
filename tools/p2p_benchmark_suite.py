"""
P2P Benchmark Suite for ExLlamaV3

This module provides comprehensive benchmarking tools for P2P GPU communication,
including performance testing, algorithm comparison, and scalability analysis.
"""

import os
import sys
import time
import json
import argparse
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import defaultdict, deque
from dataclasses import dataclass
import numpy as np
import torch

# Add the exllamav3 directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from exllamav3.model.model_tp_p2p import P2PTopology
from exllamav3.util.p2p_monitor import P2PMonitor, initialize_global_monitor
from exllamav3.util.p2p_profiler import P2PProfiler, initialize_global_profiler
from exllamav3.util.p2p_debug import P2PDebugger, initialize_global_debugger


@dataclass
class BenchmarkResult:
    """Result of a benchmark test."""
    test_name: str
    timestamp: float
    device_ids: List[int]
    operation_type: str
    algorithm: Optional[str]
    tensor_shapes: List[Tuple[int, ...]]
    tensor_dtypes: List[str]
    durations_ms: List[float]
    bandwidths_gbps: List[float]
    latencies_us: List[float]
    success_rates: List[float]
    memory_usage_mb: List[float]
    statistics: Dict[str, float]
    metadata: Dict[str, Any]


class P2PBenchmarkSuite:
    """
    Comprehensive P2P benchmark suite.
    
    This class provides extensive benchmarking capabilities for P2P operations,
    including performance testing, algorithm comparison, and scalability analysis.
    """
    
    def __init__(
        self,
        active_devices: List[int],
        output_dir: str = "./p2p_benchmarks",
        enable_monitoring: bool = True,
        enable_profiling: bool = True,
        enable_debugging: bool = False
    ):
        """
        Initialize P2P benchmark suite.
        
        Args:
            active_devices: List of active GPU device IDs
            output_dir: Directory to save benchmark results
            enable_monitoring: Enable performance monitoring
            enable_profiling: Enable profiling
            enable_debugging: Enable debugging
        """
        self.active_devices = active_devices
        self.num_devices = len(active_devices)
        self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize monitoring tools
        self.monitor = None
        self.profiler = None
        self.debugger = None
        
        if enable_monitoring:
            self.monitor = initialize_global_monitor(
                active_devices=active_devices,
                monitoring_level="comprehensive",
                output_dir=os.path.join(output_dir, "monitoring")
            )
        
        if enable_profiling:
            self.profiler = initialize_global_profiler(
                monitor=self.monitor,
                output_dir=os.path.join(output_dir, "profiling")
            )
        
        if enable_debugging:
            self.debugger = initialize_global_debugger(
                monitor=self.monitor,
                debug_level="detailed",
                output_dir=os.path.join(output_dir, "debugging")
            )
        
        # Initialize topology
        self.topology = P2PTopology(active_devices)
        if self.monitor:
            self.monitor.set_topology(self.topology)
        
        # Benchmark results
        self.results: List[BenchmarkResult] = []
        
        # Test configurations
        self.test_configs = {
            "tensor_sizes": [
                (1024,),           # 1K elements
                (1024 * 1024,),    # 1M elements
                (1024 * 1024 * 4,), # 4M elements
                (4096, 4096),      # 16M elements
                (1024, 1024, 1024) # 1B elements
            ],
            "dtypes": [torch.float32, torch.float16, torch.int32],
            "algorithms": ["ring", "binary_tree", "kary_tree", "balanced_tree"],
            "operations": ["broadcast", "all_reduce", "gather", "direct_copy"]
        }
    
    def run_comprehensive_benchmarks(
        self,
        num_iterations: int = 10,
        warmup_iterations: int = 3,
        test_all_configs: bool = False
    ) -> Dict[str, BenchmarkResult]:
        """
        Run comprehensive benchmarks for all P2P operations.
        
        Args:
            num_iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations
            test_all_configs: Test all configuration combinations
            
        Returns:
            Dictionary of benchmark results
        """
        print("Starting comprehensive P2P benchmarks...")
        print(f"Devices: {self.active_devices}")
        print(f"Iterations: {num_iterations} (warmup: {warmup_iterations})")
        
        results = {}
        
        # Test each operation type
        for operation in self.test_configs["operations"]:
            print(f"\nBenchmarking {operation}...")
            
            if operation == "direct_copy":
                # Test direct copy between device pairs
                operation_results = self._benchmark_direct_copy(
                    num_iterations, warmup_iterations
                )
            else:
                # Test collective operations
                operation_results = self._benchmark_collective_operation(
                    operation, num_iterations, warmup_iterations, test_all_configs
                )
            
            results[operation] = operation_results
            self.results.extend(operation_results)
        
        # Generate summary report
        self._generate_summary_report(results)
        
        print("\nComprehensive benchmarks complete!")
        return results
    
    def _benchmark_collective_operation(
        self,
        operation: str,
        num_iterations: int,
        warmup_iterations: int,
        test_all_configs: bool
    ) -> List[BenchmarkResult]:
        """Benchmark a collective operation."""
        results = []
        
        # Test configurations
        tensor_sizes = self.test_configs["tensor_sizes"]
        dtypes = self.test_configs["dtypes"]
        algorithms = self.test_configs["algorithms"] if operation == "all_reduce" else [None]
        
        total_tests = len(tensor_sizes) * len(dtypes) * len(algorithms)
        current_test = 0
        
        for tensor_shape in tensor_sizes:
            for dtype in dtypes:
                for algorithm in algorithms:
                    current_test += 1
                    print(f"  Test {current_test}/{total_tests}: {tensor_shape}, {dtype}, {algorithm or 'default'}")
                    
                    try:
                        result = self._benchmark_single_config(
                            operation, tensor_shape, dtype, algorithm,
                            num_iterations, warmup_iterations
                        )
                        results.append(result)
                    except Exception as e:
                        print(f"    Failed: {e}")
                        if self.debugger:
                            self.debugger.log_error(
                                error_type="benchmark_error",
                                operation_type=operation,
                                device_id=self.active_devices[0],
                                error_message=str(e),
                                context={
                                    "tensor_shape": tensor_shape,
                                    "dtype": str(dtype),
                                    "algorithm": algorithm
                                }
                            )
        
        return results
    
    def _benchmark_direct_copy(
        self,
        num_iterations: int,
        warmup_iterations: int
    ) -> List[BenchmarkResult]:
        """Benchmark direct copy operations between device pairs."""
        results = []
        
        # Test configurations
        tensor_sizes = self.test_configs["tensor_sizes"]
        dtypes = self.test_configs["dtypes"]
        
        # Test all device pairs
        device_pairs = []
        for i in range(len(self.active_devices)):
            for j in range(len(self.active_devices)):
                if i != j:
                    device_pairs.append((self.active_devices[i], self.active_devices[j]))
        
        total_tests = len(tensor_sizes) * len(dtypes) * len(device_pairs)
        current_test = 0
        
        for tensor_shape in tensor_sizes:
            for dtype in dtypes:
                for src_device, dst_device in device_pairs:
                    current_test += 1
                    print(f"  Test {current_test}/{total_tests}: {tensor_shape}, {dtype}, {src_device}->{dst_device}")
                    
                    try:
                        result = self._benchmark_direct_copy_config(
                            src_device, dst_device, tensor_shape, dtype,
                            num_iterations, warmup_iterations
                        )
                        results.append(result)
                    except Exception as e:
                        print(f"    Failed: {e}")
                        if self.debugger:
                            self.debugger.log_error(
                                error_type="benchmark_error",
                                operation_type="direct_copy",
                                device_id=src_device,
                                peer_device=dst_device,
                                error_message=str(e),
                                context={
                                    "tensor_shape": tensor_shape,
                                    "dtype": str(dtype)
                                }
                            )
        
        return results
    
    def _benchmark_single_config(
        self,
        operation: str,
        tensor_shape: Tuple[int, ...],
        dtype: torch.dtype,
        algorithm: Optional[str],
        num_iterations: int,
        warmup_iterations: int
    ) -> BenchmarkResult:
        """Benchmark a single configuration."""
        # Create test tensor
        tensor = torch.randn(tensor_shape, dtype=dtype, device=self.active_devices[0])
        
        # Record test start
        test_name = f"{operation}_{tensor_shape}_{dtype}_{algorithm or 'default'}"
        start_time = time.time()
        
        # Warmup iterations
        for _ in range(warmup_iterations):
            self._execute_operation(operation, tensor, algorithm)
        
        # Benchmark iterations
        durations_ms = []
        bandwidths_gbps = []
        memory_usage_mb = []
        success_rates = []
        
        for i in range(num_iterations):
            iter_start = time.time()
            success = True
            
            try:
                # Record memory before operation
                if self.monitor:
                    memory_before = self.monitor.get_device_metrics(self.active_devices[0]).gpu_memory_used
                
                # Execute operation
                self._execute_operation(operation, tensor, algorithm)
                
                # Record memory after operation
                if self.monitor:
                    memory_after = self.monitor.get_device_metrics(self.active_devices[0]).gpu_memory_used
                    memory_usage_mb.append((memory_after - memory_before) / (1024**2))
                
            except Exception as e:
                success = False
                if self.debugger:
                    self.debugger.log_error(
                        error_type="benchmark_iteration_error",
                        operation_type=operation,
                        device_id=self.active_devices[0],
                        error_message=str(e)
                    )
            
            iter_end = time.time()
            duration_ms = (iter_end - iter_start) * 1000.0
            durations_ms.append(duration_ms)
            
            # Calculate bandwidth
            tensor_size_bytes = tensor.numel() * tensor.element_size()
            if duration_ms > 0:
                bandwidth_gbps = (tensor_size_bytes / (1024**3)) / (duration_ms / 1000.0)
                bandwidths_gbps.append(bandwidth_gbps)
            
            success_rates.append(1.0 if success else 0.0)
        
        # Calculate statistics
        statistics = {
            "avg_duration_ms": np.mean(durations_ms),
            "min_duration_ms": np.min(durations_ms),
            "max_duration_ms": np.max(durations_ms),
            "std_duration_ms": np.std(durations_ms),
            "avg_bandwidth_gbps": np.mean(bandwidths_gbps) if bandwidths_gbps else 0.0,
            "min_bandwidth_gbps": np.min(bandwidths_gbps) if bandwidths_gbps else 0.0,
            "max_bandwidth_gbps": np.max(bandwidths_gbps) if bandwidths_gbps else 0.0,
            "std_bandwidth_gbps": np.std(bandwidths_gbps) if bandwidths_gbps else 0.0,
            "avg_memory_mb": np.mean(memory_usage_mb) if memory_usage_mb else 0.0,
            "max_memory_mb": np.max(memory_usage_mb) if memory_usage_mb else 0.0,
            "success_rate": np.mean(success_rates),
            "total_time": time.time() - start_time
        }
        
        # Create result
        result = BenchmarkResult(
            test_name=test_name,
            timestamp=start_time,
            device_ids=self.active_devices.copy(),
            operation_type=operation,
            algorithm=algorithm,
            tensor_shapes=[tensor_shape],
            tensor_dtypes=[str(dtype)],
            durations_ms=durations_ms,
            bandwidths_gbps=bandwidths_gbps,
            latencies_us=[],  # Not measured in this benchmark
            success_rates=success_rates,
            memory_usage_mb=memory_usage_mb,
            statistics=statistics,
            metadata={
                "num_iterations": num_iterations,
                "warmup_iterations": warmup_iterations,
                "tensor_size_bytes": tensor.numel() * tensor.element_size()
            }
        )
        
        return result
    
    def _benchmark_direct_copy_config(
        self,
        src_device: int,
        dst_device: int,
        tensor_shape: Tuple[int, ...],
        dtype: torch.dtype,
        num_iterations: int,
        warmup_iterations: int
    ) -> BenchmarkResult:
        """Benchmark direct copy configuration."""
        # Create test tensor
        tensor = torch.randn(tensor_shape, dtype=dtype, device=src_device)
        
        # Record test start
        test_name = f"direct_copy_{tensor_shape}_{dtype}_{src_device}->{dst_device}"
        start_time = time.time()
        
        # Warmup iterations
        for _ in range(warmup_iterations):
            self._execute_direct_copy(src_device, dst_device, tensor)
        
        # Benchmark iterations
        durations_ms = []
        bandwidths_gbps = []
        memory_usage_mb = []
        success_rates = []
        
        for i in range(num_iterations):
            iter_start = time.time()
            success = True
            
            try:
                # Record memory before operation
                if self.monitor:
                    memory_before = self.monitor.get_device_metrics(dst_device).gpu_memory_used
                
                # Execute direct copy
                self._execute_direct_copy(src_device, dst_device, tensor)
                
                # Record memory after operation
                if self.monitor:
                    memory_after = self.monitor.get_device_metrics(dst_device).gpu_memory_used
                    memory_usage_mb.append((memory_after - memory_before) / (1024**2))
                
            except Exception as e:
                success = False
                if self.debugger:
                    self.debugger.log_error(
                        error_type="benchmark_iteration_error",
                        operation_type="direct_copy",
                        device_id=src_device,
                        peer_device=dst_device,
                        error_message=str(e)
                    )
            
            iter_end = time.time()
            duration_ms = (iter_end - iter_start) * 1000.0
            durations_ms.append(duration_ms)
            
            # Calculate bandwidth
            tensor_size_bytes = tensor.numel() * tensor.element_size()
            if duration_ms > 0:
                bandwidth_gbps = (tensor_size_bytes / (1024**3)) / (duration_ms / 1000.0)
                bandwidths_gbps.append(bandwidth_gbps)
            
            success_rates.append(1.0 if success else 0.0)
        
        # Calculate statistics
        statistics = {
            "avg_duration_ms": np.mean(durations_ms),
            "min_duration_ms": np.min(durations_ms),
            "max_duration_ms": np.max(durations_ms),
            "std_duration_ms": np.std(durations_ms),
            "avg_bandwidth_gbps": np.mean(bandwidths_gbps) if bandwidths_gbps else 0.0,
            "min_bandwidth_gbps": np.min(bandwidths_gbps) if bandwidths_gbps else 0.0,
            "max_bandwidth_gbps": np.max(bandwidths_gbps) if bandwidths_gbps else 0.0,
            "std_bandwidth_gbps": np.std(bandwidths_gbps) if bandwidths_gbps else 0.0,
            "avg_memory_mb": np.mean(memory_usage_mb) if memory_usage_mb else 0.0,
            "max_memory_mb": np.max(memory_usage_mb) if memory_usage_mb else 0.0,
            "success_rate": np.mean(success_rates),
            "total_time": time.time() - start_time
        }
        
        # Create result
        result = BenchmarkResult(
            test_name=test_name,
            timestamp=start_time,
            device_ids=[src_device, dst_device],
            operation_type="direct_copy",
            algorithm=None,
            tensor_shapes=[tensor_shape],
            tensor_dtypes=[str(dtype)],
            durations_ms=durations_ms,
            bandwidths_gbps=bandwidths_gbps,
            latencies_us=[],  # Not measured in this benchmark
            success_rates=success_rates,
            memory_usage_mb=memory_usage_mb,
            statistics=statistics,
            metadata={
                "num_iterations": num_iterations,
                "warmup_iterations": warmup_iterations,
                "tensor_size_bytes": tensor.numel() * tensor.element_size(),
                "src_device": src_device,
                "dst_device": dst_device
            }
        )
        
        return result
    
    def _execute_operation(
        self,
        operation: str,
        tensor: torch.Tensor,
        algorithm: Optional[str]
    ):
        """Execute a P2P operation (placeholder for actual implementation)."""
        # This would be replaced with actual P2P operation calls
        # For now, simulate with device synchronization
        if self.num_devices > 1:
            for device in self.active_devices:
                if device >= 0:
                    with torch.cuda.device(device):
                        torch.cuda.synchronize()
        
        # Simulate operation time based on tensor size
        tensor_size = tensor.numel() * tensor.element_size()
        base_time = 0.001  # 1ms base time
        size_factor = tensor_size / (1024 * 1024)  # Size in MB
        simulated_time = base_time + size_factor * 0.0001  # 0.1ms per MB
        time.sleep(simulated_time)
    
    def _execute_direct_copy(
        self,
        src_device: int,
        dst_device: int,
        tensor: torch.Tensor
    ):
        """Execute direct copy operation (placeholder for actual implementation)."""
        # This would be replaced with actual P2P direct copy calls
        # For now, simulate with tensor copy
        if src_device != dst_device:
            dst_tensor = torch.empty_like(tensor, device=dst_device)
            dst_tensor.copy_(tensor)
        
        # Simulate copy time
        tensor_size = tensor.numel() * tensor.element_size()
        base_time = 0.0005  # 0.5ms base time
        size_factor = tensor_size / (1024 * 1024)  # Size in MB
        simulated_time = base_time + size_factor * 0.00005  # 0.05ms per MB
        time.sleep(simulated_time)
    
    def run_scalability_benchmark(
        self,
        max_devices: Optional[int] = None,
        tensor_size: int = 1024 * 1024,
        num_iterations: int = 10
    ) -> Dict[str, BenchmarkResult]:
        """
        Run scalability benchmark with varying device counts.
        
        Args:
            max_devices: Maximum number of devices to test
            tensor_size: Size of test tensor
            num_iterations: Number of iterations
            
        Returns:
            Dictionary of scalability results
        """
        if max_devices is None:
            max_devices = self.num_devices
        
        print(f"Running scalability benchmark up to {max_devices} devices...")
        
        results = {}
        
        for device_count in range(2, max_devices + 1):
            print(f"  Testing with {device_count} devices...")
            
            # Use subset of devices
            test_devices = self.active_devices[:device_count]
            
            # Create test tensor
            tensor = torch.randn(tensor_size, dtype=torch.float32, device=test_devices[0])
            
            # Benchmark all_reduce (representative collective operation)
            try:
                result = self._benchmark_single_config(
                    "all_reduce", (tensor_size,), torch.float32, "binary_tree",
                    num_iterations, 3
                )
                results[f"{device_count}_devices"] = result
            except Exception as e:
                print(f"    Failed: {e}")
        
        # Generate scalability report
        self._generate_scalability_report(results)
        
        return results
    
    def run_algorithm_comparison(
        self,
        operation: str = "all_reduce",
        tensor_size: int = 1024 * 1024,
        num_iterations: int = 10
    ) -> Dict[str, BenchmarkResult]:
        """
        Run algorithm comparison for a specific operation.
        
        Args:
            operation: Operation to test
            tensor_size: Size of test tensor
            num_iterations: Number of iterations
            
        Returns:
            Dictionary of algorithm comparison results
        """
        print(f"Running algorithm comparison for {operation}...")
        
        results = {}
        algorithms = self.test_configs["algorithms"] if operation == "all_reduce" else [None]
        
        for algorithm in algorithms:
            print(f"  Testing {algorithm or 'default'} algorithm...")
            
            try:
                result = self._benchmark_single_config(
                    operation, (tensor_size,), torch.float32, algorithm,
                    num_iterations, 3
                )
                results[algorithm or "default"] = result
            except Exception as e:
                print(f"    Failed: {e}")
        
        # Generate comparison report
        self._generate_algorithm_comparison_report(results)
        
        return results
    
    def _generate_summary_report(self, results: Dict[str, List[BenchmarkResult]]):
        """Generate summary benchmark report."""
        report = {
            "timestamp": time.time(),
            "device_count": self.num_devices,
            "devices": self.active_devices,
            "topology_info": self.topology.get_topology_summary(),
            "summary": {},
            "detailed_results": {}
        }
        
        # Generate summary statistics
        all_bandwidths = []
        all_durations = []
        all_success_rates = []
        
        for operation, operation_results in results.items():
            operation_bandwidths = []
            operation_durations = []
            operation_success_rates = []
            
            for result in operation_results:
                if result.bandwidths_gbps:
                    operation_bandwidths.extend(result.bandwidths_gbps)
                    all_bandwidths.extend(result.bandwidths_gbps)
                
                operation_durations.extend(result.durations_ms)
                all_durations.extend(result.durations_ms)
                
                operation_success_rates.extend(result.success_rates)
                all_success_rates.extend(result.success_rates)
            
            report["summary"][operation] = {
                "num_tests": len(operation_results),
                "avg_bandwidth_gbps": np.mean(operation_bandwidths) if operation_bandwidths else 0.0,
                "avg_duration_ms": np.mean(operation_durations) if operation_durations else 0.0,
                "avg_success_rate": np.mean(operation_success_rates) if operation_success_rates else 0.0
            }
        
        # Overall summary
        report["summary"]["overall"] = {
            "total_tests": sum(len(r) for r in results.values()),
            "avg_bandwidth_gbps": np.mean(all_bandwidths) if all_bandwidths else 0.0,
            "avg_duration_ms": np.mean(all_durations) if all_durations else 0.0,
            "avg_success_rate": np.mean(all_success_rates) if all_success_rates else 0.0
        }
        
        # Add detailed results
        for operation, operation_results in results.items():
            report["detailed_results"][operation] = [
                {
                    "test_name": result.test_name,
                    "statistics": result.statistics,
                    "metadata": result.metadata
                }
                for result in operation_results
            ]
        
        # Save report
        report_path = os.path.join(self.output_dir, f"benchmark_summary_{int(time.time())}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Benchmark summary report saved to: {report_path}")
    
    def _generate_scalability_report(self, results: Dict[str, BenchmarkResult]):
        """Generate scalability benchmark report."""
        report = {
            "timestamp": time.time(),
            "test_type": "scalability",
            "results": {}
        }
        
        for key, result in results.items():
            report["results"][key] = {
                "device_count": int(key.split("_")[0]),
                "statistics": result.statistics,
                "metadata": result.metadata
            }
        
        # Save report
        report_path = os.path.join(self.output_dir, f"scalability_report_{int(time.time())}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Scalability report saved to: {report_path}")
    
    def _generate_algorithm_comparison_report(self, results: Dict[str, BenchmarkResult]):
        """Generate algorithm comparison report."""
        report = {
            "timestamp": time.time(),
            "test_type": "algorithm_comparison",
            "results": {}
        }
        
        # Find winner
        best_algorithm = None
        best_bandwidth = 0.0
        
        for algorithm, result in results.items():
            avg_bandwidth = result.statistics.get("avg_bandwidth_gbps", 0.0)
            report["results"][algorithm] = {
                "statistics": result.statistics,
                "metadata": result.metadata
            }
            
            if avg_bandwidth > best_bandwidth:
                best_bandwidth = avg_bandwidth
                best_algorithm = algorithm
        
        report["winner"] = {
            "algorithm": best_algorithm,
            "avg_bandwidth_gbps": best_bandwidth
        }
        
        # Save report
        report_path = os.path.join(self.output_dir, f"algorithm_comparison_{int(time.time())}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Algorithm comparison report saved to: {report_path}")
    
    def export_results(self, filename: Optional[str] = None) -> str:
        """
        Export all benchmark results.
        
        Args:
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to exported file
        """
        if not filename:
            timestamp = int(time.time())
            filename = f"benchmark_results_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Prepare export data
        export_data = {
            "export_timestamp": time.time(),
            "device_count": self.num_devices,
            "devices": self.active_devices,
            "topology_info": self.topology.get_topology_summary(),
            "test_configs": self.test_configs,
            "results": [
                {
                    "test_name": result.test_name,
                    "timestamp": result.timestamp,
                    "device_ids": result.device_ids,
                    "operation_type": result.operation_type,
                    "algorithm": result.algorithm,
                    "tensor_shapes": result.tensor_shapes,
                    "tensor_dtypes": result.tensor_dtypes,
                    "durations_ms": result.durations_ms,
                    "bandwidths_gbps": result.bandwidths_gbps,
                    "latencies_us": result.latencies_us,
                    "success_rates": result.success_rates,
                    "memory_usage_mb": result.memory_usage_mb,
                    "statistics": result.statistics,
                    "metadata": result.metadata
                }
                for result in self.results
            ]
        }
        
        # Write to file
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Benchmark results exported to: {filepath}")
        return filepath


def main():
    """Main function for benchmark suite."""
    parser = argparse.ArgumentParser(description="P2P Benchmark Suite")
    parser.add_argument("--devices", nargs="+", type=int, required=True,
                        help="List of device IDs")
    parser.add_argument("--output-dir", default="./p2p_benchmarks",
                        help="Output directory for results")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Number of warmup iterations")
    parser.add_argument("--comprehensive", action="store_true",
                        help="Run comprehensive benchmarks")
    parser.add_argument("--scalability", action="store_true",
                        help="Run scalability benchmarks")
    parser.add_argument("--algorithm-comparison", action="store_true",
                        help="Run algorithm comparison")
    parser.add_argument("--all", action="store_true",
                        help="Run all benchmark types")
    parser.add_argument("--no-monitoring", action="store_true",
                        help="Disable performance monitoring")
    parser.add_argument("--no-profiling", action="store_true",
                        help="Disable profiling")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debugging")
    
    args = parser.parse_args()
    
    # Create benchmark suite
    suite = P2PBenchmarkSuite(
        active_devices=args.devices,
        output_dir=args.output_dir,
        enable_monitoring=not args.no_monitoring,
        enable_profiling=not args.no_profiling,
        enable_debugging=args.debug
    )
    
    # Run benchmarks
    if args.all or args.comprehensive:
        suite.run_comprehensive_benchmarks(
            num_iterations=args.iterations,
            warmup_iterations=args.warmup
        )
    
    if args.all or args.scalability:
        suite.run_scalability_benchmark(
            num_iterations=args.iterations
        )
    
    if args.all or args.algorithm_comparison:
        suite.run_algorithm_comparison(
            num_iterations=args.iterations
        )
    
    # Export results
    suite.export_results()
    
    print("Benchmark suite complete!")


if __name__ == "__main__":
    main()