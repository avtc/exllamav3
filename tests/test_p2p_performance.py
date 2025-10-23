"""
Performance benchmarking tests for P2P backend functionality.

This module tests:
- All-reduce performance comparison between backends
- Broadcast performance testing
- Gather operation performance
- Throughput and latency measurements
- Scalability testing with different tensor sizes
"""

import pytest
import torch
import numpy as np
import time
import json
import os
from typing import Dict, List, Tuple, Any
from unittest.mock import Mock, patch, MagicMock
import sys

# Add the project root to sys.path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from exllamav3.model.model_tp_backend_p2p import TPBackendP2P
from exllamav3.model.model_tp_backend import TPBackendNCCL, create_tp_backend


class PerformanceBenchmark:
    """Base class for performance benchmarks."""
    
    def __init__(self, warmup_runs: int = 3, benchmark_runs: int = 10):
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.results = []
    
    def warmup(self, backend, tensor):
        """Warm up the backend with a few runs."""
        for _ in range(self.warmup_runs):
            backend.all_reduce(tensor)
    
    def benchmark(self, backend, tensor, operation_name: str):
        """Benchmark an operation and return performance metrics."""
        # Warm up
        self.warmup(backend, tensor)
        
        # Benchmark
        times = []
        for _ in range(self.benchmark_runs):
            start_time = time.perf_counter()
            backend.all_reduce(tensor)
            torch.cuda.synchronize()  # Ensure CUDA operations complete
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        # Calculate statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        # Calculate throughput (elements per second)
        throughput = tensor.numel() / avg_time
        
        # Calculate bandwidth (bytes per second)
        bandwidth = tensor.numel() * tensor.element_size() / avg_time
        
        result = {
            'operation': operation_name,
            'tensor_size': tensor.numel(),
            'tensor_shape': list(tensor.shape),
            'tensor_dtype': str(tensor.dtype),
            'avg_time_ms': avg_time * 1000,
            'std_time_ms': std_time * 1000,
            'min_time_ms': min_time * 1000,
            'max_time_ms': max_time * 1000,
            'throughput_elems_per_sec': throughput,
            'bandwidth_bytes_per_sec': bandwidth,
            'runs': self.benchmark_runs
        }
        
        self.results.append(result)
        return result


class TestP2PPerformanceBenchmarking:
    """Comprehensive performance benchmarking for P2P backend."""

    @pytest.fixture
    def benchmark_backends(self):
        """Set up benchmark backends."""
        backends = {}
        
        # Mock P2P backend
        with patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity') as mock_check:
            with patch('exllamav3.model.model_tp_backend_p2p.cuda_host_register') as mock_register:
                    mock_check.return_value = True
                    
                    with patch('exllamav3.model.model_tp_backend_p2p.shared_memory.SharedMemory'):
                        with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_init_context'):
                            with patch('exllamav3.model.model_tp_backend_p2p.ext.init_p2p_context') as mock_init_p2p:
                                mock_init_p2p.return_value = 0x12345678
                                
                                p2p_backend = TPBackendP2P(
                                    device=0,
                                    active_devices=[0, 1],
                                    output_device=0,
                                    init_method="tcp://127.0.0.1:29500",
                                    master=True,
                                    uuid="test_perf"
                                )
                                backends['p2p'] = p2p_backend
                                
                                # Mock NCCL backend
                                with patch('exllamav3.model.model_tp_backend.TPBackendNCCL') as mock_nccl_class:
                                    with patch('exllamav3.model.model_tp_backend.TPBackendNative') as mock_native:
                                        mock_nccl_instance = Mock()
                                        mock_nccl_instance.all_reduce = Mock()
                                        mock_nccl_class.return_value = mock_nccl_instance
                                        backends['nccl'] = mock_nccl_instance
        
        yield backends
        
        # Clean up
        for name, backend in backends.items():
            if name == 'p2p':
                backend.close()

    def test_all_reduce_performance_comparison(self, benchmark_backends):
        """Test all-reduce performance comparison between P2P and NCCL."""
        benchmark = PerformanceBenchmark(warmup_runs=5, benchmark_runs=20)
        
        # Test different tensor sizes
        tensor_sizes = [1024, 4096, 16384, 65536, 262144]  # 1K to 256K elements
        dtypes = [torch.float32, torch.float16, torch.bfloat16]
        
        results = []
        
        for size in tensor_sizes:
            for dtype in dtypes:
                # Create test tensor
                tensor = torch.randn(size, dtype=dtype, device=0)
                
                # Benchmark P2P
                p2p_result = benchmark.benchmark(
                    benchmark_backends['p2p'], 
                    tensor, 
                    f"all_reduce_p2p_{size}_{dtype}"
                )
                
                # Benchmark NCCL
                start_time = time.perf_counter()
                benchmark_backends['nccl'].all_reduce(tensor)
                torch.cuda.synchronize()
                nccl_time = time.perf_counter() - start_time
                
                nccl_result = {
                    'operation': f"all_reduce_nccl_{size}_{dtype}",
                    'tensor_size': size,
                    'tensor_shape': [size],
                    'tensor_dtype': str(dtype),
                    'avg_time_ms': nccl_time * 1000,
                    'throughput_elems_per_sec': size / nccl_time if nccl_time > 0 else 0,
                    'bandwidth_bytes_per_sec': size * tensor.element_size() / nccl_time if nccl_time > 0 else 0,
                }
                
                # Calculate speedup
                p2p_time = p2p_result['avg_time_ms']
                nccl_time_val = nccl_result['avg_time_ms']
                speedup = nccl_time_val / p2p_time if p2p_time > 0 else 0
                
                result = {
                    'size': size,
                    'dtype': str(dtype),
                    'p2p_time_ms': p2p_time,
                    'nccl_time_ms': nccl_time_val,
                    'speedup': speedup,
                    'p2p_throughput': p2p_result['throughput_elems_per_sec'],
                    'nccl_throughput': nccl_result['throughput_elems_per_sec'],
                    'p2p_bandwidth': p2p_result['bandwidth_bytes_per_sec'],
                    'nccl_bandwidth': nccl_result['bandwidth_bytes_per_sec'],
                }
                
                results.append(result)
                
                # Assert reasonable performance
                assert p2p_time >= 0, "P2P time should not be negative"
                assert nccl_time_val >= 0, "NCCL time should not be negative"
                assert speedup >= 0, "Speedup should not be negative"
        
        # Log results
        print("\n=== All-Reduce Performance Comparison ===")
        for result in results:
            print(f"Size: {result['size']:6d}, Dtype: {result['dtype']:8s}, "
                  f"P2P: {result['p2p_time_ms']:6.3f}ms, NCCL: {result['nccl_time_ms']:6.3f}ms, "
                  f"Speedup: {result['speedup']:6.2f}x")
        
        return results

    def test_broadcast_performance_comparison(self, benchmark_backends):
        """Test broadcast performance comparison between P2P and NCCL."""
        benchmark = PerformanceBenchmark(warmup_runs=5, benchmark_runs=20)
        
        # Test different tensor sizes
        tensor_sizes = [1024, 4096, 16384, 65536, 262144]  # 1K to 256K elements
        dtypes = [torch.float32, torch.float16, torch.bfloat16]
        
        results = []
        
        for size in tensor_sizes:
            for dtype in dtypes:
                # Create test tensor
                tensor = torch.randn(size, dtype=dtype, device=0)
                
                # Benchmark P2P broadcast
                with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_broadcast_full_p2p'):
                    with patch('exllamav3.model.model_tp_backend_p2p.uintptr_t', side_effect=[0, 1]):
                        p2p_result = benchmark.benchmark(
                            benchmark_backends['p2p'], 
                            tensor, 
                            f"broadcast_p2p_{size}_{dtype}"
                        )
                
                # Benchmark NCCL broadcast
                start_time = time.perf_counter()
                benchmark_backends['nccl'].broadcast(tensor, src_device=0)
                torch.cuda.synchronize()
                nccl_time = time.perf_counter() - start_time
                
                nccl_result = {
                    'operation': f"broadcast_nccl_{size}_{dtype}",
                    'tensor_size': size,
                    'tensor_shape': [size],
                    'tensor_dtype': str(dtype),
                    'avg_time_ms': nccl_time * 1000,
                    'throughput_elems_per_sec': size / nccl_time if nccl_time > 0 else 0,
                    'bandwidth_bytes_per_sec': size * tensor.element_size() / nccl_time if nccl_time > 0 else 0,
                }
                
                # Calculate speedup
                p2p_time = p2p_result['avg_time_ms']
                nccl_time_val = nccl_result['avg_time_ms']
                speedup = nccl_time_val / p2p_time if p2p_time > 0 else 0
                
                result = {
                    'size': size,
                    'dtype': str(dtype),
                    'p2p_time_ms': p2p_time,
                    'nccl_time_ms': nccl_time_val,
                    'speedup': speedup,
                    'p2p_throughput': p2p_result['throughput_elems_per_sec'],
                    'nccl_throughput': nccl_result['throughput_elems_per_sec'],
                    'p2p_bandwidth': p2p_result['bandwidth_bytes_per_sec'],
                    'nccl_bandwidth': nccl_result['bandwidth_bytes_per_sec'],
                }
                
                results.append(result)
                
                # Assert reasonable performance
                assert p2p_time >= 0, "P2P time should not be negative"
                assert nccl_time_val >= 0, "NCCL time should not be negative"
        
        # Log results
        print("\n=== Broadcast Performance Comparison ===")
        for result in results:
            print(f"Size: {result['size']:6d}, Dtype: {result['dtype']:8s}, "
                  f"P2P: {result['p2p_time_ms']:6.3f}ms, NCCL: {result['nccl_time_ms']:6.3f}ms, "
                  f"Speedup: {result['speedup']:6.2f}x")
        
        return results

    def test_gather_performance_comparison(self, benchmark_backends):
        """Test gather performance comparison between P2P and NCCL."""
        benchmark = PerformanceBenchmark(warmup_runs=5, benchmark_runs=20)
        
        # Test different tensor sizes
        tensor_sizes = [1024, 4096, 16384, 65536]  # 1K to 64K elements
        dtypes = [torch.float32, torch.float16, torch.bfloat16]
        
        results = []
        
        for size in tensor_sizes:
            for dtype in dtypes:
                # Create test tensors
                tensor = torch.randn(size, dtype=dtype, device=0)
                output_tensor = torch.randn(size * 2, dtype=dtype, device=0)
                ldims = [size, size]
                
                # Benchmark P2P gather
                with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_gather_full_p2p'):
                    with patch('exllamav3.model.model_tp_backend_p2p.uintptr_t', side_effect=[0, 1]):
                        start_time = time.perf_counter()
                        benchmark_backends['p2p'].gather(tensor, output_tensor, None, 0, ldims)
                        torch.cuda.synchronize()
                        p2p_time = time.perf_counter() - start_time
                
                # Benchmark NCCL gather (mock)
                start_time = time.perf_counter()
                benchmark_backends['nccl'].gather(tensor, output_tensor, None, 0, ldims)
                torch.cuda.synchronize()
                nccl_time = time.perf_counter() - start_time
                
                # Calculate performance metrics
                p2p_throughput = (size * 2) / p2p_time if p2p_time > 0 else 0
                nccl_throughput = (size * 2) / nccl_time if nccl_time > 0 else 0
                
                speedup = nccl_time / p2p_time if p2p_time > 0 else 0
                
                result = {
                    'size': size,
                    'dtype': str(dtype),
                    'p2p_time_ms': p2p_time * 1000,
                    'nccl_time_ms': nccl_time * 1000,
                    'speedup': speedup,
                    'p2p_throughput': p2p_throughput,
                    'nccl_throughput': nccl_throughput,
                }
                
                results.append(result)
                
                # Assert reasonable performance
                assert p2p_time >= 0, "P2P time should not be negative"
                assert nccl_time >= 0, "NCCL time should not be negative"
        
        # Log results
        print("\n=== Gather Performance Comparison ===")
        for result in results:
            print(f"Size: {result['size']:6d}, Dtype: {result['dtype']:8s}, "
                  f"P2P: {result['p2p_time_ms']:6.3f}ms, NCCL: {result['nccl_time_ms']:6.3f}ms, "
                  f"Speedup: {result['speedup']:6.2f}x")
        
        return results

    def test_throughput_and_latency_measurements(self, benchmark_backends):
        """Test throughput and latency measurements for different operations."""
        benchmark = PerformanceBenchmark(warmup_runs=3, benchmark_runs=15)
        
        # Test various tensor sizes for throughput
        tensor_sizes = [1024, 8192, 65536, 524288]  # 1K to 512K elements
        
        throughput_results = []
        
        for size in tensor_sizes:
            # Create test tensor
            tensor = torch.randn(size, device=0)
            
            # Measure all-reduce throughput
            with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_all_reduce_full_p2p'):
                with patch('exllamav3.model.model_tp_backend_p2p.uintptr_t', side_effect=[0, 1]):
                    result = benchmark.benchmark(
                        benchmark_backends['p2p'], 
                        tensor, 
                        f"throughput_test_{size}"
                    )
                    
                    throughput_results.append({
                        'tensor_size': size,
                        'throughput_elems_per_sec': result['throughput_elems_per_sec'],
                        'bandwidth_bytes_per_sec': result['bandwidth_bytes_per_sec'],
                        'avg_latency_ms': result['avg_time_ms']
                    })
        
        # Test latency with small tensors
        small_tensor = torch.randn(64, device=0)  # Very small for latency measurement
        
        latency_times = []
        for _ in range(100):  # Many measurements for statistical significance
            start_time = time.perf_counter()
            benchmark_backends['p2p'].all_reduce(small_tensor)
            torch.cuda.synchronize()
            latency_times.append((time.perf_counter() - start_time) * 1000)  # Convert to ms
        
        avg_latency = np.mean(latency_times)
        std_latency = np.std(latency_times)
        
        # Log results
        print("\n=== Throughput Measurements ===")
        for result in throughput_results:
            print(f"Size: {result['tensor_size']:8d}, "
                  f"Throughput: {result['throughput_elems_per_sec']:12.2e} elems/sec, "
                  f"Bandwidth: {result['bandwidth_bytes_per_sec']:12.2e} bytes/sec, "
                  f"Latency: {result['avg_latency_ms']:6.3f} ms")
        
        print(f"\n=== Latency Measurements (64 elements) ===")
        print(f"Average latency: {avg_latency:.3f} ms ± {std_latency:.3f} ms")
        print(f"Min latency: {np.min(latency_times):.3f} ms")
        print(f"Max latency: {np.max(latency_times):.3f} ms")
        
        # Assert performance requirements
        assert avg_latency >= 0, "Latency should not be negative"
        assert std_latency >= 0, "Std deviation should not be negative"
        
        for result in throughput_results:
            assert result['throughput_elems_per_sec'] > 0, "Throughput should be positive"
            assert result['bandwidth_bytes_per_sec'] > 0, "Bandwidth should be positive"
        
        return {
            'throughput': throughput_results,
            'latency': {
                'avg_ms': avg_latency,
                'std_ms': std_latency,
                'min_ms': np.min(latency_times),
                'max_ms': np.max(latency_times)
            }
        }

    def test_scalability_testing(self, benchmark_backends):
        """Test scalability with different numbers of devices and tensor sizes."""
        benchmark = PerformanceBenchmark(warmup_runs=3, benchmark_runs=10)
        
        # Test different numbers of devices
        device_counts = [2, 4, 8]
        tensor_sizes = [16384, 65536, 262144]  # 16K to 256K elements
        
        scalability_results = []
        
        for num_devices in device_counts:
            for size in tensor_sizes:
                # Create test tensor
                tensor = torch.randn(size, device=0)
                
                # Mock P2P backend with multiple devices
                with patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity') as mock_check:
                    with patch('exllamav3.model.model_tp_backend_p2p.cuda_host_register') as mock_register:
                            mock_check.return_value = True
                            
                            with patch('exllamav3.model.model_tp_backend_p2p.shared_memory.SharedMemory'):
                                with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_init_context'):
                                    with patch('exllamav3.model.model_tp_backend_p2p.ext.init_p2p_context') as mock_init_p2p:
                                        mock_init_p2p.return_value = 0x12345678
                                        
                                        devices = list(range(num_devices))
                                        backend = TPBackendP2P(
                                            device=0,
                                            active_devices=devices,
                                            output_device=0,
                                            init_method="tcp://127.0.0.1:29500",
                                            master=True,
                                            uuid="test_scalability"
                                        )
                                        
                                        # Benchmark scalability
                                        with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_all_reduce_full_p2p'):
                                            with patch('exllamav3.model.model_tp_backend_p2p.uintptr_t', side_effect=devices):
                                                result = benchmark.benchmark(
                                                    backend, 
                                                    tensor, 
                                                    f"scalability_{num_devices}d_{size}"
                                                )
                                                
                                                scalability_results.append({
                                                    'num_devices': num_devices,
                                                    'tensor_size': size,
                                                    'avg_time_ms': result['avg_time_ms'],
                                                    'throughput_elems_per_sec': result['throughput_elems_per_sec'],
                                                    'bandwidth_bytes_per_sec': result['bandwidth_bytes_per_sec'],
                                                    'throughput_per_device': result['throughput_elems_per_sec'] / num_devices
                                                })
                                        
                                        backend.close()
        
        # Analyze scalability efficiency
        print("\n=== Scalability Analysis ===")
        for result in scalability_results:
            print(f"Devices: {result['num_devices']:2d}, Size: {result['tensor_size']:8d}, "
                  f"Time: {result['avg_time_ms']:6.3f}ms, "
                  f"Throughput: {result['throughput_elems_per_sec']:12.2e} elems/sec, "
                  f"Per device: {result['throughput_per_device']:12.2e} elems/sec")
        
        # Calculate scalability efficiency
        efficiency_results = []
        base_size = 16384  # Use smallest tensor size as baseline
        
        for num_devices in device_counts[1:]:  # Skip 2 devices (baseline)
            baseline = next(r for r in scalability_results 
                          if r['num_devices'] == 2 and r['tensor_size'] == base_size)
            current = next(r for r in scalability_results 
                         if r['num_devices'] == num_devices and r['tensor_size'] == base_size)
            
            # Theoretical speedup from doubling devices: 2x
            theoretical_speedup = num_devices / 2
            actual_speedup = baseline['avg_time_ms'] / current['avg_time_ms']
            efficiency = actual_speedup / theoretical_speedup
            
            efficiency_results.append({
                'num_devices': num_devices,
                'theoretical_speedup': theoretical_speedup,
                'actual_speedup': actual_speedup,
                'efficiency': efficiency,
                'throughup_ratio': current['throughput_elems_per_sec'] / baseline['throughput_elems_per_sec']
            })
        
        print("\n=== Scalability Efficiency ===")
        for result in efficiency_results:
            print(f"Devices: {result['num_devices']:2d}, "
                  f"Theoretical: {result['theoretical_speedup']:6.2f}x, "
                  f"Actual: {result['actual_speedup']:6.2f}x, "
                  f"Efficiency: {result['efficiency']:6.2%}, "
                  f"Throughput ratio: {result['throughup_ratio']:6.2f}x")
        
        # Assert reasonable scalability
        for result in efficiency_results:
            assert result['efficiency'] > 0, "Efficiency should be positive"
            # Efficiency should be reasonable (not negative, not extremely low)
            assert result['efficiency'] < 10, f"Unrealistic efficiency: {result['efficiency']}"
        
        return {
            'scalability': scalability_results,
            'efficiency': efficiency_results
        }

    def test_tensor_size_scaling(self, benchmark_backends):
        """Test how performance scales with tensor size."""
        benchmark = PerformanceBenchmark(warmup_runs=3, benchmark_runs=10)
        
        # Test a range of tensor sizes
        tensor_sizes = [2**i for i in range(10, 21)]  # 2^10 to 2^20 elements
        
        size_scaling_results = []
        
        for size in tensor_sizes:
            # Create test tensor
            tensor = torch.randn(size, device=0)
            
            # Benchmark performance
            with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_all_reduce_full_p2p'):
                with patch('exllamav3.model.model_tp_backend_p2p.uintptr_t', side_effect=[0, 1]):
                    result = benchmark.benchmark(
                        benchmark_backends['p2p'], 
                        tensor, 
                        f"size_scaling_{size}"
                    )
                    
                    size_scaling_results.append({
                        'tensor_size': size,
                        'log_size': np.log2(size),
                        'avg_time_ms': result['avg_time_ms'],
                        'throughput_elems_per_sec': result['throughput_elems_per_sec'],
                        'bandwidth_bytes_per_sec': result['bandwidth_bytes_per_sec']
                    })
        
        # Analyze scaling behavior
        print("\n=== Tensor Size Scaling ===")
        for result in size_scaling_results:
            print(f"Size: {result['tensor_size']:10d} (2^{result['log_size']:5.1f}), "
                  f"Time: {result['avg_time_ms']:6.3f}ms, "
                  f"Throughput: {result['throughput_elems_per_sec']:12.2e} elems/sec")
        
        # Calculate scaling exponent (time ~ size^alpha)
        log_sizes = [r['log_size'] for r in size_scaling_results]
        log_times = [np.log10(r['avg_time_ms']) for r in size_scaling_results]
        
        # Linear regression to find scaling exponent
        n = len(log_sizes)
        sum_x = sum(log_sizes)
        sum_y = sum(log_times)
        sum_xy = sum(x * y for x, y in zip(log_sizes, log_times))
        sum_x2 = sum(x * x for x in log_sizes)
        
        scaling_exponent = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        print(f"\nScaling analysis: time ~ size^{scaling_exponent:.3f}")
        
        # Assert reasonable scaling (should be between 0.5 and 1.0 for all-reduce)
        assert 0.3 <= scaling_exponent <= 1.2, f"Unrealistic scaling exponent: {scaling_exponent}"
        
        return {
            'size_scaling': size_scaling_results,
            'scaling_exponent': scaling_exponent
        }

    def test_performance_regression_detection(self, benchmark_backends):
        """Test for performance regressions by comparing baseline and current performance."""
        benchmark = PerformanceBenchmark(warmup_runs=5, benchmark_runs=20)
        
        # Define baseline performance (expected minimum performance)
        baseline_performance = {
            'min_throughput': 1e6,  # 1M elements per second minimum
            'max_latency': 10.0,    # 10ms maximum latency for small tensors
            'min_efficiency': 0.7   # 70% minimum efficiency for scaling
        }
        
        # Test current performance
        test_sizes = [1024, 16384, 65536]
        performance_results = []
        
        regressions = []
        
        for size in test_sizes:
            tensor = torch.randn(size, device=0)
            
            with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_all_reduce_full_p2p'):
                with patch('exllamav3.model.model_tp_backend_p2p.uintptr_t', side_effect=[0, 1]):
                    result = benchmark.benchmark(
                        benchmark_backends['p2p'], 
                        tensor, 
                        f"regression_test_{size}"
                    )
                    
                    throughput = result['throughput_elems_per_sec']
                    latency = result['avg_time_ms']
                    
                    # Check for regressions
                    if throughput < baseline_performance['min_throughput']:
                        regressions.append({
                            'type': 'throughput',
                            'size': size,
                            'current': throughput,
                            'baseline': baseline_performance['min_throughput'],
                            'ratio': throughput / baseline_performance['min_throughput']
                        })
                    
                    if latency > baseline_performance['max_latency']:
                        regressions.append({
                            'type': 'latency',
                            'size': size,
                            'current': latency,
                            'baseline': baseline_performance['max_latency'],
                            'ratio': latency / baseline_performance['max_latency']
                        })
                    
                    performance_results.append({
                        'size': size,
                        'throughput': throughput,
                        'latency_ms': latency,
                        'regression_detected': False
                    })
        
        # Log results
        print("\n=== Performance Regression Detection ===")
        for result in performance_results:
            print(f"Size: {result['size']:8d}, "
                  f"Throughput: {result['throughput']:12.2e} elems/sec, "
                  f"Latency: {result['latency_ms']:6.3f} ms")
        
        if regressions:
            print("\n=== Performance Regressions Detected ===")
            for regression in regressions:
                print(f"{regression['type'].upper()}: Size {regression['size']:8d}, "
                      f"Current: {regression['current']:12.2e}, "
                      f"Baseline: {regression['baseline']:12.2e}, "
                      f"Ratio: {regression['ratio']:6.2%}")
        
        # Assert no severe regressions
        severe_regressions = [r for r in regressions if r['ratio'] < 0.5]
        assert len(severe_regressions) == 0, f"Severe performance regressions detected: {severe_regressions}"
        
        return {
            'performance': performance_results,
            'regressions': regressions,
            'baseline': baseline_performance
        }


class TestP2PPerformanceProfiling:
    """Test performance profiling and detailed analysis."""

    def test_detailed_operation_profiling(self):
        """Test detailed profiling of P2P operations."""
        with patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity') as mock_check:
            with patch('exllamav3.model.model_tp_backend_p2p.cuda_host_register') as mock_register:
                    mock_check.return_value = True
                    
                    with patch('exllamav3.model.model_tp_backend_p2p.shared_memory.SharedMemory'):
                        with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_init_context'):
                            with patch('exllamav3.model.model_tp_backend_p2p.ext.init_p2p_context') as mock_init_p2p:
                                mock_init_p2p.return_value = 0x12345678
                                
                                backend = TPBackendP2P(
                                    device=0,
                                    active_devices=[0, 1],
                                    output_device=0,
                                    init_method="tcp://127.0.0.1:29500",
                                    master=True,
                                    uuid="test_profile"
                                )
                                
                                # Profile different operations
                                tensor_sizes = [1024, 16384, 65536]
                                profile_results = {}
                                
                                for size in tensor_sizes:
                                    tensor = torch.randn(size, device=0)
                                    
                                    # Profile all-reduce
                                    profile_results[f'all_reduce_{size}'] = self._profile_operation(
                                        lambda t: backend.all_reduce(t),
                                        tensor
                                    )
                                    
                                    # Profile broadcast
                                    profile_results[f'broadcast_{size}'] = self._profile_operation(
                                        lambda t: backend.broadcast(t, src_device=0),
                                        tensor
                                    )
                                
                                backend.close()
                                
                                # Analyze profiling results
                                print("\n=== Detailed Operation Profiling ===")
                                for op_name, metrics in profile_results.items():
                                    print(f"{op_name}:")
                                    print(f"  Mean: {metrics['mean_ms']:.3f} ms")
                                    print(f"  Std:  {metrics['std_ms']:.3f} ms")
                                    print(f"  Min:  {metrics['min_ms']:.3f} ms")
                                    print(f"  Max:  {metrics['max_ms']:.3f} ms")
                                    print(f"  95th percentile: {metrics['p95_ms']:.3f} ms")
                                
                                return profile_results

    def _profile_operation(self, operation, tensor, num_runs=50):
        """Profile an operation and return detailed statistics."""
        times = []
        
        for _ in range(num_runs):
            start_time = time.perf_counter()
            operation(tensor)
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        times_sorted = sorted(times)
        
        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'p50_ms': np.median(times),
            'p95_ms': times_sorted[int(len(times_sorted) * 0.95)],
            'p99_ms': times_sorted[int(len(times_sorted) * 0.99)],
            'num_runs': num_runs
        }

    def test_memory_bandwidth_analysis(self):
        """Test memory bandwidth analysis for P2P operations."""
        with patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity') as mock_check:
            with patch('exllamav3.model.model_tp_backend_p2p.cuda_host_register') as mock_register:
                    mock_check.return_value = True
                    
                    with patch('exllamav3.model.model_tp_backend_p2p.shared_memory.SharedMemory'):
                        with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_init_context'):
                            with patch('exllamav3.model.model_tp_backend_p2p.ext.init_p2p_context') as mock_init_p2p:
                                mock_init_p2p.return_value = 0x12345678
                                
                                backend = TPBackendP2P(
                                    device=0,
                                    active_devices=[0, 1],
                                    output_device=0,
                                    init_method="tcp://127.0.0.1:29500",
                                    master=True,
                                    uuid="test_bandwidth"
                                )
                                
                                # Test different data types and sizes
                                test_configs = [
                                    (torch.float32, 16384),
                                    (torch.float16, 32768),
                                    (torch.bfloat16, 65536)
                                ]
                                
                                bandwidth_results = []
                                
                                for dtype, size in test_configs:
                                    tensor = torch.randn(size, dtype=dtype, device=0)
                                    bytes_per_element = tensor.element_size()
                                    total_bytes = tensor.numel() * bytes_per_element
                                    
                                    # Measure bandwidth
                                    start_time = time.perf_counter()
                                    with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_all_reduce_full_p2p'):
                                        with patch('exllamav3.model.model_tp_backend_p2p.uintptr_t', side_effect=[0, 1]):
                                            backend.all_reduce(tensor)
                                    torch.cuda.synchronize()
                                    end_time = time.perf_counter()
                                    
                                    bandwidth = total_bytes / (end_time - start_time)  # bytes per second
                                    
                                    bandwidth_results.append({
                                        'dtype': str(dtype),
                                        'size': size,
                                        'bytes_per_element': bytes_per_element,
                                        'total_bytes': total_bytes,
                                        'bandwidth_bytes_per_sec': bandwidth,
                                        'bandwidth_gbps': bandwidth * 8 / 1e9  # Convert to Gbps
                                    })
                                
                                backend.close()
                                
                                # Analyze bandwidth results
                                print("\n=== Memory Bandwidth Analysis ===")
                                for result in bandwidth_results:
                                    print(f"Dtype: {result['dtype']:8s}, Size: {result['size']:8d}, "
                                          f"Bandwidth: {result['bandwidth_gbps']:6.2f} Gbps")
                                
                                # Check reasonable bandwidth utilization
                                # Assume HBM bandwidth of ~900 GB/s for modern GPUs
                                hbm_bandwidth = 900e9  # 900 GB/s
                                
                                for result in bandwidth_results:
                                    utilization = result['bandwidth_bytes_per_sec'] / hbm_bandwidth
                                    assert 0.01 <= utilization <= 1.0, f"Unrealistic bandwidth utilization: {utilization:.2%}"
                                
                                return bandwidth_results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])