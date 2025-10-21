#!/usr/bin/env python3
"""
Benchmark script for shared memory optimizations in ExLlamaV3.

This script tests the performance improvements from the optimized shared memory management,
including adaptive buffer sizing, zero-copy operations, memory pooling, and batched processing.
"""

import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import gc
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exllamav3.model.model_tp_shared import SMProducer, SMConsumer, _memory_pool
from exllamav3.model.model_tp_cuda import OptimizedMemoryManager, P2PMemoryUtils


class SharedMemoryBenchmark:
    """Benchmark suite for shared memory optimizations."""
    
    def __init__(self, device: int = 0):
        self.device = device
        self.results = {}
        torch.cuda.set_device(device)
        
    def run_all_benchmarks(self) -> Dict[str, Dict]:
        """Run all benchmarks and collect results."""
        print("Running Shared Memory Optimization Benchmarks...")
        print("=" * 60)
        
        # Benchmark 1: Adaptive Buffer Sizing
        print("\n1. Testing Adaptive Buffer Sizing...")
        self.results['adaptive_buffer'] = self.benchmark_adaptive_buffer_sizing()
        
        # Benchmark 2: Zero-Copy Operations
        print("\n2. Testing Zero-Copy Operations...")
        self.results['zero_copy'] = self.benchmark_zero_copy_operations()
        
        # Benchmark 3: Memory Pool Performance
        print("\n3. Testing Memory Pool Performance...")
        self.results['memory_pool'] = self.benchmark_memory_pool_performance()
        
        # Benchmark 4: Batched Processing
        print("\n4. Testing Batched Processing...")
        self.results['batched_processing'] = self.benchmark_batched_processing()
        
        # Benchmark 5: Tensor Serialization
        print("\n5. Testing Tensor Serialization...")
        self.results['tensor_serialization'] = self.benchmark_tensor_serialization()
        
        # Benchmark 6: Memory Efficiency
        print("\n6. Testing Memory Efficiency...")
        self.results['memory_efficiency'] = self.benchmark_memory_efficiency()
        
        # Generate summary report
        self.generate_summary_report()
        
        return self.results
    
    def benchmark_adaptive_buffer_sizing(self) -> Dict:
        """Benchmark adaptive buffer sizing performance."""
        results = {
            'fixed_buffer_times': [],
            'adaptive_buffer_times': [],
            'memory_usage_fixed': [],
            'memory_usage_adaptive': [],
            'tensor_sizes': []
        }
        
        tensor_sizes = [
            (1024, 1024),      # 1MB
            (2048, 2048),      # 4MB
            (4096, 4096),      # 16MB
            (8192, 8192),      # 64MB
            (16384, 16384),    # 256MB
        ]
        
        for size in tensor_sizes:
            results['tensor_sizes'].append(size[0] * size[1])
            
            # Test fixed buffer size
            producer_fixed = SMProducer(buffer_size=64*1024*1024, adaptive_sizing=False)
            consumer_fixed = SMConsumer(producer_fixed, device=self.device, pin_memory=True)
            
            tensor = torch.randn(size, device=self.device, dtype=torch.float16)
            
            start_time = time.time()
            for _ in range(10):
                imp = producer_fixed.send(tensor)
                received = consumer_fixed.recv(imp, cuda=True)
            end_time = time.time()
            
            results['fixed_buffer_times'].append((end_time - start_time) / 10)
            results['memory_usage_fixed'].append(producer_fixed.buffer_size)
            
            producer_fixed.close()
            consumer_fixed.close()
            
            # Test adaptive buffer size
            producer_adaptive = SMProducer(buffer_size=32*1024*1024, adaptive_sizing=True)
            consumer_adaptive = SMConsumer(producer_adaptive, device=self.device, pin_memory=True)
            
            start_time = time.time()
            for _ in range(10):
                imp = producer_adaptive.send(tensor)
                received = consumer_adaptive.recv(imp, cuda=True)
            end_time = time.time()
            
            results['adaptive_buffer_times'].append((end_time - start_time) / 10)
            results['memory_usage_adaptive'].append(producer_adaptive.buffer_size)
            
            producer_adaptive.close()
            consumer_adaptive.close()
            
            print(f"  Size {size[0]}x{size[1]}: Fixed={results['fixed_buffer_times'][-1]*1000:.2f}ms, "
                  f"Adaptive={results['adaptive_buffer_times'][-1]*1000:.2f}ms")
        
        return results
    
    def benchmark_zero_copy_operations(self) -> Dict:
        """Benchmark zero-copy operations performance."""
        results = {
            'zero_copy_times': [],
            'copy_times': [],
            'tensor_sizes': []
        }
        
        tensor_sizes = [
            (512, 512),
            (1024, 1024),
            (2048, 2048),
            (4096, 4096),
        ]
        
        for size in tensor_sizes:
            results['tensor_sizes'].append(size[0] * size[1])
            
            producer = SMProducer(buffer_size=128*1024*1024)
            consumer = SMConsumer(producer, device=self.device, pin_memory=True, enable_zero_copy=True)
            
            # Test zero-copy path
            tensor = torch.randn(size, device=self.device, dtype=torch.float16)
            
            start_time = time.time()
            for _ in range(20):
                imp = producer.send(tensor)
                received = consumer.recv(imp, cuda=True)
            end_time = time.time()
            
            results['zero_copy_times'].append((end_time - start_time) / 20)
            
            # Test regular copy path
            consumer_no_zero_copy = SMConsumer(producer, device=self.device, pin_memory=True, enable_zero_copy=False)
            
            start_time = time.time()
            for _ in range(20):
                imp = producer.send(tensor)
                received = consumer_no_zero_copy.recv(imp, cuda=True)
            end_time = time.time()
            
            results['copy_times'].append((end_time - start_time) / 20)
            
            producer.close()
            consumer.close()
            consumer_no_zero_copy.close()
            
            print(f"  Size {size[0]}x{size[1]}: Zero-copy={results['zero_copy_times'][-1]*1000:.2f}ms, "
                  f"Copy={results['copy_times'][-1]*1000:.2f}ms")
        
        return results
    
    def benchmark_memory_pool_performance(self) -> Dict:
        """Benchmark memory pool performance."""
        results = {
            'pool_times': [],
            'no_pool_times': [],
            'tensor_sizes': []
        }
        
        # Test memory manager
        memory_manager = OptimizedMemoryManager()
        
        tensor_sizes = [
            (256, 256),
            (512, 512),
            (1024, 1024),
            (2048, 2048),
        ]
        
        for size in tensor_sizes:
            results['tensor_sizes'].append(size[0] * size[1])
            
            # Test with memory pool
            start_time = time.time()
            for _ in range(50):
                buffer_ptr = memory_manager.get_pinned_buffer(size[0] * size[1] * 2)  # 2 bytes per element
                memory_manager.return_pinned_buffer(buffer_ptr, size[0] * size[1] * 2)
            end_time = time.time()
            
            results['pool_times'].append((end_time - start_time) / 50)
            
            # Test without memory pool (direct allocation)
            start_time = time.time()
            for _ in range(50):
                buffer = torch.empty(size[0] * size[1], dtype=torch.float16, pin_memory=True)
                del buffer
            end_time = time.time()
            
            results['no_pool_times'].append((end_time - start_time) / 50)
            
            print(f"  Size {size[0]}x{size[1]}: Pool={results['pool_times'][-1]*1000:.2f}ms, "
                  f"No-pool={results['no_pool_times'][-1]*1000:.2f}ms")
        
        # Print memory manager stats
        stats = memory_manager.get_stats()
        print(f"  Memory Manager Stats: {stats}")
        
        return results
    
    def benchmark_batched_processing(self) -> Dict:
        """Benchmark batched processing performance."""
        results = {
            'batched_times': [],
            'individual_times': [],
            'batch_sizes': []
        }
        
        batch_sizes = [1, 5, 10, 20, 50]
        tensor_size = (1024, 1024)
        
        for batch_size in batch_sizes:
            results['batch_sizes'].append(batch_size)
            
            producer = SMProducer(buffer_size=256*1024*1024)
            consumer = SMConsumer(producer, device=self.device, pin_memory=True)
            
            # Create batch of tensors
            tensors = [torch.randn(tensor_size, device=self.device, dtype=torch.float16) for _ in range(batch_size)]
            
            # Test batched processing
            start_time = time.time()
            for _ in range(10):
                imps = [producer.send(tensor) for tensor in tensors]
                received = consumer.recv_batch(imps, cuda=True)
            end_time = time.time()
            
            results['batched_times'].append((end_time - start_time) / 10)
            
            # Test individual processing
            start_time = time.time()
            for _ in range(10):
                for tensor in tensors:
                    imp = producer.send(tensor)
                    received = consumer.recv(imp, cuda=True)
            end_time = time.time()
            
            results['individual_times'].append((end_time - start_time) / 10)
            
            producer.close()
            consumer.close()
            
            print(f"  Batch size {batch_size}: Batched={results['batched_times'][-1]*1000:.2f}ms, "
                  f"Individual={results['individual_times'][-1]*1000:.2f}ms")
        
        return results
    
    def benchmark_tensor_serialization(self) -> Dict:
        """Benchmark tensor serialization performance."""
        results = {
            'optimized_times': [],
            'standard_times': [],
            'tensor_sizes': []
        }
        
        tensor_sizes = [
            (512, 512),
            (1024, 1024),
            (2048, 2048),
            (4096, 4096),
        ]
        
        for size in tensor_sizes:
            results['tensor_sizes'].append(size[0] * size[1])
            
            producer = SMProducer(buffer_size=128*1024*1024)
            consumer = SMConsumer(producer, device=self.device, pin_memory=True, cache_tensors=True)
            
            tensor = torch.randn(size, device=self.device, dtype=torch.float16)
            
            # Test optimized serialization (with caching)
            start_time = time.time()
            for _ in range(30):
                imp = producer.send(tensor)
                received = consumer.recv(imp, cuda=True)
            end_time = time.time()
            
            results['optimized_times'].append((end_time - start_time) / 30)
            
            # Test standard serialization (without caching)
            consumer_no_cache = SMConsumer(producer, device=self.device, pin_memory=True, cache_tensors=False)
            
            start_time = time.time()
            for _ in range(30):
                imp = producer.send(tensor)
                received = consumer_no_cache.recv(imp, cuda=True)
            end_time = time.time()
            
            results['standard_times'].append((end_time - start_time) / 30)
            
            producer.close()
            consumer.close()
            consumer_no_cache.close()
            
            print(f"  Size {size[0]}x{size[1]}: Optimized={results['optimized_times'][-1]*1000:.2f}ms, "
                  f"Standard={results['standard_times'][-1]*1000:.2f}ms")
        
        return results
    
    def benchmark_memory_efficiency(self) -> Dict:
        """Benchmark memory efficiency improvements."""
        results = {
            'peak_usage_fixed': [],
            'peak_usage_adaptive': [],
            'fragmentation_fixed': [],
            'fragmentation_adaptive': [],
            'tensor_sizes': []
        }
        
        tensor_sizes = [
            (256, 256),
            (512, 512),
            (1024, 1024),
            (2048, 2048),
        ]
        
        for size in tensor_sizes:
            results['tensor_sizes'].append(size[0] * size[1])
            
            # Test fixed buffer
            producer_fixed = SMProducer(buffer_size=128*1024*1024, adaptive_sizing=False)
            
            tensors = [torch.randn(size, device=self.device, dtype=torch.float16) for _ in range(20)]
            
            peak_usage = 0
            for tensor in tensors:
                imp = producer_fixed.send(tensor)
                peak_usage = max(peak_usage, producer_fixed.next_offset)
            
            results['peak_usage_fixed'].append(peak_usage)
            results['fragmentation_fixed'].append(producer_fixed.peak_usage - peak_usage)
            
            producer_fixed.close()
            
            # Test adaptive buffer
            producer_adaptive = SMProducer(buffer_size=32*1024*1024, adaptive_sizing=True)
            
            peak_usage = 0
            for tensor in tensors:
                imp = producer_adaptive.send(tensor)
                peak_usage = max(peak_usage, producer_adaptive.next_offset)
            
            results['peak_usage_adaptive'].append(peak_usage)
            results['fragmentation_adaptive'].append(producer_adaptive.peak_usage - peak_usage)
            
            producer_adaptive.close()
            
            print(f"  Size {size[0]}x{size[1]}: Fixed={results['peak_usage_fixed'][-1]/1024**2:.1f}MB, "
                  f"Adaptive={results['peak_usage_adaptive'][-1]/1024**2:.1f}MB")
        
        return results
    
    def generate_summary_report(self):
        """Generate a summary report of all benchmark results."""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY REPORT")
        print("=" * 60)
        
        # Calculate improvements
        improvements = {}
        
        # Adaptive buffer sizing
        if 'adaptive_buffer' in self.results:
            fixed_times = np.mean(self.results['adaptive_buffer']['fixed_buffer_times'])
            adaptive_times = np.mean(self.results['adaptive_buffer']['adaptive_buffer_times'])
            improvements['adaptive_buffer'] = ((fixed_times - adaptive_times) / fixed_times) * 100
        
        # Zero-copy operations
        if 'zero_copy' in self.results:
            copy_times = np.mean(self.results['zero_copy']['copy_times'])
            zero_copy_times = np.mean(self.results['zero_copy']['zero_copy_times'])
            improvements['zero_copy'] = ((copy_times - zero_copy_times) / copy_times) * 100
        
        # Memory pool
        if 'memory_pool' in self.results:
            no_pool_times = np.mean(self.results['memory_pool']['no_pool_times'])
            pool_times = np.mean(self.results['memory_pool']['pool_times'])
            improvements['memory_pool'] = ((no_pool_times - pool_times) / no_pool_times) * 100
        
        # Batched processing
        if 'batched_processing' in self.results:
            individual_times = np.mean(self.results['batched_processing']['individual_times'])
            batched_times = np.mean(self.results['batched_processing']['batched_times'])
            improvements['batched_processing'] = ((individual_times - batched_times) / individual_times) * 100
        
        # Tensor serialization
        if 'tensor_serialization' in self.results:
            standard_times = np.mean(self.results['tensor_serialization']['standard_times'])
            optimized_times = np.mean(self.results['tensor_serialization']['optimized_times'])
            improvements['tensor_serialization'] = ((standard_times - optimized_times) / standard_times) * 100
        
        # Memory efficiency
        if 'memory_efficiency' in self.results:
            fixed_usage = np.mean(self.results['memory_efficiency']['peak_usage_fixed'])
            adaptive_usage = np.mean(self.results['memory_efficiency']['peak_usage_adaptive'])
            improvements['memory_efficiency'] = ((fixed_usage - adaptive_usage) / fixed_usage) * 100
        
        print("\nPerformance Improvements:")
        print("-" * 40)
        for feature, improvement in improvements.items():
            print(f"{feature.replace('_', ' ').title()}: {improvement:.1f}% improvement")
        
        # Overall improvement
        if improvements:
            overall_improvement = np.mean(list(improvements.values()))
            print(f"\nOverall Performance Improvement: {overall_improvement:.1f}%")
        
        print("\nDetailed results saved to 'shared_memory_benchmark_results.npy'")
        np.save('shared_memory_benchmark_results.npy', self.results)
    
    def plot_results(self):
        """Plot benchmark results."""
        try:
            import matplotlib.pyplot as plt
            
            # Create subplots
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('Shared Memory Optimization Benchmarks', fontsize=16)
            
            # Plot 1: Adaptive Buffer Sizing
            if 'adaptive_buffer' in self.results:
                ax = axes[0, 0]
                x = self.results['adaptive_buffer']['tensor_sizes']
                ax.plot(x, self.results['adaptive_buffer']['fixed_buffer_times'], 'r-', label='Fixed Buffer')
                ax.plot(x, self.results['adaptive_buffer']['adaptive_buffer_times'], 'b-', label='Adaptive Buffer')
                ax.set_xlabel('Tensor Size (elements)')
                ax.set_ylabel('Time (s)')
                ax.set_title('Adaptive Buffer Sizing')
                ax.legend()
                ax.grid(True)
            
            # Plot 2: Zero-Copy Operations
            if 'zero_copy' in self.results:
                ax = axes[0, 1]
                x = self.results['zero_copy']['tensor_sizes']
                ax.plot(x, self.results['zero_copy']['copy_times'], 'r-', label='Copy')
                ax.plot(x, self.results['zero_copy']['zero_copy_times'], 'b-', label='Zero-Copy')
                ax.set_xlabel('Tensor Size (elements)')
                ax.set_ylabel('Time (s)')
                ax.set_title('Zero-Copy Operations')
                ax.legend()
                ax.grid(True)
            
            # Plot 3: Memory Pool Performance
            if 'memory_pool' in self.results:
                ax = axes[0, 2]
                x = self.results['memory_pool']['tensor_sizes']
                ax.plot(x, self.results['memory_pool']['no_pool_times'], 'r-', label='No Pool')
                ax.plot(x, self.results['memory_pool']['pool_times'], 'b-', label='Memory Pool')
                ax.set_xlabel('Tensor Size (elements)')
                ax.set_ylabel('Time (s)')
                ax.set_title('Memory Pool Performance')
                ax.legend()
                ax.grid(True)
            
            # Plot 4: Batched Processing
            if 'batched_processing' in self.results:
                ax = axes[1, 0]
                x = self.results['batched_processing']['batch_sizes']
                ax.plot(x, self.results['batched_processing']['individual_times'], 'r-', label='Individual')
                ax.plot(x, self.results['batched_processing']['batched_times'], 'b-', label='Batched')
                ax.set_xlabel('Batch Size')
                ax.set_ylabel('Time (s)')
                ax.set_title('Batched Processing')
                ax.legend()
                ax.grid(True)
            
            # Plot 5: Tensor Serialization
            if 'tensor_serialization' in self.results:
                ax = axes[1, 1]
                x = self.results['tensor_serialization']['tensor_sizes']
                ax.plot(x, self.results['tensor_serialization']['standard_times'], 'r-', label='Standard')
                ax.plot(x, self.results['tensor_serialization']['optimized_times'], 'b-', label='Optimized')
                ax.set_xlabel('Tensor Size (elements)')
                ax.set_ylabel('Time (s)')
                ax.set_title('Tensor Serialization')
                ax.legend()
                ax.grid(True)
            
            # Plot 6: Memory Efficiency
            if 'memory_efficiency' in self.results:
                ax = axes[1, 2]
                x = self.results['memory_efficiency']['tensor_sizes']
                ax.plot(x, np.array(self.results['memory_efficiency']['peak_usage_fixed'])/1024**2, 'r-', label='Fixed Buffer')
                ax.plot(x, np.array(self.results['memory_efficiency']['peak_usage_adaptive'])/1024**2, 'b-', label='Adaptive Buffer')
                ax.set_xlabel('Tensor Size (elements)')
                ax.set_ylabel('Peak Memory Usage (MB)')
                ax.set_title('Memory Efficiency')
                ax.legend()
                ax.grid(True)
            
            plt.tight_layout()
            plt.savefig('shared_memory_benchmark_plots.png', dpi=300, bbox_inches='tight')
            print("\nBenchmark plots saved to 'shared_memory_benchmark_plots.png'")
            
        except ImportError:
            print("\nMatplotlib not available. Skipping plot generation.")


def main():
    """Main function to run benchmarks."""
    print("ExLlamaV3 Shared Memory Optimization Benchmark")
    print("=" * 60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA not available. This benchmark requires CUDA.")
        return
    
    # Get device
    device = 0
    if torch.cuda.device_count() > 1:
        print(f"Found {torch.cuda.device_count()} CUDA devices. Using device {device}.")
    
    # Run benchmarks
    benchmark = SharedMemoryBenchmark(device=device)
    results = benchmark.run_all_benchmarks()
    
    # Generate plots
    benchmark.plot_results()
    
    print("\nBenchmark completed successfully!")
    print("Results saved to 'shared_memory_benchmark_results.npy'")
    print("Plots saved to 'shared_memory_benchmark_plots.png'")


if __name__ == "__main__":
    main()