#!/usr/bin/env python3
"""
Performance benchmarks for P2P direct memory access functions.
This script measures bandwidth, latency, and throughput of various P2P memory operations.
"""

import torch
import time
import numpy as np
import argparse
import sys
import os

# Add the exllamav3 directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'exllamav3'))

from exllamav3.model.model_tp_backend import TPBackendP2P
from exllamav3.model.model_tp_p2p import P2PTopology
import exllamav3_ext as ext


def benchmark_p2p_bandwidth(backend, src_device, dst_device, sizes_mb, num_iterations=10):
    """Benchmark P2P bandwidth between two devices."""
    print(f"\n=== P2P Bandwidth Benchmark: {src_device} -> {dst_device} ===")
    
    results = {}
    
    for size_mb in sizes_mb:
        size_bytes = size_mb * 1024 * 1024
        size_elements = size_bytes // 4  # Assuming float32
        
        # Create test tensors
        src_tensor = torch.randn(size_elements, dtype=torch.float32, device=src_device)
        dst_tensor = torch.zeros(size_elements, dtype=torch.float32, device=dst_device)
        
        # Warm up
        for _ in range(3):
            ext.p2p_copy_tensor_async(src_device, dst_device, src_tensor, dst_tensor, backend.abort_flag)
        
        torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        
        for _ in range(num_iterations):
            ext.p2p_copy_tensor_async(src_device, dst_device, src_tensor, dst_tensor, backend.abort_flag)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        total_bytes = size_bytes * num_iterations
        bandwidth_gb_s = (total_bytes / (1024**3)) / elapsed_time
        
        results[size_mb] = bandwidth_gb_s
        print(f"Size: {size_mb:4d} MB | Bandwidth: {bandwidth_gb_s:8.2f} GB/s | Time: {elapsed_time/num_iterations*1000:6.2f} ms")
    
    return results


def benchmark_p2p_latency(backend, src_device, dst_device, sizes_kb, num_iterations=100):
    """Benchmark P2P latency between two devices."""
    print(f"\n=== P2P Latency Benchmark: {src_device} -> {dst_device} ===")
    
    results = {}
    
    for size_kb in sizes_kb:
        size_bytes = size_kb * 1024
        size_elements = size_bytes // 4  # Assuming float32
        
        # Create test tensors
        src_tensor = torch.randn(size_elements, dtype=torch.float32, device=src_device)
        dst_tensor = torch.zeros(size_elements, dtype=torch.float32, device=dst_device)
        
        # Warm up
        for _ in range(3):
            ext.p2p_copy_tensor_async(src_device, dst_device, src_tensor, dst_tensor, backend.abort_flag)
        
        torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        
        for _ in range(num_iterations):
            ext.p2p_copy_tensor_async(src_device, dst_device, src_tensor, dst_tensor, backend.abort_flag)
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        avg_latency_us = (elapsed_time / num_iterations) * 1000000  # Convert to microseconds
        
        results[size_kb] = avg_latency_us
        print(f"Size: {size_kb:4d} KB | Latency: {avg_latency_us:8.2f} μs")
    
    return results


def benchmark_p2p_batch_copy(backend, src_device, dst_device, batch_sizes, tensor_size_mb=16):
    """Benchmark P2P batch copy operations."""
    print(f"\n=== P2P Batch Copy Benchmark: {src_device} -> {dst_device} ===")
    
    results = {}
    
    for batch_size in batch_sizes:
        size_bytes = tensor_size_mb * 1024 * 1024
        size_elements = size_bytes // 4  # Assuming float32
        
        # Create test tensors
        src_tensors = []
        dst_tensors = []
        
        for _ in range(batch_size):
            src_tensor = torch.randn(size_elements, dtype=torch.float32, device=src_device)
            dst_tensor = torch.zeros(size_elements, dtype=torch.float32, device=dst_device)
            src_tensors.append(src_tensor)
            dst_tensors.append(dst_tensor)
        
        # Warm up
        for _ in range(3):
            ext.p2p_copy_tensor_batch(src_device, dst_device, src_tensors, dst_tensors, backend.abort_flag)
        
        torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        
        for _ in range(10):  # Fewer iterations for batch operations
            ext.p2p_copy_tensor_batch(src_device, dst_device, src_tensors, dst_tensors, backend.abort_flag)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        total_bytes = size_bytes * batch_size * 10  # 10 iterations
        bandwidth_gb_s = (total_bytes / (1024**3)) / elapsed_time
        
        results[batch_size] = bandwidth_gb_s
        print(f"Batch: {batch_size:2d} tensors | Bandwidth: {bandwidth_gb_s:8.2f} GB/s | Time: {elapsed_time/10*1000:6.2f} ms")
    
    return results


def benchmark_p2p_vs_host_copy(backend, src_device, dst_device, sizes_mb, num_iterations=10):
    """Compare P2P direct copy vs host-mediated copy."""
    print(f"\n=== P2P vs Host Copy Comparison: {src_device} -> {dst_device} ===")
    
    results = {'p2p': {}, 'host': {}}
    
    for size_mb in sizes_mb:
        size_bytes = size_mb * 1024 * 1024
        size_elements = size_bytes // 4  # Assuming float32
        
        # Create test tensors
        src_tensor = torch.randn(size_elements, dtype=torch.float32, device=src_device)
        dst_tensor = torch.zeros(size_elements, dtype=torch.float32, device=dst_device)
        
        # Benchmark P2P direct copy
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(num_iterations):
            ext.p2p_copy_tensor_async(src_device, dst_device, src_tensor, dst_tensor, backend.abort_flag)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        p2p_time = end_time - start_time
        p2p_bandwidth = (size_bytes * num_iterations / (1024**3)) / p2p_time
        
        # Benchmark host-mediated copy
        host_tensor = torch.empty_like(src_tensor, device='cpu')
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(num_iterations):
            # Copy to host
            host_tensor.copy_(src_tensor, non_blocking=True)
            # Copy from host
            dst_tensor.copy_(host_tensor, non_blocking=True)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        host_time = end_time - start_time
        host_bandwidth = (size_bytes * num_iterations / (1024**3)) / host_time
        
        results['p2p'][size_mb] = p2p_bandwidth
        results['host'][size_mb] = host_bandwidth
        
        speedup = host_bandwidth / p2p_bandwidth
        print(f"Size: {size_mb:4d} MB | P2P: {p2p_bandwidth:8.2f} GB/s | Host: {host_bandwidth:8.2f} GB/s | Speedup: {speedup:.2f}x")
    
    return results


def benchmark_memory_pool_performance(backend, device, allocation_sizes_mb, num_allocations=100):
    """Benchmark P2P memory pool allocation and deallocation."""
    print(f"\n=== Memory Pool Performance Benchmark: Device {device} ===")
    
    results = {}
    
    for size_mb in allocation_sizes_mb:
        size_bytes = size_mb * 1024 * 1024
        
        # Benchmark allocation
        start_time = time.time()
        pointers = []
        
        for _ in range(num_allocations):
            ptr = ext.p2p_allocate_from_direct_pool(device, size_bytes, -1, backend.abort_flag)
            if ptr:
                pointers.append(ptr)
        
        end_time = time.time()
        alloc_time = end_time - start_time
        
        # Benchmark deallocation
        start_time = time.time()
        
        for ptr in pointers:
            ext.p2p_free_to_direct_pool(device, ptr, size_bytes, backend.abort_flag)
        
        end_time = time.time()
        dealloc_time = end_time - start_time
        
        alloc_rate = num_allocations / alloc_time
        dealloc_rate = num_allocations / dealloc_time
        
        results[size_mb] = {
            'alloc_rate': alloc_rate,
            'dealloc_rate': dealloc_rate,
            'alloc_time_ms': alloc_time * 1000,
            'dealloc_time_ms': dealloc_time * 1000
        }
        
        print(f"Size: {size_mb:4d} MB | Alloc: {alloc_rate:8.1f} ops/s | Dealloc: {dealloc_rate:8.1f} ops/s")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='P2P Direct Memory Performance Benchmarks')
    parser.add_argument('--devices', type=str, default='0,1', help='Comma-separated list of GPU devices to use')
    parser.add_argument('--sizes-mb', type=str, default='1,4,16,64,256', help='Comma-separated list of sizes in MB')
    parser.add_argument('--sizes-kb', type=str, default='1,4,16,64,256', help='Comma-separated list of sizes in KB')
    parser.add_argument('--batch-sizes', type=str, default='1,4,8,16', help='Comma-separated list of batch sizes')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations for bandwidth tests')
    parser.add_argument('--latency-iterations', type=int, default=100, help='Number of iterations for latency tests')
    
    args = parser.parse_args()
    
    # Parse arguments
    devices = [int(d) for d in args.devices.split(',')]
    sizes_mb = [int(s) for s in args.sizes_mb.split(',')]
    sizes_kb = [int(s) for s in args.sizes_kb.split(',')]
    batch_sizes = [int(s) for s in args.batch_sizes.split(',')]
    
    print("P2P Direct Memory Performance Benchmarks")
    print(f"Devices: {devices}")
    print(f"Sizes (MB): {sizes_mb}")
    print(f"Sizes (KB): {sizes_kb}")
    print(f"Batch sizes: {batch_sizes}")
    
    # Check available devices
    device_count = torch.cuda.device_count()
    print(f"Available CUDA devices: {device_count}")
    
    if len(devices) < 2:
        print("Error: At least 2 GPU devices are required for P2P benchmarks")
        return
    
    # Initialize P2P topology
    try:
        topology = P2PTopology(devices)
        topology_summary = topology.get_topology_summary()
        print(f"P2P Topology: {topology_summary}")
        
        if topology_summary.get("connectivity_ratio", 0) == 0:
            print("Warning: No P2P connectivity detected between devices")
            return
    except Exception as e:
        print(f"Error initializing P2P topology: {e}")
        return
    
    # Initialize backend for each device
    backends = {}
    for device in devices:
        try:
            backend = TPBackendP2P(
                device=device,
                active_devices=devices,
                output_device=devices[0],
                init_method='tcp://localhost:12345',
                master=(device == devices[0]),
                uuid='p2p_benchmark'
            )
            backends[device] = backend
            print(f"Initialized P2P backend for device {device}")
        except Exception as e:
            print(f"Error initializing backend for device {device}: {e}")
            return
    
    try:
        # Run benchmarks
        if len(devices) >= 2:
            src_device, dst_device = devices[0], devices[1]
            
            # Bandwidth benchmarks
            bandwidth_results = benchmark_p2p_bandwidth(
                backends[src_device], src_device, dst_device, 
                sizes_mb, args.iterations
            )
            
            # Latency benchmarks
            latency_results = benchmark_p2p_latency(
                backends[src_device], src_device, dst_device,
                sizes_kb, args.latency_iterations
            )
            
            # Batch copy benchmarks
            batch_results = benchmark_p2p_batch_copy(
                backends[src_device], src_device, dst_device,
                batch_sizes
            )
            
            # P2P vs Host copy comparison
            comparison_results = benchmark_p2p_vs_host_copy(
                backends[src_device], src_device, dst_device,
                sizes_mb, args.iterations
            )
        
        # Memory pool benchmarks
        pool_results = benchmark_memory_pool_performance(
            backends[devices[0]], devices[0],
            [1, 4, 16, 64], 50  # Smaller allocation sizes for pool test
        )
        
        print("\n=== Benchmark Summary ===")
        print("All benchmarks completed successfully!")
        
    finally:
        # Cleanup backends
        for device, backend in backends.items():
            try:
                backend.close()
                print(f"Closed P2P backend for device {device}")
            except Exception as e:
                print(f"Error closing backend for device {device}: {e}")


if __name__ == '__main__':
    main()