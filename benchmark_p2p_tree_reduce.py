#!/usr/bin/env python3
"""
Performance benchmark for P2P tree-based reduction operations.

This script benchmarks the tree-based reduction against ring-based reduction
and measures the performance improvements.
"""

import torch
import time
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

# Add the exllamav3 directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'exllamav3'))

from exllamav3.model.model_tp_p2p import P2PTopology
from exllamav3.util import log_tp

class PerformanceBenchmark:
    """Performance benchmark for P2P reduction operations."""
    
    def __init__(self):
        self.results = {}
        self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
    def benchmark_theoretical_performance(self):
        """Benchmark theoretical performance based on communication steps."""
        print("Theoretical Performance Analysis")
        print("=" * 50)
        
        device_counts = [2, 4, 8, 16, 32]
        
        for num_devices in device_counts:
            print(f"\nDevice count: {num_devices}")
            
            # Ring reduction steps
            ring_steps = num_devices - 1
            
            # Tree reduction steps (binary tree)
            tree_steps = int(np.ceil(np.log2(num_devices)))
            
            # K-ary tree steps (4-ary)
            kary_steps = int(np.ceil(np.log(num_devices, 4))) if num_devices > 1 else 0
            
            # Balanced tree steps (optimal branching)
            if num_devices <= 4:
                balanced_steps = int(np.ceil(np.log2(num_devices)))
            elif num_devices <= 16:
                balanced_steps = int(np.ceil(np.log(num_devices, 4)))
            else:
                balanced_steps = int(np.ceil(np.log(num_devices, 8)))
            
            print(f"  Ring reduction steps:     {ring_steps}")
            print(f"  Binary tree steps:        {tree_steps}")
            print(f"  4-ary tree steps:        {kary_steps}")
            print(f"  Balanced tree steps:     {balanced_steps}")
            
            # Calculate theoretical speedup
            ring_improvement = ring_steps / tree_steps if tree_steps > 0 else 0
            kary_improvement = ring_steps / kary_steps if kary_steps > 0 else 0
            balanced_improvement = ring_steps / balanced_steps if balanced_steps > 0 else 0
            
            print(f"  Binary tree speedup:      {ring_improvement:.2f}x")
            print(f"  4-ary tree speedup:      {kary_improvement:.2f}x")
            print(f"  Balanced tree speedup:   {balanced_improvement:.2f}x")
            
            # Store results
            self.results[num_devices] = {
                'ring_steps': ring_steps,
                'binary_tree_steps': tree_steps,
                'kary_tree_steps': kary_steps,
                'balanced_tree_steps': balanced_steps,
                'binary_speedup': ring_improvement,
                'kary_speedup': kary_improvement,
                'balanced_speedup': balanced_improvement
            }
    
    def benchmark_scalability(self):
        """Benchmark scalability with different tensor sizes."""
        print("\nScalability Analysis")
        print("=" * 50)
        
        # Test different tensor sizes (in elements)
        tensor_sizes = [1024, 4096, 16384, 65536, 262144, 1048576]  # 1K to 1M elements
        
        # Test different device counts
        device_counts = [2, 4, 8, 16]
        
        for num_devices in device_counts:
            print(f"\nDevice count: {num_devices}")
            
            # Create topology
            devices = list(range(num_devices))
            topology = P2PTopology(devices)
            
            for size in tensor_sizes:
                # Calculate theoretical communication time
                # Assuming bandwidth of 10 GB/s and latency of 1 μs
                
                data_size_mb = size * 4 / (1024 * 1024)  # Assuming float32
                
                # Ring reduction: (N-1) steps
                ring_time = (num_devices - 1) * (data_size_mb / 10000 + 0.001)  # in seconds
                
                # Tree reduction: log2(N) steps
                tree_time = int(np.ceil(np.log2(num_devices))) * (data_size_mb / 10000 + 0.001)
                
                speedup = ring_time / tree_time if tree_time > 0 else 0
                
                print(f"  Size: {size:6d} elements ({data_size_mb:5.1f} MB) - "
                      f"Ring: {ring_time*1000:6.2f}ms, Tree: {tree_time*1000:6.2f}ms, "
                      f"Speedup: {speedup:.2f}x")
    
    def benchmark_topology_impact(self):
        """Benchmark impact of different P2P topologies."""
        print("\nTopology Impact Analysis")
        print("=" * 50)
        
        # Simulate different connectivity ratios
        connectivity_ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
        device_counts = [4, 8, 16]
        
        for num_devices in device_counts:
            print(f"\nDevice count: {num_devices}")
            
            for ratio in connectivity_ratios:
                # Simulate topology with given connectivity ratio
                devices = list(range(num_devices))
                topology = P2PTopology(devices)
                
                # Override connectivity ratio for simulation
                topology.connectivity_ratio = ratio
                
                # Select algorithm based on connectivity
                tensor_size = 1024 * 1024  # 1MB
                algorithm = topology.select_reduce_algorithm(tensor_size)
                
                # Calculate expected performance
                if algorithm == "ring":
                    steps = num_devices - 1
                elif algorithm == "binary_tree":
                    steps = int(np.ceil(np.log2(num_devices)))
                elif algorithm == "kary_tree":
                    steps = int(np.ceil(np.log(num_devices, 4)))
                else:  # balanced_tree
                    steps = int(np.ceil(np.log2(num_devices)))
                
                print(f"  Connectivity: {ratio:.1f} - Algorithm: {algorithm:12s} - Steps: {steps}")
    
    def generate_performance_report(self):
        """Generate a performance report."""
        print("\nPerformance Summary")
        print("=" * 50)
        
        if not self.results:
            print("No benchmark results available")
            return
        
        # Calculate average speedup across all device counts
        binary_speedups = [r['binary_speedup'] for r in self.results.values()]
        kary_speedups = [r['kary_speedup'] for r in self.results.values()]
        balanced_speedups = [r['balanced_speedup'] for r in self.results.values()]
        
        avg_binary_speedup = np.mean(binary_speedups)
        avg_kary_speedup = np.mean(kary_speedups)
        avg_balanced_speedup = np.mean(balanced_speedups)
        
        print(f"Average binary tree speedup:     {avg_binary_speedup:.2f}x")
        print(f"Average 4-ary tree speedup:     {avg_kary_speedup:.2f}x")
        print(f"Average balanced tree speedup:  {avg_balanced_speedup:.2f}x")
        
        # Find best performing configuration
        max_speedup = 0
        best_config = None
        
        for num_devices, result in self.results.items():
            if result['balanced_speedup'] > max_speedup:
                max_speedup = result['balanced_speedup']
                best_config = (num_devices, "balanced_tree")
        
        print(f"\nBest configuration: {best_config[1]} with {best_config[0]} devices")
        print(f"Maximum speedup: {max_speedup:.2f}x")
    
    def plot_performance_comparison(self):
        """Plot performance comparison between algorithms."""
        if not self.results:
            print("No results to plot")
            return
        
        try:
            device_counts = sorted(self.results.keys())
            binary_speedups = [self.results[d]['binary_speedup'] for d in device_counts]
            kary_speedups = [self.results[d]['kary_speedup'] for d in device_counts]
            balanced_speedups = [self.results[d]['balanced_speedup'] for d in device_counts]
            
            plt.figure(figsize=(10, 6))
            plt.plot(device_counts, binary_speedups, 'o-', label='Binary Tree')
            plt.plot(device_counts, kary_speedups, 's-', label='4-ary Tree')
            plt.plot(device_counts, balanced_speedups, '^-', label='Balanced Tree')
            
            plt.xlabel('Number of Devices')
            plt.ylabel('Speedup vs Ring Reduction')
            plt.title('P2P Tree Reduction Performance')
            plt.legend()
            plt.grid(True)
            
            # Save plot
            plt.savefig('p2p_tree_reduce_performance.png')
            print("\nPerformance plot saved to 'p2p_tree_reduce_performance.png'")
            
        except ImportError:
            print("\nMatplotlib not available, skipping plot generation")
    
    def run_all_benchmarks(self):
        """Run all benchmarks."""
        print("P2P Tree Reduction Performance Benchmark")
        print("=" * 50)
        
        if self.device_count == 0:
            print("CUDA not available, running theoretical benchmarks only")
        
        self.benchmark_theoretical_performance()
        self.benchmark_scalability()
        self.benchmark_topology_impact()
        self.generate_performance_report()
        self.plot_performance_comparison()
        
        print("\nBenchmark completed!")

def main():
    """Main function."""
    benchmark = PerformanceBenchmark()
    benchmark.run_all_benchmarks()

if __name__ == "__main__":
    main()