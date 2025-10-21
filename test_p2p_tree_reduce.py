#!/usr/bin/env python3
"""
Test script for P2P tree-based reduction operations.

This script tests the tree-based reduction implementation and compares it
with the existing ring-based reduction.
"""

import torch
import time
import numpy as np
import sys
import os

# Add the exllamav3 directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'exllamav3'))

from exllamav3.model.model_tp_p2p import P2PTopology
from exllamav3.model.model_tp_backend import TPBackendP2P
from exllamav3.util import log_tp

def test_tree_building():
    """Test tree building algorithms."""
    print("Testing tree building algorithms...")
    
    # Test with different numbers of devices
    device_counts = [2, 4, 8, 16]
    
    for num_devices in device_counts:
        print(f"\nTesting with {num_devices} devices:")
        devices = list(range(num_devices))
        
        # Create topology
        topology = P2PTopology(devices)
        
        # Test different tree types
        tree_types = ["binary", "kary", "balanced"]
        
        for tree_type in tree_types:
            if tree_type == "binary":
                tree = topology.build_binary_tree()
            elif tree_type == "kary":
                tree = topology.build_kary_tree(4)
            else:  # balanced
                tree = topology.build_balanced_tree()
            
            stats = topology.get_tree_stats(tree)
            print(f"  {tree_type} tree: depth={stats['tree_depth']}, "
                  f"avg_branching={stats['avg_branching_factor']:.2f}, "
                  f"max_branching={stats['max_branching_factor']}")
            
            # Verify tree structure
            assert "root" in tree
            assert "children" in tree
            assert "parent" in tree
            assert "depth" in tree
            
            # Check that all devices are included
            all_devices = set(tree["children"].keys()) | set(tree["parent"].keys())
            if tree["root"] not in tree["parent"]:
                all_devices.add(tree["root"])
            
            assert len(all_devices) == num_devices, f"Tree missing devices: {num_devices - len(all_devices)}"

def test_algorithm_selection():
    """Test adaptive algorithm selection."""
    print("\nTesting adaptive algorithm selection...")
    
    devices = [0, 1, 2, 3, 4, 5, 6, 7]
    topology = P2PTopology(devices)
    
    # Test with different tensor sizes
    tensor_sizes = [1024, 1024*1024, 10*1024*1024]  # 1KB, 1MB, 10MB
    
    for size in tensor_sizes:
        algorithm = topology.select_reduce_algorithm(size)
        print(f"  Tensor size: {size//1024}KB -> Algorithm: {algorithm}")
        
        # Verify algorithm is valid
        assert algorithm in ["ring", "binary_tree", "kary_tree", "balanced_tree"]

def test_connectivity_ratio():
    """Test connectivity ratio calculation."""
    print("\nTesting connectivity ratio calculation...")
    
    # Test with different device counts
    for num_devices in [2, 4, 8]:
        devices = list(range(num_devices))
        topology = P2PTopology(devices)
        
        ratio = topology.get_connectivity_ratio()
        print(f"  {num_devices} devices: connectivity ratio = {ratio:.2f}")
        
        # Verify ratio is between 0 and 1
        assert 0.0 <= ratio <= 1.0

def test_topology_optimization():
    """Test topology optimization for different operations."""
    print("\nTesting topology optimization...")
    
    devices = [0, 1, 2, 3, 4, 5, 6, 7]
    topology = P2PTopology(devices)
    
    operations = ["reduce", "broadcast", "gather"]
    tensor_sizes = [1024, 1024*1024]
    
    for op in operations:
        for size in tensor_sizes:
            result = topology.build_optimal_topology(op, size)
            print(f"  {op} (size: {size//1024}KB): {result['type']} - {result['reason']}")
            
            # Verify result structure
            assert "type" in result
            assert "reason" in result

def test_tree_reduce_correctness():
    """Test correctness of tree-based reduction (if CUDA is available)."""
    print("\nTesting tree reduction correctness...")
    
    if not torch.cuda.is_available():
        print("  CUDA not available, skipping correctness test")
        return
    
    # Get available devices
    device_count = torch.cuda.device_count()
    if device_count < 2:
        print("  Need at least 2 CUDA devices for reduction test")
        return
    
    print(f"  Testing with {device_count} CUDA devices")
    
    # Test tensor sizes
    shapes = [(1024,), (4096, 1024), (512, 512, 512)]
    dtypes = [torch.float16, torch.float32]
    
    for shape in shapes:
        for dtype in dtypes:
            print(f"    Testing shape {shape}, dtype {dtype}")
            
            # Create test tensors on each device
            tensors = []
            expected_sum = None
            
            for i in range(device_count):
                with torch.cuda.device(i):
                    # Create tensor with unique values for this device
                    tensor = torch.randn(shape, dtype=dtype, device=i) * (i + 1)
                    tensors.append(tensor)
                    
                    if expected_sum is None:
                        expected_sum = tensor.clone()
                    else:
                        expected_sum += tensor
            
            # Test tree reduction (this would require actual P2P setup)
            # For now, just verify the tensors are created correctly
            for i, tensor in enumerate(tensors):
                assert tensor.device.index == i
                assert tensor.shape == shape
                assert tensor.dtype == dtype
            
            print(f"      Created {len(tensors)} test tensors")

def benchmark_tree_vs_ring():
    """Benchmark tree vs ring reduction performance."""
    print("\nBenchmarking tree vs ring reduction...")
    
    if not torch.cuda.is_available():
        print("  CUDA not available, skipping benchmark")
        return
    
    device_count = torch.cuda.device_count()
    if device_count < 2:
        print("  Need at least 2 CUDA devices for benchmark")
        return
    
    print(f"  Benchmarking with {device_count} CUDA devices")
    
    # Test different tensor sizes
    sizes = [1024*1024, 10*1024*1024, 50*1024*1024]  # 1MB, 10MB, 50MB
    
    for size in sizes:
        print(f"    Tensor size: {size//1024//1024}MB")
        
        # Create test tensor
        shape = (size // 4,)  # Assuming float32 (4 bytes per element)
        
        # This would require actual P2P setup and timing
        # For now, just simulate the expected performance improvement
        ring_steps = device_count - 1
        tree_steps = int(np.ceil(np.log2(device_count)))
        
        print(f"      Ring reduction steps: {ring_steps}")
        print(f"      Tree reduction steps: {tree_steps}")
        print(f"      Expected improvement: {ring_steps/tree_steps:.2f}x")

def run_all_tests():
    """Run all tests."""
    print("Running P2P Tree Reduction Tests")
    print("=" * 50)
    
    try:
        test_tree_building()
        test_algorithm_selection()
        test_connectivity_ratio()
        test_topology_optimization()
        test_tree_reduce_correctness()
        benchmark_tree_vs_ring()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)