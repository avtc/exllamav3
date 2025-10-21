#!/usr/bin/env python3
"""
Test script for P2P topology detection and validation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'exllamav3'))

import torch
from exllamav3.model.model_tp_p2p import P2PTopology, detect_p2p_capabilities, build_optimal_topology, is_fully_connected, find_optimal_tree

def test_p2p_topology():
    """Test P2P topology detection functionality."""
    print("Testing P2P topology detection...")
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available, skipping P2P tests")
        return
        
    # Get available devices
    device_count = torch.cuda.device_count()
    print(f"Found {device_count} CUDA devices")
    
    if device_count < 2:
        print("Need at least 2 GPUs for P2P testing")
        return
        
    # Test with first 4 devices or all if less than 4
    active_devices = list(range(min(4, device_count)))
    print(f"Testing with devices: {active_devices}")
    
    # First test PyTorch's built-in P2P detection
    print("\nTesting PyTorch P2P detection:")
    for i in active_devices:
        for j in active_devices:
            if i != j:
                try:
                    can_access = torch.cuda.can_device_access_peer(i, j)
                    print(f"  Device {i} can access device {j}: {can_access}")
                except Exception as e:
                    print(f"  Error checking {i} -> {j}: {e}")
    
    try:
        # Test P2PTopology class
        topology = P2PTopology(active_devices)
        summary = topology.get_topology_summary()
        print(f"Topology summary: {summary}")
        
        # Test P2P matrix
        p2p_matrix = topology.get_p2p_matrix()
        print(f"P2P matrix shape: {p2p_matrix.shape}")
        print("P2P matrix:")
        print(p2p_matrix)
        
        # Test individual functions
        print(f"Fully connected: {topology.is_fully_connected()}")
        
        # Test peer access
        for i in active_devices:
            for j in active_devices:
                if i != j:
                    can_access = topology.can_access_peer(i, j)
                    print(f"Device {i} can access device {j}: {can_access}")
        
        # Test topology building
        reduce_topology = topology.build_optimal_topology("reduce")
        print(f"Reduce topology: {reduce_topology}")
        
        broadcast_topology = topology.build_optimal_topology("broadcast")
        print(f"Broadcast topology: {broadcast_topology}")
        
        # Test standalone functions
        p2p_matrix_func = detect_p2p_capabilities(active_devices)
        print(f"Standalone function matrix shape: {p2p_matrix_func.shape}")
        
        fully_connected = is_fully_connected(p2p_matrix_func)
        print(f"Standalone fully connected: {fully_connected}")
        
        optimal_tree = find_optimal_tree(p2p_matrix_func)
        print(f"Optimal tree: {optimal_tree}")
        
        optimal_topology = build_optimal_topology(p2p_matrix_func, "reduce")
        print(f"Optimal topology: {optimal_topology}")
        
        print("P2P topology tests completed successfully!")
        
    except Exception as e:
        print(f"P2P topology test failed: {e}")
        import traceback
        traceback.print_exc()

def test_backend_selection():
    """Test backend selection logic."""
    print("\nTesting backend selection...")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping backend tests")
        return
        
    device_count = torch.cuda.device_count()
    if device_count < 2:
        print("Need at least 2 GPUs for backend testing")
        return
        
    try:
        from exllamav3.model.model_tp_fn import init_pg
        
        # Test different backend types
        active_devices = list(range(min(2, device_count)))
        output_device = active_devices[0]
        
        backend_args = {
            "init_method": "tcp://127.0.0.1:12345",
            "uuid": "test-uuid"
        }
        
        # Test P2P backend
        print("Testing P2P backend...")
        backend_args["type"] = "p2p"
        try:
            context = init_pg(active_devices[0], active_devices, output_device, backend_args, master=True)
            print("P2P backend initialized successfully")
            context["backend"].close()
        except Exception as e:
            print(f"P2P backend initialization failed: {e}")
        
        # Test auto backend
        print("Testing auto backend...")
        backend_args["type"] = "auto"
        try:
            context = init_pg(active_devices[0], active_devices, output_device, backend_args, master=True)
            print("Auto backend initialized successfully")
            context["backend"].close()
        except Exception as e:
            print(f"Auto backend initialization failed: {e}")
            
        print("Backend selection tests completed!")
        
    except Exception as e:
        print(f"Backend selection test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_p2p_topology()
    test_backend_selection()