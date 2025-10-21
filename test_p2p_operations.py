#!/usr/bin/env python3
"""
Test script for P2P operations in ExLlamaV3.

This script tests the P2P backend implementation with basic operations.
"""

import torch
import sys
import os

# Add the exllamav3 directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'exllamav3'))

from exllamav3.model.model_tp_backend import TPBackendP2P
from exllamav3.model.model_tp_p2p import P2PTopology

def test_p2p_topology_detection():
    """Test P2P topology detection."""
    print("Testing P2P topology detection...")
    
    # Get available devices
    if not torch.cuda.is_available():
        print("CUDA not available, skipping P2P tests")
        return False
        
    device_count = torch.cuda.device_count()
    if device_count < 2:
        print("Need at least 2 GPUs for P2P testing, skipping")
        return False
    
    # Test with first two devices
    active_devices = [0, 1]
    
    try:
        topology = P2PTopology(active_devices)
        summary = topology.get_topology_summary()
        print(f"P2P topology summary: {summary}")
        
        # Check if P2P is available
        if summary.get("connectivity_ratio", 0) > 0:
            print("P2P is available between devices")
            return True
        else:
            print("P2P is not available between devices")
            return False
    except Exception as e:
        print(f"P2P topology detection failed: {e}")
        return False

def test_p2p_backend():
    """Test P2P backend initialization and operations."""
    print("\nTesting P2P backend...")
    
    # Get available devices
    if not torch.cuda.is_available():
        print("CUDA not available, skipping P2P tests")
        return False
        
    device_count = torch.cuda.device_count()
    if device_count < 2:
        print("Need at least 2 GPUs for P2P testing, skipping")
        return False
    
    # Test with first two devices
    active_devices = [0, 1]
    
    try:
        # Initialize P2P backend for device 0 (master)
        backend_0 = TPBackendP2P(
            device=0,
            active_devices=active_devices,
            output_device=0,
            init_method="tcp://localhost:12345",
            master=True,
            uuid="test_p2p"
        )
        
        # Initialize P2P backend for device 1
        backend_1 = TPBackendP2P(
            device=1,
            active_devices=active_devices,
            output_device=0,
            init_method="tcp://localhost:12345",
            master=False,
            uuid="test_p2p"
        )
        
        if not backend_0.use_p2p or not backend_1.use_p2p:
            print("P2P not available, skipping operation tests")
            backend_0.close()
            backend_1.close()
            return False
        
        print("P2P backend initialized successfully")
        
        # Test broadcast operation
        print("Testing broadcast operation...")
        test_tensor = torch.randn(100, 100, device=0)
        backend_0.broadcast(test_tensor, 0)
        print("Broadcast completed")
        
        # Test all_reduce operation
        print("Testing all_reduce operation...")
        test_tensor = torch.randn(100, 100, device=0)
        backend_0.all_reduce(test_tensor)
        print("All_reduce completed")
        
        # Test gather operation
        print("Testing gather operation...")
        test_tensor = torch.randn(100, 100, device=0)
        out_tensor = torch.zeros(200, 100, device=0)
        gather_devices = torch.tensor([0, 1], dtype=torch.int)
        ldims = [100, 100]
        backend_0.gather(test_tensor, out_tensor, gather_devices, 0, ldims)
        print("Gather completed")
        
        # Test barrier
        print("Testing barrier...")
        backend_0.fwd_barrier()
        print("Barrier completed")
        
        # Cleanup
        backend_0.close()
        backend_1.close()
        print("P2P backend closed successfully")
        
        return True
    except Exception as e:
        print(f"P2P backend test failed: {e}")
        return False

def main():
    """Main test function."""
    print("ExLlamaV3 P2P Operations Test")
    print("=" * 40)
    
    # Test P2P topology detection
    topology_available = test_p2p_topology_detection()
    
    if topology_available:
        # Test P2P backend
        backend_success = test_p2p_backend()
        
        if backend_success:
            print("\nAll P2P tests passed!")
            return 0
        else:
            print("\nP2P backend tests failed!")
            return 1
    else:
        print("\nP2P not available, skipping tests")
        return 0

if __name__ == "__main__":
    sys.exit(main())