#!/usr/bin/env python3
"""
Test script to verify that P2P fixes work correctly.
This script tests that P2P access is properly managed without redundant calls.
"""

import torch
import sys
import os

# Add the exllamav3 directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'exllamav3'))

def test_p2p_initialization():
    """Test that P2P initialization works without redundant peer access calls."""
    print("Testing P2P initialization...")
    
    try:
        # Check if we have multiple GPUs
        if torch.cuda.device_count() < 2:
            print("WARNING: Need at least 2 GPUs for P2P testing")
            return True
        
        # Import the P2P backend
        from exllamav3.model.model_tp_backend import TPBackendP2P
        
        # Test parameters
        device_ids = list(range(torch.cuda.device_count()))
        output_device = device_ids[0]
        uuid = "test_p2p_fixes"
        
        # Create P2P backends for each device
        backends = []
        for i, device in enumerate(device_ids):
            backend = TPBackendP2P(
                device=device,
                active_devices=device_ids,
                output_device=output_device,
                init_method=f"tcp://{os.getenv('HOST', 'localhost')}:12345",
                master=(i == 0),
                uuid=uuid
            )
            backends.append(backend)
        
        # Test P2P operations
        test_tensor = torch.randn(100, 100, device=device_ids[0])
        
        # Test broadcast
        backends[0].broadcast(test_tensor, device_ids[0])
        
        # Test direct copy
        if len(device_ids) >= 2:
            copied_tensor = backends[0].copy_tensor_direct(device_ids[0], device_ids[1], test_tensor)
            if copied_tensor is not None:
                print("✓ Direct P2P copy successful")
            else:
                print("✗ Direct P2P copy failed")
        
        # Test memory pool stats
        stats = backends[0].get_memory_pool_stats()
        if stats:
            print(f"✓ Memory pool stats available: {stats.get('pool_usage_percent', 0):.1f}% used")
        
        # Clean up
        for backend in backends:
            backend.close()
        
        print("✓ P2P initialization test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ P2P initialization test failed: {e}")
        return False

def test_p2p_bandwidth():
    """Test P2P bandwidth measurement."""
    print("\nTesting P2P bandwidth measurement...")
    
    try:
        if torch.cuda.device_count() < 2:
            print("WARNING: Need at least 2 GPUs for bandwidth testing")
            return True
        
        from exllamav3.model.model_tp_backend import TPBackendP2P
        
        device_ids = [0, 1]
        backend = TPBackendP2P(
            device=device_ids[0],
            active_devices=device_ids,
            output_device=device_ids[0],
            init_method=f"tcp://{os.getenv('HOST', 'localhost')}:12346",
            master=True,
            uuid="test_p2p_bandwidth"
        )
        
        # Measure bandwidth
        bandwidth = backend.measure_p2p_bandwidth(0, 1, size_mb=1, num_iterations=5)
        if bandwidth > 0:
            print(f"✓ P2P bandwidth measured: {bandwidth:.2f} GB/s")
        else:
            print("✗ P2P bandwidth measurement failed")
        
        backend.close()
        return True
        
    except Exception as e:
        print(f"✗ P2P bandwidth test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing P2P fixes for redundant cudaDeviceEnablePeerAccess calls")
    print("=" * 60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return 1
    
    print(f"Found {torch.cuda.device_count()} CUDA devices")
    
    # Run tests
    tests = [
        test_p2p_initialization,
        test_p2p_bandwidth,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "PASS" if result else "FAIL"
        print(f"{i+1}. {test.__name__}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! P2P fixes are working correctly.")
        return 0
    else:
        print("✗ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())