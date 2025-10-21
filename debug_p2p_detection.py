#!/usr/bin/env python3
"""
Debug script to test different P2P detection methods and identify the issue.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'exllamav3'))

import torch
import numpy as np
import ctypes
from exllamav3.model.model_tp_cuda import _cudart

def test_pytorch_p2p_detection(devices):
    """Test PyTorch's P2P detection method."""
    print("\n=== Testing PyTorch P2P Detection ===")
    p2p_matrix = np.zeros((len(devices), len(devices)), dtype=bool)
    
    for i, device_i in enumerate(devices):
        for j, device_j in enumerate(devices):
            if i == j:
                p2p_matrix[i][j] = True
                continue
                
            try:
                can_access = torch.cuda.can_device_access_peer(device_i, device_j)
                p2p_matrix[i][j] = can_access
                print(f"  PyTorch: Device {device_i} can access device {device_j}: {can_access}")
            except Exception as e:
                print(f"  PyTorch: Error checking {device_i} -> {device_j}: {e}")
                p2p_matrix[i][j] = False
    
    return p2p_matrix

def test_cuda_runtime_p2p_detection(devices):
    """Test CUDA runtime API P2P detection method."""
    print("\n=== Testing CUDA Runtime API P2P Detection ===")
    
    try:
        cudart = _cudart()
        
        # Get CUDA API functions
        cuda_device_can_access_peer = cudart.cudaDeviceCanAccessPeer
        cuda_device_can_access_peer.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        cuda_device_can_access_peer.restype = ctypes.c_int
        
        p2p_matrix = np.zeros((len(devices), len(devices)), dtype=bool)
        
        for i, device_i in enumerate(devices):
            for j, device_j in enumerate(devices):
                if i == j:
                    p2p_matrix[i][j] = True
                    continue
                    
                try:
                    can_access = ctypes.c_int()
                    result = cuda_device_can_access_peer(
                        ctypes.c_int(device_i),
                        ctypes.c_int(device_j),
                        ctypes.byref(can_access)
                    )
                    
                    if result == 0:  # CUDA_SUCCESS
                        p2p_matrix[i][j] = (can_access.value == 1)
                        print(f"  CUDA Runtime: Device {device_i} can access device {device_j}: {can_access.value == 1}")
                    else:
                        print(f"  CUDA Runtime: Error checking {device_i} -> {device_j}: CUDA error {result}")
                        p2p_matrix[i][j] = False
                        
                except Exception as e:
                    print(f"  CUDA Runtime: Error checking {device_i} -> {device_j}: {e}")
                    p2p_matrix[i][j] = False
        
        return p2p_matrix
        
    except Exception as e:
        print(f"  CUDA Runtime API initialization failed: {e}")
        return np.zeros((len(devices), len(devices)), dtype=bool)

def test_direct_p2p_access(devices):
    """Test direct P2P access by attempting to enable peer access."""
    print("\n=== Testing Direct P2P Access ===")
    p2p_matrix = np.zeros((len(devices), len(devices)), dtype=bool)
    
    for i, device_i in enumerate(devices):
        for j, device_j in enumerate(devices):
            if i == j:
                p2p_matrix[i][j] = True
                continue
                
            try:
                # Try to enable peer access and test it
                with torch.cuda.device(device_i):
                    # Try to enable peer access
                    result = torch.cuda.device_count()  # Just to ensure CUDA context
                    
                    # Create test tensors
                    test_src = torch.zeros(10, device=device_i)
                    test_dst = torch.zeros(10, device=device_j)
                    
                    # Try direct copy
                    try:
                        # This will fail if P2P is not available
                        test_dst.copy_(test_src, non_blocking=True)
                        torch.cuda.synchronize()
                        p2p_matrix[i][j] = True
                        print(f"  Direct Access: Device {device_i} can access device {device_j}: True")
                    except RuntimeError as e:
                        if "peer" in str(e).lower():
                            p2p_matrix[i][j] = False
                            print(f"  Direct Access: Device {device_i} can access device {device_j}: False ({e})")
                        else:
                            # Some other error, assume P2P might work
                            p2p_matrix[i][j] = True
                            print(f"  Direct Access: Device {device_i} can access device {device_j}: True (other error)")
                    
            except Exception as e:
                print(f"  Direct Access: Error checking {device_i} -> {device_j}: {e}")
                p2p_matrix[i][j] = False
    
    return p2p_matrix

def test_cuda_driver_p2p_detection(devices):
    """Test CUDA driver API P2P detection method."""
    print("\n=== Testing CUDA Driver API P2P Detection ===")
    
    try:
        # Try to use pycuda if available
        import pycuda.driver as drv
        drv.init()
        
        p2p_matrix = np.zeros((len(devices), len(devices)), dtype=bool)
        
        for i, device_i in enumerate(devices):
            for j, device_j in enumerate(devices):
                if i == j:
                    p2p_matrix[i][j] = True
                    continue
                    
                try:
                    can_access = drv.Device(device_i).can_access_peer(drv.Device(device_j))
                    p2p_matrix[i][j] = can_access
                    print(f"  CUDA Driver: Device {device_i} can access device {device_j}: {can_access}")
                except Exception as e:
                    print(f"  CUDA Driver: Error checking {device_i} -> {device_j}: {e}")
                    p2p_matrix[i][j] = False
        
        return p2p_matrix
        
    except ImportError:
        print("  PyCUDA not available, skipping CUDA Driver API test")
        return np.zeros((len(devices), len(devices)), dtype=bool)
    except Exception as e:
        print(f"  CUDA Driver API initialization failed: {e}")
        return np.zeros((len(devices), len(devices)), dtype=bool)

def compare_detection_methods(devices):
    """Compare all P2P detection methods."""
    print("\n=== Comparing P2P Detection Methods ===")
    
    # Get matrices from all methods
    pytorch_matrix = test_pytorch_p2p_detection(devices)
    cuda_runtime_matrix = test_cuda_runtime_p2p_detection(devices)
    direct_access_matrix = test_direct_p2p_access(devices)
    cuda_driver_matrix = test_cuda_driver_p2p_detection(devices)
    
    # Compare results
    print("\n=== Comparison Summary ===")
    for i, device_i in enumerate(devices):
        for j, device_j in enumerate(devices):
            if i == j:
                continue
                
            pytorch_result = pytorch_matrix[i][j]
            cuda_runtime_result = cuda_runtime_matrix[i][j]
            direct_result = direct_access_matrix[i][j]
            cuda_driver_result = cuda_driver_matrix[i][j]
            
            print(f"  {device_i} -> {device_j}:")
            print(f"    PyTorch: {pytorch_result}")
            print(f"    CUDA Runtime: {cuda_runtime_result}")
            print(f"    Direct Access: {direct_result}")
            print(f"    CUDA Driver: {cuda_driver_result}")
            
            # Check for inconsistencies
            results = [pytorch_result, cuda_runtime_result, direct_result, cuda_driver_result]
            if not all(r == results[0] for r in results):
                print(f"    *** INCONSISTENCY DETECTED ***")
    
    return {
        'pytorch': pytorch_matrix,
        'cuda_runtime': cuda_runtime_matrix,
        'direct_access': direct_access_matrix,
        'cuda_driver': cuda_driver_matrix
    }

def main():
    print("P2P Detection Debug Script")
    print("=" * 50)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available, exiting")
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
    
    # Compare all detection methods
    matrices = compare_detection_methods(active_devices)
    
    # Provide recommendations
    print("\n=== Recommendations ===")
    
    # Check which method detected the most P2P connections
    connection_counts = {}
    for method, matrix in matrices.items():
        count = np.sum(matrix) - len(active_devices)  # Subtract diagonal
        connection_counts[method] = count
        print(f"{method}: {count} P2P connections detected")
    
    # Find the method with most connections
    best_method = max(connection_counts, key=connection_counts.get)
    print(f"\nRecommendation: Use '{best_method}' method for P2P detection")
    
    # If PyTorch is not the best method, suggest a fix
    if best_method != 'pytorch':
        print("\nThe P2PTopology class should be updated to use a more reliable detection method.")

if __name__ == "__main__":
    main()