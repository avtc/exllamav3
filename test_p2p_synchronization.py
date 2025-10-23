#!/usr/bin/env python3
"""
Test for the sophisticated P2P synchronization mechanism.
"""

import torch
import sys
import os
import time

# Add the exllamav3 directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'exllamav3'))

from exllamav3.model.model_tp_backend import TPBackendP2P
import exllamav3_ext as ext

def test_p2p_synchronization():
    """Test the sophisticated P2P synchronization mechanism."""
    print("P2P Synchronization Test")
    
    # Check available devices
    device_count = torch.cuda.device_count()
    print(f"Available CUDA devices: {device_count}")
    
    if device_count < 2:
        print("Error: At least 2 GPU devices are required for P2P synchronization test")
        return
    
    devices = [0, 1]
    
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
                uuid='p2p_sync_test'
            )
            backends[device] = backend
            print(f"Initialized P2P backend for device {device}")
        except Exception as e:
            print(f"Error initializing backend for device {device}: {e}")
            return
    
    try:
        # Test 1: Single device barrier (should work with simple sync)
        print("\nTest 1: Single device barrier...")
        start_time = time.time()
        ext.p2p_device_barrier([0], 0, backends[0].abort_flag)
        end_time = time.time()
        print(f"Single device barrier completed in {end_time - start_time:.4f} seconds")
        
        # Test 2: Two device barrier with P2P
        print("\nTest 2: Two device barrier with P2P...")
        start_time = time.time()
        ext.p2p_device_barrier([0, 1], 0, backends[0].abort_flag)
        end_time = time.time()
        print(f"Two device barrier completed in {end_time - start_time:.4f} seconds")
        
        # Test 3: Multiple barriers to test consistency
        print("\nTest 3: Multiple barriers...")
        for i in range(5):
            start_time = time.time()
            ext.p2p_device_barrier([0, 1], 0, backends[0].abort_flag)
            end_time = time.time()
            print(f"Barrier {i+1} completed in {end_time - start_time:.4f} seconds")
        
        # Test 4: Test with different device ordering
        print("\nTest 4: Different device ordering...")
        start_time = time.time()
        ext.p2p_device_barrier([1, 0], 1, backends[1].abort_flag)
        end_time = time.time()
        print(f"Reordered barrier completed in {end_time - start_time:.4f} seconds")
        
        # Test 5: Test with more devices if available
        if device_count >= 3:
            print("\nTest 5: Three device barrier...")
            devices_3 = [0, 1, 2]
            backends_3 = {}
            
            # Initialize third device backend
            try:
                backend_3 = TPBackendP2P(
                    device=2,
                    active_devices=devices_3,
                    output_device=devices_3[0],
                    init_method='tcp://localhost:12345',
                    master=False,
                    uuid='p2p_sync_test_3'
                )
                backends_3[2] = backend_3
                print(f"Initialized P2P backend for device 2")
                
                start_time = time.time()
                ext.p2p_device_barrier([0, 1, 2], 0, backends[0].abort_flag)
                end_time = time.time()
                print(f"Three device barrier completed in {end_time - start_time:.4f} seconds")
                
                # Cleanup third device
                backend_3.close()
                
            except Exception as e:
                print(f"Error testing with 3 devices: {e}")
        
        print("\nP2P synchronization test completed successfully!")
        
    except Exception as e:
        print(f"Error during P2P synchronization test: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup backends
        for device, backend in backends.items():
            try:
                backend.close()
                print(f"Closed P2P backend for device {device}")
            except Exception as e:
                print(f"Error closing backend for device {device}: {e}")

def test_p2p_with_computation():
    """Test P2P synchronization with actual computation."""
    print("\nP2P Synchronization with Computation Test")
    
    device_count = torch.cuda.device_count()
    if device_count < 2:
        print("Skipping computation test - need at least 2 GPUs")
        return
    
    devices = [0, 1]
    
    # Initialize backends
    backends = {}
    for device in devices:
        try:
            backend = TPBackendP2P(
                device=device,
                active_devices=devices,
                output_device=devices[0],
                init_method='tcp://localhost:12345',
                master=(device == devices[0]),
                uuid='p2p_comp_test'
            )
            backends[device] = backend
        except Exception as e:
            print(f"Error initializing backend for device {device}: {e}")
            return
    
    try:
        # Create tensors on different devices
        size = 1024 * 1024  # 1M elements
        tensor0 = torch.randn(size, device=0)
        tensor1 = torch.randn(size, device=1)
        
        print("Performing computations on different devices...")
        
        # Perform computation on device 0
        with torch.cuda.device(0):
            result0 = torch.matmul(tensor0[:512], tensor0[:512].T)
        
        # Perform computation on device 1
        with torch.cuda.device(1):
            result1 = torch.matmul(tensor1[:512], tensor1[:512].T)
        
        print("Synchronizing across devices...")
        start_time = time.time()
        ext.p2p_device_barrier([0, 1], 0, backends[0].abort_flag)
        end_time = time.time()
        
        print(f"Computation sync completed in {end_time - start_time:.4f} seconds")
        print("P2P synchronization with computation test completed successfully!")
        
    except Exception as e:
        print(f"Error during computation test: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        for device, backend in backends.items():
            try:
                backend.close()
            except Exception as e:
                print(f"Error closing backend for device {device}: {e}")

if __name__ == '__main__':
    test_p2p_synchronization()
    test_p2p_with_computation()