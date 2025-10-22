#!/usr/bin/env python3
"""
Simple script to verify P2P setup and identify the exact issue.
"""

import torch
import sys
import os

# Add the exllamav3 directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'exllamav3'))

from exllamav3.model.model_tp_backend import TPBackendP2P
import exllamav3_ext as ext


def test_p2p_setup():
    """Test P2P setup and verify what's happening."""
    print("P2P Setup Verification")
    print("=" * 50)
    
    # Check available devices
    device_count = torch.cuda.device_count()
    print(f"Available CUDA devices: {device_count}")
    
    if device_count < 2:
        print("Error: At least 2 GPU devices are required for P2P testing")
        return
    
    devices = [0, 1]
    src_device, dst_device = devices[0], devices[1]
    
    # Check CUDA P2P capability first
    print(f"\n=== CUDA P2P Capability Check ===")
    # Use the correct PyTorch method
    can_access = torch.cuda.can_device_access_peer(src_device, dst_device)
    print(f"Device {src_device} can access device {dst_device}: {can_access}")
    
    can_access_reverse = torch.cuda.can_device_access_peer(dst_device, src_device)
    print(f"Device {dst_device} can access device {src_device}: {can_access_reverse}")
    
    if not can_access or not can_access_reverse:
        print("ERROR: P2P is not available between these devices")
        return
    
    # Test direct CUDA P2P without custom backend
    print(f"\n=== Direct CUDA P2P Test ===")
    try:
        torch.cuda.set_device(src_device)
        torch.cuda.enable_peer_access(dst_device)
        print(f"Enabled P2P access from {src_device} to {dst_device}")
        
        torch.cuda.set_device(dst_device)
        torch.cuda.enable_peer_access(src_device)
        print(f"Enabled P2P access from {dst_device} to {src_device}")
        
        # Test a simple transfer
        test_tensor = torch.randn(1000, dtype=torch.float32, device=src_device)
        dst_tensor = torch.zeros(1000, dtype=torch.float32, device=dst_device)
        
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        dst_tensor.copy_(test_tensor, non_blocking=True)
        end_time.record()
        end_time.synchronize()
        
        transfer_time = start_time.elapsed_time(end_time)
        bandwidth = (test_tensor.numel() * test_tensor.element_size()) / (transfer_time / 1000.0) / (1024**3)
        print(f"Direct CUDA P2P transfer: {transfer_time:.3f} ms, {bandwidth:.2f} GB/s")
        
    except Exception as e:
        print(f"Direct CUDA P2P test failed: {e}")
        return
    
    # Now test with custom backend
    print(f"\n=== Custom P2P Backend Test ===")
    backends = {}
    
    for device in devices:
        try:
            print(f"Initializing backend for device {device}...")
            backend = TPBackendP2P(
                device=device,
                active_devices=devices,
                output_device=devices[0],
                init_method='tcp://localhost:12345',
                master=(device == devices[0]),
                uuid='p2p_verification'
            )
            backends[device] = backend
            print(f"Backend initialized for device {device}")
            
            # Check if P2P is working from this device
            other_device = devices[1] if device == devices[0] else devices[0]
            can_access_custom = ext.p2p_can_access_peer_direct(device, other_device, backend.abort_flag)
            print(f"Device {device} can access device {other_device}: {can_access_custom}")
            
        except Exception as e:
            print(f"Error initializing backend for device {device}: {e}")
            return
    
    try:
        # Test custom P2P transfer
        print(f"\n=== Custom P2P Transfer Test ===")
        test_tensor = torch.randn(1000, dtype=torch.float32, device=src_device)
        dst_tensor = torch.zeros(1000, dtype=torch.float32, device=dst_device)
        
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        ext.p2p_copy_tensor_async(src_device, dst_device, test_tensor, dst_tensor, backends[src_device].abort_flag)
        end_time.record()
        end_time.synchronize()
        
        transfer_time = start_time.elapsed_time(end_time)
        bandwidth = (test_tensor.numel() * test_tensor.element_size()) / (transfer_time / 1000.0) / (1024**3)
        print(f"Custom P2P transfer: {transfer_time:.3f} ms, {bandwidth:.2f} GB/s")
        
        # Compare results
        print(f"\n=== Comparison ===")
        print(f"Direct CUDA P2P: {bandwidth:.2f} GB/s")
        print(f"Custom P2P: {bandwidth:.2f} GB/s")
        
        if bandwidth < 1.0:
            print("WARNING: Custom P2P is very slow - likely not using true P2P!")
        elif bandwidth < 5.0:
            print("WARNING: Custom P2P is slower than expected")
        else:
            print("Custom P2P performance looks good")
        
        # Check memory pool stats
        print(f"\n=== Memory Pool Stats ===")
        for device in devices:
            try:
                stats = backends[device].get_memory_pool_stats()
                if stats:
                    print(f"Device {device}: pool usage {stats.get('pool_usage_percent', 0):.1f}%")
            except Exception as e:
                print(f"Error getting stats for device {device}: {e}")
    
    finally:
        # Cleanup
        for device, backend in backends.items():
            try:
                backend.close()
                print(f"Closed backend for device {device}")
            except Exception as e:
                print(f"Error closing backend for device {device}: {e}")


if __name__ == '__main__':
    test_p2p_setup()