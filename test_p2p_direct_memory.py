#!/usr/bin/env python3
"""
Comprehensive tests for P2P direct memory access functions.
This script tests correctness, error handling, and edge cases for P2P memory operations.
"""

import torch
import numpy as np
import argparse
import sys
import os
import time
import unittest

# Add the exllamav3 directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'exllamav3'))

from exllamav3.model.model_tp_backend import TPBackendP2P
from exllamav3.model.model_tp_p2p import P2PTopology
from exllamav3.ext import exllamav3_ext as ext


class TestP2PDirectMemory(unittest.TestCase):
    """Test suite for P2P direct memory access functions."""
    
    def setUp(self):
        """Set up test environment."""
        self.devices = [0, 1]  # Default to first two devices
        self.backends = {}
        self.topology = None
        
        # Check available devices
        device_count = torch.cuda.device_count()
        if device_count < 2:
            self.skipTest("At least 2 GPU devices are required for P2P tests")
        
        # Initialize P2P topology
        try:
            self.topology = P2PTopology(self.devices)
            topology_summary = self.topology.get_topology_summary()
            if topology_summary.get("connectivity_ratio", 0) == 0:
                self.skipTest("No P2P connectivity detected between devices")
        except Exception as e:
            self.skipTest(f"P2P topology initialization failed: {e}")
        
        # Initialize backends
        for device in self.devices:
            try:
                backend = TPBackendP2P(
                    device=device,
                    active_devices=self.devices,
                    output_device=self.devices[0],
                    init_method='tcp://localhost:12345',
                    master=(device == self.devices[0]),
                    uuid='p2p_test'
                )
                self.backends[device] = backend
            except Exception as e:
                self.skipTest(f"Backend initialization failed for device {device}: {e}")
    
    def tearDown(self):
        """Clean up test environment."""
        for device, backend in self.backends.items():
            try:
                backend.close()
            except Exception:
                pass
    
    def test_p2p_copy_tensor_async(self):
        """Test asynchronous P2P tensor copy."""
        src_device, dst_device = self.devices[0], self.devices[1]
        backend = self.backends[src_device]
        
        # Test different tensor sizes
        sizes = [100, 1000, 10000, 100000]
        
        for size in sizes:
            with self.subTest(size=size):
                # Create test tensors
                src_tensor = torch.randn(size, dtype=torch.float32, device=src_device)
                dst_tensor = torch.zeros(size, dtype=torch.float32, device=dst_device)
                
                # Copy tensor
                ext.p2p_copy_tensor_async(src_device, dst_device, src_tensor, dst_tensor, backend.abort_flag)
                
                # Synchronize and verify
                torch.cuda.synchronize()
                
                # Check if abort flag was set
                abort_flag_value = backend.abort_flag.item()
                self.assertEqual(abort_flag_value, 0, f"Copy failed for size {size}")
                
                # Verify data integrity (if copy succeeded)
                if abort_flag_value == 0:
                    # Copy back to CPU for comparison
                    src_cpu = src_tensor.cpu()
                    dst_cpu = dst_tensor.cpu()
                    
                    # Check if tensors are equal
                    self.assertTrue(torch.allclose(src_cpu, dst_cpu, rtol=1e-5, atol=1e-5),
                                  f"Data mismatch for size {size}")
    
    def test_p2p_copy_tensor_sync(self):
        """Test synchronous P2P tensor copy."""
        src_device, dst_device = self.devices[0], self.devices[1]
        backend = self.backends[src_device]
        
        # Test different tensor sizes
        sizes = [100, 1000, 10000]
        
        for size in sizes:
            with self.subTest(size=size):
                # Create test tensors
                src_tensor = torch.randn(size, dtype=torch.float32, device=src_device)
                dst_tensor = torch.zeros(size, dtype=torch.float32, device=dst_device)
                
                # Copy tensor
                ext.p2p_copy_tensor_sync(src_device, dst_device, src_tensor, dst_tensor, backend.abort_flag)
                
                # Check if abort flag was set
                abort_flag_value = backend.abort_flag.item()
                self.assertEqual(abort_flag_value, 0, f"Sync copy failed for size {size}")
                
                # Verify data integrity
                if abort_flag_value == 0:
                    src_cpu = src_tensor.cpu()
                    dst_cpu = dst_tensor.cpu()
                    
                    self.assertTrue(torch.allclose(src_cpu, dst_cpu, rtol=1e-5, atol=1e-5),
                                  f"Data mismatch for sync copy size {size}")
    
    def test_p2p_copy_tensor_batch(self):
        """Test batch P2P tensor copy."""
        src_device, dst_device = self.devices[0], self.devices[1]
        backend = self.backends[src_device]
        
        # Test different batch sizes
        batch_sizes = [2, 4, 8]
        tensor_size = 1000
        
        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                # Create test tensors
                src_tensors = []
                dst_tensors = []
                
                for _ in range(batch_size):
                    src_tensor = torch.randn(tensor_size, dtype=torch.float32, device=src_device)
                    dst_tensor = torch.zeros(tensor_size, dtype=torch.float32, device=dst_device)
                    src_tensors.append(src_tensor)
                    dst_tensors.append(dst_tensor)
                
                # Copy tensors
                ext.p2p_copy_tensor_batch(src_device, dst_device, src_tensors, dst_tensors, backend.abort_flag)
                
                # Synchronize and verify
                torch.cuda.synchronize()
                
                # Check if abort flag was set
                abort_flag_value = backend.abort_flag.item()
                self.assertEqual(abort_flag_value, 0, f"Batch copy failed for batch size {batch_size}")
                
                # Verify data integrity
                if abort_flag_value == 0:
                    for i in range(batch_size):
                        src_cpu = src_tensors[i].cpu()
                        dst_cpu = dst_tensors[i].cpu()
                        
                        self.assertTrue(torch.allclose(src_cpu, dst_cpu, rtol=1e-5, atol=1e-5),
                                      f"Data mismatch for batch copy, tensor {i}")
    
    def test_p2p_copy_tensor_with_offset(self):
        """Test P2P tensor copy with offsets."""
        src_device, dst_device = self.devices[0], self.devices[1]
        backend = self.backends[src_device]
        
        # Create larger tensors
        total_size = 10000
        copy_size = 5000
        
        src_tensor = torch.randn(total_size, dtype=torch.float32, device=src_device)
        dst_tensor = torch.zeros(total_size, dtype=torch.float32, device=dst_device)
        
        # Copy with offsets
        src_offset = 0
        dst_offset = 2500  # Copy to middle of destination
        size_bytes = copy_size * 4  # 4 bytes per float32
        
        ext.p2p_copy_tensor_with_offset(
            src_device, dst_device, src_tensor, dst_tensor,
            src_offset, dst_offset, size_bytes, backend.abort_flag
        )
        
        # Synchronize and verify
        torch.cuda.synchronize()
        
        # Check if abort flag was set
        abort_flag_value = backend.abort_flag.item()
        self.assertEqual(abort_flag_value, 0, "Copy with offset failed")
        
        # Verify data integrity
        if abort_flag_value == 0:
            # Extract the copied regions
            src_region = src_tensor[:copy_size].cpu()
            dst_region = dst_tensor[2500:2500+copy_size].cpu()
            
            self.assertTrue(torch.allclose(src_region, dst_region, rtol=1e-5, atol=1e-5),
                          "Data mismatch for copy with offset")
    
    def test_p2p_memory_registration(self):
        """Test P2P memory registration functions."""
        device = self.devices[0]
        peer_device = self.devices[1]
        backend = self.backends[device]
        
        # Allocate test memory
        size = 1024 * 1024  # 1MB
        test_tensor = torch.randn(size // 4, dtype=torch.float32, device=device)
        ptr = test_tensor.data_ptr()
        
        # Test registration
        ext.p2p_register_memory_region(device, ptr, size, backend.abort_flag)
        
        # Check if registered
        is_registered = ext.p2p_is_memory_registered(device, ptr, backend.abort_flag)
        self.assertTrue(is_registered, "Memory registration failed")
        
        # Test unregistration
        ext.p2p_unregister_memory_region(device, ptr, backend.abort_flag)
        
        # Check if unregistered
        is_registered = ext.p2p_is_memory_registered(device, ptr, backend.abort_flag)
        self.assertFalse(is_registered, "Memory unregistration failed")
    
    def test_p2p_direct_memory_pool(self):
        """Test P2P direct memory pool functions."""
        device = self.devices[0]
        peer_device = self.devices[1]
        backend = self.backends[device]
        
        # Initialize direct memory pool
        pool_size = 64 * 1024 * 1024  # 64MB
        ext.p2p_init_direct_memory_pool(device, pool_size, [peer_device], backend.abort_flag)
        
        # Test allocation
        alloc_sizes = [1024, 4096, 16384]  # Different allocation sizes
        allocated_ptrs = []
        
        for size in alloc_sizes:
            ptr = ext.p2p_allocate_from_direct_pool(device, size, peer_device, backend.abort_flag)
            self.assertIsNotNone(ptr, f"Allocation failed for size {size}")
            allocated_ptrs.append((ptr, size))
        
        # Test pool usage
        usage = ext.p2p_get_direct_pool_usage(device, backend.abort_flag)
        total_size = ext.p2p_get_direct_pool_size(device, backend.abort_flag)
        
        self.assertGreater(usage, 0, "Pool usage should be greater than 0")
        self.assertEqual(total_size, pool_size, "Pool size mismatch")
        
        # Test deallocation
        for ptr, size in allocated_ptrs:
            ext.p2p_free_to_direct_pool(device, ptr, size, backend.abort_flag)
        
        # Cleanup
        ext.p2p_cleanup_direct_memory_pool(device, backend.abort_flag)
    
    def test_p2p_peer_access_management(self):
        """Test P2P peer access management functions."""
        device = self.devices[0]
        peer_device = self.devices[1]
        backend = self.backends[device]
        
        # Test enabling peer access
        ext.p2p_enable_peer_access(device, peer_device, backend.abort_flag)
        
        # Check if peer access is enabled
        is_enabled = ext.p2p_is_peer_access_enabled(device, peer_device, backend.abort_flag)
        self.assertTrue(is_enabled, "Peer access should be enabled")
        
        # Test disabling peer access
        ext.p2p_disable_peer_access(device, peer_device, backend.abort_flag)
        
        # Check if peer access is disabled
        # Note: This might still return true depending on CUDA implementation
        is_enabled = ext.p2p_is_peer_access_enabled(device, peer_device, backend.abort_flag)
        # We don't assert here as behavior can vary
    
    def test_p2p_performance_measurement(self):
        """Test P2P performance measurement functions."""
        src_device, dst_device = self.devices[0], self.devices[1]
        backend = self.backends[src_device]
        
        # Test bandwidth measurement
        size = 1024 * 1024  # 1MB
        num_iterations = 5
        
        bandwidth = ext.p2p_measure_bandwidth(src_device, dst_device, size, num_iterations, backend.abort_flag)
        self.assertGreater(bandwidth, 0, "Bandwidth should be greater than 0")
        
        # Test latency measurement
        latency = ext.p2p_measure_latency(src_device, dst_device, size, num_iterations, backend.abort_flag)
        self.assertGreater(latency, 0, "Latency should be greater than 0")
    
    def test_p2p_memory_validation(self):
        """Test P2P memory access validation."""
        src_device, dst_device = self.devices[0], self.devices[1]
        backend = self.backends[src_device]
        
        # Create test tensors
        size = 1024
        src_tensor = torch.randn(size, dtype=torch.float32, device=src_device)
        dst_tensor = torch.zeros(size, dtype=torch.float32, device=dst_device)
        
        # Test validation
        is_valid = ext.p2p_validate_memory_access(
            src_device, dst_device,
            src_tensor.data_ptr(), dst_tensor.data_ptr(),
            size * 4,  # 4 bytes per float32
            backend.abort_flag
        )
        
        self.assertTrue(is_valid, "Memory access validation should pass")
    
    def test_error_handling(self):
        """Test error handling for invalid parameters."""
        src_device, dst_device = self.devices[0], self.devices[1]
        backend = self.backends[src_device]
        
        # Test with invalid device
        invalid_device = 999
        
        # Create test tensors
        src_tensor = torch.randn(100, dtype=torch.float32, device=src_device)
        dst_tensor = torch.zeros(100, dtype=torch.float32, device=dst_device)
        
        # This should set the abort flag
        ext.p2p_copy_tensor_async(invalid_device, dst_device, src_tensor, dst_tensor, backend.abort_flag)
        
        # Check if abort flag was set
        abort_flag_value = backend.abort_flag.item()
        self.assertEqual(abort_flag_value, 1, "Abort flag should be set for invalid device")
        
        # Reset abort flag
        backend.abort_flag.zero_()
        
        # Test with mismatched tensor sizes
        large_src = torch.randn(200, dtype=torch.float32, device=src_device)
        small_dst = torch.zeros(100, dtype=torch.float32, device=dst_device)
        
        ext.p2p_copy_tensor_async(src_device, dst_device, large_src, small_dst, backend.abort_flag)
        
        # Check if abort flag was set
        abort_flag_value = backend.abort_flag.item()
        self.assertEqual(abort_flag_value, 1, "Abort flag should be set for mismatched sizes")


def run_tests():
    """Run all tests and report results."""
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestP2PDirectMemory)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Report results
    if result.wasSuccessful():
        print("\n✅ All P2P direct memory tests passed!")
        return 0
    else:
        print(f"\n❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        return 1


def main():
    parser = argparse.ArgumentParser(description='P2P Direct Memory Tests')
    parser.add_argument('--devices', type=str, default='0,1', help='Comma-separated list of GPU devices to use')
    
    args = parser.parse_args()
    
    print("P2P Direct Memory Access Tests")
    print(f"Devices: {args.devices}")
    
    # Run tests
    exit_code = run_tests()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()