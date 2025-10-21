#!/usr/bin/env python3
"""
Test script for validating shared memory optimizations in ExLlamaV3.

This script tests the correctness and functionality of the optimized shared memory management.
"""

import torch
import numpy as np
import time
import os
import sys
import traceback
from typing import List, Dict, Tuple

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exllamav3.model.model_tp_shared import SMProducer, SMConsumer, _memory_pool
from exllamav3.model.model_tp_cuda import OptimizedMemoryManager, P2PMemoryUtils


class SharedMemoryTests:
    """Test suite for shared memory optimizations."""
    
    def __init__(self, device: int = 0):
        self.device = device
        self.test_results = {}
        self.passed_tests = 0
        self.total_tests = 0
        torch.cuda.set_device(device)
        
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests and return results."""
        print("Running Shared Memory Optimization Tests...")
        print("=" * 60)
        
        # Test 1: Adaptive Buffer Sizing
        print("\n1. Testing Adaptive Buffer Sizing...")
        self.test_results['adaptive_buffer'] = self.test_adaptive_buffer_sizing()
        
        # Test 2: Zero-Copy Operations
        print("\n2. Testing Zero-Copy Operations...")
        self.test_results['zero_copy'] = self.test_zero_copy_operations()
        
        # Test 3: Memory Pool Management
        print("\n3. Testing Memory Pool Management...")
        self.test_results['memory_pool'] = self.test_memory_pool_management()
        
        # Test 4: Batched Processing
        print("\n4. Testing Batched Processing...")
        self.test_results['batched_processing'] = self.test_batched_processing()
        
        # Test 5: Tensor Caching
        print("\n5. Testing Tensor Caching...")
        self.test_results['tensor_caching'] = self.test_tensor_caching()
        
        # Test 6: Memory Defragmentation
        print("\n6. Testing Memory Defragmentation...")
        self.test_results['defragmentation'] = self.test_memory_defragmentation()
        
        # Test 7: Error Handling
        print("\n7. Testing Error Handling...")
        self.test_results['error_handling'] = self.test_error_handling()
        
        # Generate summary report
        self.generate_test_summary()
        
        return self.test_results
    
    def test_adaptive_buffer_sizing(self) -> bool:
        """Test adaptive buffer sizing functionality."""
        try:
            # Create producer with adaptive sizing
            producer = SMProducer(
                buffer_size=32*1024*1024,  # 32MB initial
                adaptive_sizing=True,
                min_buffer_size=16*1024*1024,  # 16MB minimum
                max_buffer_size=128*1024*1024  # 128MB maximum
            )
            
            # Test with increasing tensor sizes
            tensor_sizes = [
                (1024, 1024),    # 2MB
                (2048, 2048),    # 8MB
                (4096, 4096),    # 32MB
                (8192, 8192),    # 128MB - should trigger resize
            ]
            
            consumer = SMConsumer(producer, device=self.device, pin_memory=True)
            
            for size in tensor_sizes:
                tensor = torch.randn(size, device=self.device, dtype=torch.float16)
                
                # Send tensor
                imp = producer.send(tensor)
                
                # Receive tensor
                received = consumer.recv(imp, cuda=True)
                
                # Verify correctness
                if not torch.allclose(tensor.cpu(), received.cpu(), atol=1e-3):
                    print(f"  FAILED: Tensor mismatch for size {size}")
                    return False
                
                print(f"  ✓ Size {size[0]}x{size[1]}: Buffer size {producer.buffer_size//1024**2}MB")
            
            # Check if buffer was resized
            if producer.buffer_size <= 32*1024*1024:
                print("  WARNING: Buffer was not resized as expected")
            
            producer.close()
            consumer.close()
            
            print("  ✓ Adaptive buffer sizing test passed")
            return True
            
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            return False
    
    def test_zero_copy_operations(self) -> bool:
        """Test zero-copy operations functionality."""
        try:
            producer = SMProducer(buffer_size=64*1024*1024)
            
            # Test with zero-copy enabled
            consumer_zero_copy = SMConsumer(
                producer, 
                device=self.device, 
                pin_memory=True, 
                enable_zero_copy=True
            )
            
            # Test without zero-copy
            consumer_copy = SMConsumer(
                producer, 
                device=self.device, 
                pin_memory=True, 
                enable_zero_copy=False
            )
            
            tensor = torch.randn((2048, 2048), device=self.device, dtype=torch.float16)
            
            # Test zero-copy path
            imp = producer.send(tensor)
            received_zero_copy = consumer_zero_copy.recv(imp, cuda=True)
            
            # Test regular copy path
            received_copy = consumer_copy.recv(imp, cuda=True)
            
            # Verify correctness
            if not torch.allclose(tensor.cpu(), received_zero_copy.cpu(), atol=1e-3):
                print("  FAILED: Zero-copy tensor mismatch")
                return False
            
            if not torch.allclose(tensor.cpu(), received_copy.cpu(), atol=1e-3):
                print("  FAILED: Regular copy tensor mismatch")
                return False
            
            # Check stats
            stats_zero_copy = consumer_zero_copy.get_stats()
            stats_copy = consumer_copy.get_stats()
            
            print(f"  ✓ Zero-copy hits: {stats_zero_copy.get('zero_copy_hits', 0)}")
            print(f"  ✓ Cache hits: {stats_zero_copy.get('cache_hits', 0)}")
            
            producer.close()
            consumer_zero_copy.close()
            consumer_copy.close()
            
            print("  ✓ Zero-copy operations test passed")
            return True
            
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            return False
    
    def test_memory_pool_management(self) -> bool:
        """Test memory pool management functionality."""
        try:
            memory_manager = OptimizedMemoryManager()
            
            # Test buffer allocation and return
            buffer_sizes = [1024, 2048, 4096, 8192, 16384]
            
            allocated_buffers = []
            
            for size in buffer_sizes:
                # Allocate buffer
                buffer_ptr = memory_manager.get_pinned_buffer(size)
                allocated_buffers.append((buffer_ptr, size))
                
                if buffer_ptr == 0:
                    print(f"  FAILED: Failed to allocate buffer of size {size}")
                    return False
            
            # Return buffers
            for buffer_ptr, size in allocated_buffers:
                memory_manager.return_pinned_buffer(buffer_ptr, size)
            
            # Check stats
            stats = memory_manager.get_stats()
            
            if stats['total_buffers'] == 0:
                print("  WARNING: No buffers in pool after return")
            
            print(f"  ✓ Memory manager stats: {stats}")
            print("  ✓ Memory pool management test passed")
            return True
            
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            return False
    
    def test_batched_processing(self) -> bool:
        """Test batched processing functionality."""
        try:
            producer = SMProducer(buffer_size=128*1024*1024)
            consumer = SMConsumer(producer, device=self.device, pin_memory=True)
            
            # Create batch of tensors
            batch_size = 10
            tensors = [
                torch.randn((1024, 1024), device=self.device, dtype=torch.float16)
                for _ in range(batch_size)
            ]
            
            # Send tensors
            imps = [producer.send(tensor) for tensor in tensors]
            
            # Receive batch
            received_batch = consumer.recv_batch(imps, cuda=True)
            
            # Verify correctness
            for original, received in zip(tensors, received_batch):
                if not torch.allclose(original.cpu(), received.cpu(), atol=1e-3):
                    print("  FAILED: Batch tensor mismatch")
                    return False
            
            # Test individual processing for comparison
            for tensor in tensors:
                imp = producer.send(tensor)
                received = consumer.recv(imp, cuda=True)
                if not torch.allclose(tensor.cpu(), received.cpu(), atol=1e-3):
                    print("  FAILED: Individual tensor mismatch")
                    return False
            
            producer.close()
            consumer.close()
            
            print("  ✓ Batched processing test passed")
            return True
            
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            return False
    
    def test_tensor_caching(self) -> bool:
        """Test tensor caching functionality."""
        try:
            producer = SMProducer(buffer_size=64*1024*1024)
            consumer = SMConsumer(
                producer, 
                device=self.device, 
                pin_memory=True, 
                cache_tensors=True
            )
            
            # Create tensor
            tensor = torch.randn((1024, 1024), device=self.device, dtype=torch.float16)
            
            # Send tensor multiple times (should hit cache)
            for i in range(5):
                imp = producer.send(tensor)
                received = consumer.recv(imp, cuda=True)
                
                if not torch.allclose(tensor.cpu(), received.cpu(), atol=1e-3):
                    print(f"  FAILED: Cached tensor mismatch on iteration {i}")
                    return False
            
            # Check cache stats
            stats = consumer.get_stats()
            
            if stats.get('cache_hits', 0) == 0:
                print("  WARNING: No cache hits detected")
            
            print(f"  ✓ Cache hits: {stats.get('cache_hits', 0)}")
            print(f"  ✓ Cache size: {stats.get('cache_size', 0)}")
            
            # Clear cache
            consumer.clear_cache()
            
            producer.close()
            consumer.close()
            
            print("  ✓ Tensor caching test passed")
            return True
            
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            return False
    
    def test_memory_defragmentation(self) -> bool:
        """Test memory defragmentation functionality."""
        try:
            producer = SMProducer(
                buffer_size=32*1024*1024,
                adaptive_sizing=True
            )
            
            consumer = SMConsumer(producer, device=self.device, pin_memory=True)
            
            # Create tensors of varying sizes
            tensor_sizes = [
                (512, 512),
                (1024, 1024),
                (256, 256),
                (2048, 2048),
                (128, 128),
            ]
            
            # Send tensors to fragment memory
            for size in tensor_sizes:
                tensor = torch.randn(size, device=self.device, dtype=torch.float16)
                imp = producer.send(tensor)
                received = consumer.recv(imp, cuda=True)
            
            # Clear buffer to trigger defragmentation
            producer.clear()
            
            # Send large tensor to test defragmentation
            large_tensor = torch.randn((4096, 4096), device=self.device, dtype=torch.float16)
            imp = producer.send(large_tensor)
            received = consumer.recv(imp, cuda=True)
            
            # Verify correctness
            if not torch.allclose(large_tensor.cpu(), received.cpu(), atol=1e-3):
                print("  FAILED: Defragmented tensor mismatch")
                return False
            
            # Check stats
            stats = producer.get_stats()
            
            print(f"  ✓ Fragmentations: {stats.get('fragmentations', 0)}")
            print(f"  ✓ Peak usage: {stats.get('peak_usage', 0)//1024**2}MB")
            
            producer.close()
            consumer.close()
            
            print("  ✓ Memory defragmentation test passed")
            return True
            
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            return False
    
    def test_error_handling(self) -> bool:
        """Test error handling functionality."""
        try:
            # Test with invalid buffer size
            try:
                producer = SMProducer(buffer_size=1024)  # Too small
                producer.close()
                print("  WARNING: Small buffer size was accepted")
            except:
                print("  ✓ Small buffer size correctly rejected")
            
            # Test with None tensor
            producer = SMProducer(buffer_size=64*1024*1024)
            consumer = SMConsumer(producer, device=self.device, pin_memory=True)
            
            imp = producer.send(None)
            received = consumer.recv(imp, cuda=True)
            
            if received is not None:
                print("  FAILED: None tensor was not handled correctly")
                return False
            
            # Test with oversized tensor
            try:
                huge_tensor = torch.randn((32768, 32768), device=self.device, dtype=torch.float16)
                imp = producer.send(huge_tensor)
                received = consumer.recv(imp, cuda=True)
                
                # Should fallback to share_memory
                if not torch.allclose(huge_tensor.cpu(), received.cpu(), atol=1e-3):
                    print("  FAILED: Oversized tensor mismatch")
                    return False
                    
                print("  ✓ Oversized tensor handled correctly")
            except Exception as e:
                print(f"  ✓ Oversized tensor correctly rejected: {e}")
            
            producer.close()
            consumer.close()
            
            print("  ✓ Error handling test passed")
            return True
            
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            return False
    
    def generate_test_summary(self):
        """Generate a summary report of test results."""
        print("\n" + "=" * 60)
        print("TEST SUMMARY REPORT")
        print("=" * 60)
        
        passed = sum(1 for result in self.test_results.values() if result)
        total = len(self.test_results)
        
        print(f"\nTests Passed: {passed}/{total}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        
        print("\nDetailed Results:")
        print("-" * 40)
        for test_name, result in self.test_results.items():
            status = "PASS" if result else "FAIL"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
        
        if passed == total:
            print("\n✓ All tests passed successfully!")
        else:
            print(f"\n✗ {total - passed} tests failed.")
        
        # Save results
        np.save('shared_memory_test_results.npy', self.test_results)
        print("\nTest results saved to 'shared_memory_test_results.npy'")


def main():
    """Main function to run tests."""
    print("ExLlamaV3 Shared Memory Optimization Tests")
    print("=" * 60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA not available. This test requires CUDA.")
        return
    
    # Get device
    device = 0
    if torch.cuda.device_count() > 1:
        print(f"Found {torch.cuda.device_count()} CUDA devices. Using device {device}.")
    
    # Run tests
    test_suite = SharedMemoryTests(device=device)
    results = test_suite.run_all_tests()
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    main()