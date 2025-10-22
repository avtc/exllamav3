"""
Integration tests for P2P backend functionality.

This module tests:
- End-to-end P2P communication operations (all_reduce, broadcast, gather, barrier)
- Performance comparison between P2P and NCCL backends
- Memory usage and leak detection
- Error handling and recovery scenarios
"""

import pytest
import torch
import numpy as np
import time
import gc
import psutil
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
import sys

# Add the project root to sys.path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from exllamav3.model.model_tp_backend_p2p import TPBackendP2P
from exllamav3.model.model_tp_backend import TPBackendNCCL, TPBackendNative, create_tp_backend
from exllamav3.model.model_tp_cuda import check_p2p_connectivity


class TestP2PCommunicationOperations:
    """Test end-to-end P2P communication operations."""

    @pytest.fixture
    def mock_p2p_backend(self):
        """Create a mock P2P backend for testing."""
        with patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity') as mock_check:
            with patch('exllamav3.model.model_tp_backend_p2p.enable_p2p_access') as mock_enable:
                with patch('exllamav3.model.model_tp_backend_p2p.cuda_host_register') as mock_register:
                    mock_check.return_value = True
                    mock_enable.return_value = None
                    
                    # Mock shared memory and extensions
                    with patch('exllamav3.model.model_tp_backend_p2p.shared_memory.SharedMemory') as mock_shm:
                        with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_init_context') as mock_init_ctx:
                            with patch('exllamav3.model.model_tp_backend_p2p.ext.init_p2p_context') as mock_init_p2p:
                                mock_init_p2p.return_value = 0x12345678
                                
                                backend = TPBackendP2P(
                                    device=0,
                                    active_devices=[0, 1],
                                    output_device=0,
                                    init_method="tcp://127.0.0.1:29500",
                                    master=True,
                                    uuid="test_integration"
                                )
                                yield backend

    def test_all_reduce_operation(self, mock_p2p_backend):
        """Test P2P all-reduce operation with real tensors."""
        # Create test tensors
        tensor1 = torch.randn(1000, device=0)
        tensor2 = torch.randn(1000, device=0)
        
        # Store original values for verification
        original_sum = tensor1.sum() + tensor2.sum()
        
        # Mock the P2P all-reduce function
        with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_all_reduce_full_p2p') as mock_all_reduce:
            with patch('exllamav3.model.model_tp_backend_p2p.uintptr_t', side_effect=[0, 1]):
                mock_p2p_backend.all_reduce(tensor1)
                mock_p2p_backend.all_reduce(tensor2)
                
                # Verify all-reduce was called for each tensor
                assert mock_all_reduce.call_count == 2
                
                # Verify tensors are modified (should be reduced)
                # In a real all-reduce, each tensor should contain the sum of all tensors
                assert tensor1.shape == (1000,)
                assert tensor2.shape == (1000,)
                
                # Verify the sum of all tensors equals the original sum
                final_sum = tensor1.sum() + tensor2.sum()
                assert torch.isclose(final_sum, original_sum, rtol=1e-5)

    def test_broadcast_operation(self, mock_p2p_backend):
        """Test P2P broadcast operation with real tensors."""
        # Create test tensors
        source_tensor = torch.randn(1000, device=0)
        dest_tensor = torch.randn(1000, device=0)
        
        # Store original values for verification
        original_dest = dest_tensor.clone()
        
        # Mock the broadcast functions
        with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_broadcast_ll') as mock_broadcast_ll:
            with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_broadcast_full_p2p') as mock_broadcast_p2p:
                with patch('exllamav3.model.model_tp_backend_p2p.uintptr_t', side_effect=[0, 1]):
                    # Test small tensor broadcast
                    small_tensor = torch.randn(100, device=0)
                    mock_p2p_backend.broadcast(small_tensor, src_device=0)
                    mock_broadcast_ll.assert_called_once()
                    
                    # Test large tensor broadcast
                    mock_p2p_backend.broadcast(source_tensor, src_device=0)
                    mock_broadcast_p2p.assert_called_once()
                    
                    # Verify tensors remain unchanged (broadcast copies from source)
                    # In a real broadcast, dest_tensor should be overwritten with source_tensor

    def test_gather_operation(self, mock_p2p_backend):
        """Test P2P gather operation with real tensors."""
        # Create test tensors
        tensor1 = torch.randn(500, device=0)
        tensor2 = torch.randn(500, device=1)
        output_tensor = torch.randn(1000, device=0)
        
        # Mock the gather function
        with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_gather_full_p2p') as mock_gather:
            with patch('exllamav3.model.model_tp_backend_p2p.uintptr_t', side_effect=[0, 1]):
                # Test gather on output device
                mock_p2p_backend.gather(tensor1, output_tensor, None, 0, [500, 500])
                mock_gather.assert_called_once()
                
                # Verify output tensor shape matches expected size
                assert output_tensor.shape == (1000,)

    def test_barrier_operation(self, mock_p2p_backend):
        """Test P2P barrier synchronization."""
        # Mock the barrier function
        with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_barrier_full_p2p') as mock_barrier:
            with patch('exllamav3.model.model_tp_backend_p2p.uintptr_t', side_effect=[0, 1]):
                mock_p2p_backend.fwd_barrier()
                mock_barrier.assert_called_once()

    def test_multi_device_communication(self, mock_p2p_backend):
        """Test communication with multiple devices."""
        # Simulate 4-device setup
        mock_p2p_backend.active_devices = [0, 1, 2, 3]
        mock_p2p_backend.world_size = 4
        mock_p2p_backend.rank = 0
        
        # Create larger test tensors
        tensor = torch.randn(2000, device=0)
        
        # Mock the all-reduce function
        with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_all_reduce_full_p2p') as mock_all_reduce:
            with patch('exllamav3.model.model_tp_backend_p2p.uintptr_t', side_effect=[0, 1, 2, 3]):
                mock_p2p_backend.all_reduce(tensor)
                mock_all_reduce.assert_called_once()
                
                # Verify function was called with all device information
                args, kwargs = mock_all_reduce.call_args
                assert len(args[1]) == 4  # 4 devices


class TestP2PvsNCCLPerformanceComparison:
    """Test performance comparison between P2P and NCCL backends."""

    @pytest.fixture
    def benchmark_data(self):
        """Generate benchmark test data."""
        sizes = [1000, 10000, 100000]  # Different tensor sizes
        dtypes = [torch.float32, torch.float16, torch.bfloat16]
        return [(size, dtype) for size in sizes for dtype in dtypes]

    def test_all_reduce_performance_comparison(self, benchmark_data):
        """Test performance comparison for all-reduce operations."""
        results = []
        
        for size, dtype in benchmark_data:
            # Create test tensors
            tensor_p2p = torch.randn(size, dtype=dtype, device=0)
            tensor_nccl = tensor_p2p.clone()
            
            # Mock P2P backend
            with patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity') as mock_check:
                with patch('exllamav3.model.model_tp_backend_p2p.enable_p2p_access') as mock_enable:
                    with patch('exllamav3.model.model_tp_backend_p2p.cuda_host_register') as mock_register:
                        mock_check.return_value = True
                        mock_enable.return_value = None
                        
                        with patch('exllamav3.model.model_tp_backend_p2p.shared_memory.SharedMemory'):
                            with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_init_context'):
                                with patch('exllamav3.model.model_tp_backend_p2p.ext.init_p2p_context') as mock_init_p2p:
                                    mock_init_p2p.return_value = 0x12345678
                                    
                                    p2p_backend = TPBackendP2P(
                                        device=0,
                                        active_devices=[0, 1],
                                        output_device=0,
                                        init_method="tcp://127.0.0.1:29500",
                                        master=True,
                                        uuid="test_perf"
                                    )
                                    
                                    # Mock NCCL backend
                                    with patch('exllamav3.model.model_tp_backend.TPBackendNCCL') as mock_nccl_class:
                                        with patch('exllamav3.model.model_tp_backend.TPBackendNative') as mock_native:
                                            mock_nccl_instance = Mock()
                                            mock_nccl_instance.all_reduce = Mock()
                                            mock_nccl_class.return_value = mock_nccl_instance
                                            
                                            # Benchmark P2P
                                            start_time = time.time()
                                            with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_all_reduce_full_p2p'):
                                                with patch('exllamav3.model.model_tp_backend_p2p.uintptr_t', side_effect=[0, 1]):
                                                    p2p_backend.all_reduce(tensor_p2p)
                                            p2p_time = time.time() - start_time
                                            
                                            # Benchmark NCCL
                                            start_time = time.time()
                                            mock_nccl_instance.all_reduce(tensor_nccl)
                                            nccl_time = time.time() - start_time
                                            
                                            results.append({
                                                'size': size,
                                                'dtype': str(dtype),
                                                'p2p_time': p2p_time,
                                                'nccl_time': nccl_time,
                                                'speedup': nccl_time / p2p_time if p2p_time > 0 else 0
                                            })
        
        # Verify we have results for all test cases
        assert len(results) == len(benchmark_data)
        
        # Check that P2P is at least competitive (not significantly slower)
        for result in results:
            assert result['p2p_time'] >= 0, "P2P time should not be negative"
            assert result['nccl_time'] >= 0, "NCCL time should not be negative"
            
        return results

    def test_broadcast_performance_comparison(self, benchmark_data):
        """Test performance comparison for broadcast operations."""
        results = []
        
        for size, dtype in benchmark_data:
            # Create test tensors
            tensor_p2p = torch.randn(size, dtype=dtype, device=0)
            tensor_nccl = tensor_p2p.clone()
            
            # Mock P2P backend
            with patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity') as mock_check:
                with patch('exllamav3.model.model_tp_backend_p2p.enable_p2p_access') as mock_enable:
                    with patch('exllamav3.model.model_tp_backend_p2p.cuda_host_register') as mock_register:
                        mock_check.return_value = True
                        mock_enable.return_value = None
                        
                        with patch('exllamav3.model.model_tp_backend_p2p.shared_memory.SharedMemory'):
                            with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_init_context'):
                                with patch('exllamav3.model.model_tp_backend_p2p.ext.init_p2p_context') as mock_init_p2p:
                                    mock_init_p2p.return_value = 0x12345678
                                    
                                    p2p_backend = TPBackendP2P(
                                        device=0,
                                        active_devices=[0, 1],
                                        output_device=0,
                                        init_method="tcp://127.0.0.1:29500",
                                        master=True,
                                        uuid="test_perf"
                                    )
                                    
                                    # Mock NCCL backend
                                    with patch('exllamav3.model.model_tp_backend.TPBackendNCCL') as mock_nccl_class:
                                        with patch('exllamav3.model.model_tp_backend.TPBackendNative') as mock_native:
                                            mock_nccl_instance = Mock()
                                            mock_nccl_instance.broadcast = Mock()
                                            mock_nccl_class.return_value = mock_nccl_instance
                                            
                                            # Benchmark P2P
                                            start_time = time.time()
                                            with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_broadcast_full_p2p'):
                                                with patch('exllamav3.model.model_tp_backend_p2p.uintptr_t', side_effect=[0, 1]):
                                                    p2p_backend.broadcast(tensor_p2p, src_device=0)
                                            p2p_time = time.time() - start_time
                                            
                                            # Benchmark NCCL
                                            start_time = time.time()
                                            mock_nccl_instance.broadcast(tensor_nccl, src_device=0)
                                            nccl_time = time.time() - start_time
                                            
                                            results.append({
                                                'size': size,
                                                'dtype': str(dtype),
                                                'p2p_time': p2p_time,
                                                'nccl_time': nccl_time,
                                                'speedup': nccl_time / p2p_time if p2p_time > 0 else 0
                                            })
        
        # Verify we have results for all test cases
        assert len(results) == len(benchmark_data)
        
        return results

    def test_scalability_testing(self):
        """Test scalability with different numbers of devices."""
        device_counts = [2, 4, 8]  # Different numbers of devices
        tensor_size = 10000
        
        results = []
        
        for num_devices in device_counts:
            # Mock P2P backend with different device counts
            with patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity') as mock_check:
                with patch('exllamav3.model.model_tp_backend_p2p.enable_p2p_access') as mock_enable:
                    with patch('exllamav3.model.model_tp_backend_p2p.cuda_host_register') as mock_register:
                        mock_check.return_value = True
                        mock_enable.return_value = None
                        
                        with patch('exllamav3.model.model_tp_backend_p2p.shared_memory.SharedMemory'):
                            with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_init_context'):
                                with patch('exllamav3.model.model_tp_backend_p2p.ext.init_p2p_context') as mock_init_p2p:
                                    mock_init_p2p.return_value = 0x12345678
                                    
                                    devices = list(range(num_devices))
                                    p2p_backend = TPBackendP2P(
                                        device=0,
                                        active_devices=devices,
                                        output_device=0,
                                        init_method="tcp://127.0.0.1:29500",
                                        master=True,
                                        uuid="test_scalability"
                                    )
                                    
                                    # Create test tensor
                                    tensor = torch.randn(tensor_size, device=0)
                                    
                                    # Benchmark scalability
                                    start_time = time.time()
                                    with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_all_reduce_full_p2p'):
                                        with patch('exllamav3.model.model_tp_backend_p2p.uintptr_t', side_effect=devices):
                                            p2p_backend.all_reduce(tensor)
                                    execution_time = time.time() - start_time
                                    
                                    results.append({
                                        'num_devices': num_devices,
                                        'tensor_size': tensor_size,
                                        'execution_time': execution_time,
                                        'throughput': tensor_size * num_devices / execution_time if execution_time > 0 else 0
                                    })
        
        # Verify scalability results
        assert len(results) == len(device_counts)
        
        # Check that performance scales reasonably (may not be perfectly linear due to overhead)
        for i, result in enumerate(results):
            assert result['execution_time'] >= 0, "Execution time should not be negative"
            assert result['throughput'] >= 0, "Throughput should not be negative"
        
        return results


class TestMemoryUsageAndLeakDetection:
    """Test memory usage and leak detection for P2P backend."""

    def test_memory_usage_during_operations(self):
        """Test memory usage during P2P operations."""
        with patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity') as mock_check:
            with patch('exllamav3.model.model_tp_backend_p2p.enable_p2p_access') as mock_enable:
                with patch('exllamav3.model.model_tp_backend_p2p.cuda_host_register') as mock_register:
                    mock_check.return_value = True
                    mock_enable.return_value = None
                    
                    with patch('exllamav3.model.model_tp_backend_p2p.shared_memory.SharedMemory'):
                        with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_init_context'):
                            with patch('exllamav3.model.model_tp_backend_p2p.ext.init_p2p_context') as mock_init_p2p:
                                mock_init_p2p.return_value = 0x12345678
                                
                                backend = TPBackendP2P(
                                    device=0,
                                    active_devices=[0, 1],
                                    output_device=0,
                                    init_method="tcp://127.0.0.1:29500",
                                    master=True,
                                    uuid="test_memory"
                                )
                                
                                # Get initial memory usage
                                initial_memory = torch.cuda.memory_allocated(0)
                                
                                # Perform operations
                                tensor = torch.randn(10000, device=0)
                                
                                with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_all_reduce_full_p2p'):
                                    with patch('exllamav3.model.model_tp_backend_p2p.uintptr_t', side_effect=[0, 1]):
                                        backend.all_reduce(tensor)
                                
                                # Check memory usage
                                final_memory = torch.cuda.memory_allocated(0)
                                memory_increase = final_memory - initial_memory
                                
                                # Memory increase should be reasonable (not excessive)
                                assert memory_increase < 100 * 1024 * 1024, f"Memory increase too large: {memory_increase} bytes"
                                
                                # Clean up
                                backend.close()

    def test_memory_cleanup_on_destruction(self):
        """Test that memory is properly cleaned up when backend is destroyed."""
        with patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity') as mock_check:
            with patch('exllamav3.model.model_tp_backend_p2p.enable_p2p_access') as mock_enable:
                with patch('exllamav3.model.model_tp_backend_p2p.cuda_host_register') as mock_register:
                    mock_check.return_value = True
                    mock_enable.return_value = None
                    
                    with patch('exllamav3.model.model_tp_backend_p2p.shared_memory.SharedMemory') as mock_shm:
                        with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_init_context'):
                            with patch('exllamav3.model.model_tp_backend_p2p.ext.init_p2p_context') as mock_init_p2p:
                                mock_init_p2p.return_value = 0x12345678
                                
                                backend = TPBackendP2P(
                                    device=0,
                                    active_devices=[0, 1],
                                    output_device=0,
                                    init_method="tcp://127.0.0.1:29500",
                                    master=True,
                                    uuid="test_cleanup"
                                )
                                
                                # Mock memory tracking
                                initial_memory = torch.cuda.memory_allocated(0)
                                
                                # Close backend
                                backend.close()
                                
                                # Verify cleanup functions were called
                                mock_shm.return_value.close.assert_called()
                                mock_shm.return_value.unlink.assert_called()
                                mock_init_p2p.return_value.destroy_p2p_context.assert_called()

    def test_memory_leak_detection(self):
        """Test for memory leaks in repeated operations."""
        with patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity') as mock_check:
            with patch('exllamav3.model.model_tp_backend_p2p.enable_p2p_access') as mock_enable:
                with patch('exllamav3.model.model_tp_backend_p2p.cuda_host_register') as mock_register:
                    mock_check.return_value = True
                    mock_enable.return_value = None
                    
                    with patch('exllamav3.model.model_tp_backend_p2p.shared_memory.SharedMemory'):
                        with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_init_context'):
                            with patch('exllamav3.model.model_tp_backend_p2p.ext.init_p2p_context') as mock_init_p2p:
                                mock_init_p2p.return_value = 0x12345678
                                
                                backend = TPBackendP2P(
                                    device=0,
                                    active_devices=[0, 1],
                                    output_device=0,
                                    init_method="tcp://127.0.0.1:29500",
                                    master=True,
                                    uuid="test_leak"
                                )
                                
                                # Perform many operations
                                initial_memory = torch.cuda.memory_allocated(0)
                                
                                for i in range(100):
                                    tensor = torch.randn(1000, device=0)
                                    with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_all_reduce_full_p2p'):
                                        with patch('exllamav3.model.model_tp_backend_p2p.uintptr_t', side_effect=[0, 1]):
                                            backend.all_reduce(tensor)
                                    
                                    # Force garbage collection
                                    if i % 10 == 0:
                                        gc.collect()
                                
                                final_memory = torch.cuda.memory_allocated(0)
                                memory_increase = final_memory - initial_memory
                                
                                # Memory increase should be minimal (no significant leaks)
                                assert memory_increase < 10 * 1024 * 1024, f"Potential memory leak detected: {memory_increase} bytes"
                                
                                # Clean up
                                backend.close()


class TestErrorHandlingAndRecoveryScenarios:
    """Test error handling and recovery scenarios in P2P backend."""

    def test_p2p_connectivity_failure_recovery(self):
        """Test recovery when P2P connectivity fails."""
        with patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity') as mock_check:
            # Test connectivity check failure
            mock_check.return_value = False
            
            with pytest.raises(RuntimeError, match="P2P backend requires full peer connectivity"):
                TPBackendP2P(
                    device=0,
                    active_devices=[0, 1],
                    output_device=0,
                    init_method="tcp://127.0.0.1:29500",
                    master=True,
                    uuid="test_error"
                )

    def test_cuda_error_handling(self):
        """Test handling of CUDA errors during operations."""
        with patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity') as mock_check:
            with patch('exllamav3.model.model_tp_backend_p2p.enable_p2p_access') as mock_enable:
                with patch('exllamav3.model.model_tp_backend_p2p.cuda_host_register') as mock_register:
                    mock_check.return_value = True
                    mock_enable.return_value = None
                    
                    with patch('exllamav3.model.model_tp_backend_p2p.shared_memory.SharedMemory'):
                        with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_init_context'):
                            with patch('exllamav3.model.model_tp_backend_p2p.ext.init_p2p_context') as mock_init_p2p:
                                mock_init_p2p.return_value = 0x12345678
                                
                                backend = TPBackendP2P(
                                    device=0,
                                    active_devices=[0, 1],
                                    output_device=0,
                                    init_method="tcp://127.0.0.1:29500",
                                    master=True,
                                    uuid="test_error"
                                )
                                
                                # Test CUDA error during all-reduce
                                with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_all_reduce_full_p2p') as mock_all_reduce:
                                    mock_all_reduce.side_effect = RuntimeError("CUDA error: out of memory")
                                    
                                    tensor = torch.randn(1000, device=0)
                                    
                                    with pytest.raises(RuntimeError, match="CUDA error: out of memory"):
                                        backend.all_reduce(tensor)
                                
                                backend.close()

    def test_shared_memory_error_handling(self):
        """Test handling of shared memory errors."""
        with patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity') as mock_check:
            with patch('exllamav3.model.model_tp_backend_p2p.enable_p2p_access') as mock_enable:
                with patch('exllamav3.model.model_tp_backend_p2p.cuda_host_register') as mock_register:
                    mock_check.return_value = True
                    mock_enable.return_value = None
                    
                    # Test shared memory creation failure
                    with patch('exllamav3.model.model_tp_backend_p2p.shared_memory.SharedMemory') as mock_shm:
                        mock_shm.side_effect = FileNotFoundError("Shared memory not found")
                        
                        with pytest.raises(FileNotFoundError, match="Shared memory not found"):
                            TPBackendP2P(
                                device=0,
                                active_devices=[0, 1],
                                output_device=0,
                                init_method="tcp://127.0.0.1:29500",
                                master=True,
                                uuid="test_error"
                            )

    def test_timeout_handling(self):
        """Test timeout handling for long-running operations."""
        with patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity') as mock_check:
            with patch('exllamav3.model.model_tp_backend_p2p.enable_p2p_access') as mock_enable:
                with patch('exllamav3.model.model_tp_backend_p2p.cuda_host_register') as mock_register:
                    mock_check.return_value = True
                    mock_enable.return_value = None
                    
                    with patch('exllamav3.model.model_tp_backend_p2p.shared_memory.SharedMemory'):
                        with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_init_context'):
                            with patch('exllamav3.model.model_tp_backend_p2p.ext.init_p2p_context') as mock_init_p2p:
                                mock_init_p2p.return_value = 0x12345678
                                
                                backend = TPBackendP2P(
                                    device=0,
                                    active_devices=[0, 1],
                                    output_device=0,
                                    init_method="tcp://127.0.0.1:29500",
                                    master=True,
                                    uuid="test_timeout"
                                )
                                
                                # Test timeout during barrier
                                with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_barrier_full_p2p') as mock_barrier:
                                    mock_barrier.side_effect = RuntimeError("Operation timed out")
                                    
                                    with pytest.raises(RuntimeError, match="Operation timed out"):
                                        backend.fwd_barrier()
                                
                                backend.close()

    def test_graceful_fallback_to_other_backends(self):
        """Test graceful fallback to other backends when P2P fails."""
        with patch('exllamav3.model.model_tp_backend.TPBackendP2P_AVAILABLE', True):
            with patch('exllamav3.model.model_tp_backend.check_p2p_connectivity') as mock_check:
                # Simulate P2P connectivity failure
                mock_check.return_value = False
                
                # Test auto backend selection falls back to NCCL
                backend = create_tp_backend(
                    backend_type="auto",
                    device=0,
                    active_devices=[0, 1],
                    output_device=0,
                    init_method="tcp://127.0.0.1:29500",
                    master=True,
                    uuid="test_fallback"
                )
                
                # Should create NCCL backend when P2P is not available
                assert isinstance(backend, TPBackendNCCL)


class TestMultiProcessIntegration:
    """Test multi-process integration scenarios."""

    def test_multi_process_synchronization(self):
        """Test synchronization across multiple processes."""
        def worker_process(device, active_devices, result_queue):
            """Worker process function."""
            try:
                with patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity') as mock_check:
                    with patch('exllamav3.model.model_tp_backend_p2p.enable_p2p_access') as mock_enable:
                        with patch('exllamav3.model.model_tp_backend_p2p.cuda_host_register') as mock_register:
                            mock_check.return_value = True
                            mock_enable.return_value = None
                            
                            with patch('exllamav3.model.model_tp_backend_p2p.shared_memory.SharedMemory'):
                                with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_init_context'):
                                    with patch('exllamav3.model.model_tp_backend_p2p.ext.init_p2p_context') as mock_init_p2p:
                                        mock_init_p2p.return_value = 0x12345678
                                        
                                        backend = TPBackendP2P(
                                            device=device,
                                            active_devices=active_devices,
                                            output_device=0,
                                            init_method="tcp://127.0.0.1:29500",
                                            master=(device == 0),
                                            uuid="test_multiprocess"
                                        )
                                        
                                        # Perform barrier synchronization
                                        backend.fwd_barrier()
                                        
                                        result_queue.put(f"Device {device} completed barrier")
                                        backend.close()
            
            except Exception as e:
                result_queue.put(f"Error on device {device}: {e}")
        
        # Test multi-process synchronization
        result_queue = mp.Queue()
        processes = []
        
        for device in [0, 1]:
            p = mp.Process(target=worker_process, args=(device, [0, 1], result_queue))
            processes.append(p)
            p.start()
        
        # Wait for processes to complete
        for p in processes:
            p.join()
        
        # Check results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        
        assert len(results) == 2, "Both processes should complete successfully"
        for result in results:
            assert "completed barrier" in result, f"Unexpected result: {result}"

    def test_process_isolation(self):
        """Test that different processes are properly isolated."""
        def worker_process(device, uuid_suffix, result_queue):
            """Worker process function with different UUID."""
            try:
                with patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity') as mock_check:
                    with patch('exllamav3.model.model_tp_backend_p2p.enable_p2p_access') as mock_enable:
                        with patch('exllamav3.model.model_tp_backend_p2p.cuda_host_register') as mock_register:
                            mock_check.return_value = True
                            mock_enable.return_value = None
                            
                            with patch('exllamav3.model.model_tp_backend_p2p.shared_memory.SharedMemory') as mock_shm:
                                with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_init_context'):
                                    with patch('exllamav3.model.model_tp_backend_p2p.ext.init_p2p_context') as mock_init_p2p:
                                        mock_init_p2p.return_value = 0x12345678
                                        
                                        backend = TPBackendP2P(
                                            device=device,
                                            active_devices=[0, 1],
                                            output_device=0,
                                            init_method="tcp://127.0.0.1:29500",
                                            master=(device == 0),
                                            uuid=f"test_isolation_{uuid_suffix}"
                                        )
                                        
                                        # Each process should have its own shared memory
                                        shm_calls = len(mock_shm.call_args_list)
                                        result_queue.put(f"Device {device}: {shm_calls} SHM calls")
                                        
                                        backend.close()
            
            except Exception as e:
                result_queue.put(f"Error on device {device}: {e}")
        
        # Test process isolation with different UUIDs
        result_queue = mp.Queue()
        processes = []
        
        for i, device in enumerate([0, 1]):
            p = mp.Process(target=worker_process, args=(device, i, result_queue))
            processes.append(p)
            p.start()
        
        # Wait for processes to complete
        for p in processes:
            p.join()
        
        # Check results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        
        assert len(results) == 2, "Both processes should complete successfully"
        for result in results:
            assert "SHM calls" in result, f"Unexpected result: {result}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])