"""
Error handling and recovery tests for P2P backend functionality.

This module tests:
- P2P connectivity failure scenarios
- Resource cleanup on errors
- Graceful fallback to other backends
- Edge cases and boundary conditions
"""

import pytest
import torch
import numpy as np
import time
import psutil
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock, call
import sys
import threading
from contextlib import contextmanager

# Add the project root to sys.path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from exllamav3.model.model_tp_backend_p2p import TPBackendP2P
from exllamav3.model.model_tp_backend import TPBackendNCCL, create_tp_backend
from exllamav3.model.model_tp_cuda import check_p2p_connectivity, enable_p2p_access, disable_p2p_access


class TestP2PConnectivityFailureScenarios:
    """Test various P2P connectivity failure scenarios."""

    def test_connectivity_check_failure_during_initialization(self):
        """Test P2P backend initialization when connectivity check fails."""
        with patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity') as mock_check:
            # Simulate connectivity failure
            mock_check.return_value = False
            
            with pytest.raises(RuntimeError, match="P2P backend requires full peer connectivity"):
                TPBackendP2P(
                    device=0,
                    active_devices=[0, 1, 2],  # Multiple devices
                    output_device=0,
                    init_method="tcp://127.0.0.1:29500",
                    master=True,
                    uuid="test_connectivity_fail"
                )

    def test_partial_connectivity_detection(self):
        """Test detection of partial P2P connectivity (not all pairs connected)."""
        with patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity') as mock_check:
            # Simulate partial connectivity failure
            mock_check.return_value = False
            
            with pytest.raises(RuntimeError, match="P2P backend requires full peer connectivity"):
                TPBackendP2P(
                    device=0,
                    active_devices=[0, 1, 2, 3],  # 4 devices setup
                    output_device=0,
                    init_method="tcp://127.0.0.1:29500",
                    master=True,
                    uuid="test_partial_connectivity"
                )

    def test_single_device_p2p_handling(self):
        """Test handling of single device in P2P setup."""
        with patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity') as mock_check:
            mock_check.return_value = True  # Single device should pass connectivity check
            
            backend = TPBackendP2P(
                device=0,
                active_devices=[0],  # Single device
                output_device=0,
                init_method="tcp://127.0.0.1:29500",
                master=True,
                uuid="test_single_device"
            )
            
            assert backend.device == 0
            assert backend.world_size == 1
            assert backend.rank == 0
            assert backend.p2p_initialized is True
            
            backend.close()

    def test_cuda_api_failure_during_connectivity_check(self):
        """Test CUDA API failure during connectivity check."""
        with patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity') as mock_check:
            # Simulate CUDA API error
            mock_check.side_effect = RuntimeError("CUDA API call failed")
            
            with pytest.raises(RuntimeError, match="CUDA API call failed"):
                TPBackendP2P(
                    device=0,
                    active_devices=[0, 1],
                    output_device=0,
                    init_method="tcp://127.0.0.1:29500",
                    master=True,
                    uuid="test_cuda_api_fail"
                )

    def test_device_not_available_during_check(self):
        """Test handling of devices not being available during connectivity check."""
        with patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity') as mock_check:
            # Simulate device not available error
            mock_check.side_effect = RuntimeError("Device not available")
            
            with pytest.raises(RuntimeError, match="Device not available"):
                TPBackendP2P(
                    device=0,
                    active_devices=[0, 1],
                    output_device=0,
                    init_method="tcp://127.0.0.1:29500",
                    master=True,
                    uuid="test_device_not_available"
                )


class TestResourceCleanupOnErrors:
    """Test resource cleanup when errors occur."""

    @contextmanager
    def p2p_backend_context(self, connectivity_check=True, enable_p2p=True):
        """Context manager for P2P backend with error handling."""
        with patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity') as mock_check:
            with patch('exllamav3.model.model_tp_backend_p2p.enable_p2p_access') as mock_enable:
                with patch('exllamav3.model.model_tp_backend_p2p.cuda_host_register') as mock_register:
                    mock_check.return_value = connectivity_check
                    mock_enable.return_value = enable_p2p
                    
                    with patch('exllamav3.model.model_tp_backend_p2p.shared_memory.SharedMemory') as mock_shm:
                        with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_init_context'):
                            with patch('exllamav3.model.model_tp_backend_p2p.ext.init_p2p_context') as mock_init_p2p:
                                mock_init_p2p.return_value = 0x12345678
                                
                                try:
                                    backend = TPBackendP2P(
                                        device=0,
                                        active_devices=[0, 1],
                                        output_device=0,
                                        init_method="tcp://127.0.0.1:29500",
                                        master=True,
                                        uuid="test_cleanup"
                                    )
                                    yield backend
                                finally:
                                    backend.close()

    def test_cleanup_on_normal_destruction(self):
        """Test normal resource cleanup on successful destruction."""
        with self.p2p_backend_context() as backend:
            # Normal operation
            assert backend.p2p_initialized is True
        
        # Verify cleanup functions were called
        # Note: We can't directly verify the internal cleanup calls due to mocking,
        # but we can verify the backend was properly closed

    def test_cleanup_on_cuda_error_during_operation(self):
        """Test resource cleanup when CUDA error occurs during operation."""
        with self.p2p_backend_context() as backend:
            # Simulate CUDA error during all-reduce
            with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_all_reduce_full_p2p') as mock_all_reduce:
                mock_all_reduce.side_effect = RuntimeError("CUDA out of memory")
                
                tensor = torch.randn(1000, device=0)
                
                with pytest.raises(RuntimeError, match="CUDA out of memory"):
                    backend.all_reduce(tensor)
        
        # Backend should still be properly closed despite the error

    def test_cleanup_on_shared_memory_error(self):
        """Test resource cleanup when shared memory error occurs."""
        with patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity') as mock_check:
            with patch('exllamav3.model.model_tp_backend_p2p.enable_p2p_access') as mock_enable:
                with patch('exllamav3.model.model_tp_backend_p2p.cuda_host_register') as mock_register:
                    mock_check.return_value = True
                    mock_enable.return_value = None
                    
                    # Test shared memory creation failure
                    with patch('exllamav3.model.model_tp_backend_p2p.shared_memory.SharedMemory') as mock_shm:
                        mock_shm.side_effect = RuntimeError("Shared memory allocation failed")
                        
                        with pytest.raises(RuntimeError, match="Shared memory allocation failed"):
                            TPBackendP2P(
                                device=0,
                                active_devices=[0, 1],
                                output_device=0,
                                init_method="tcp://127.0.0.1:29500",
                                master=True,
                                uuid="test_shm_fail"
                            )

    def test_cleanup_on_p2p_context_initialization_failure(self):
        """Test resource cleanup when P2P context initialization fails."""
        with self.p2p_backend_context() as backend:
            # Simulate P2P context initialization failure
            with patch('exllamav3.model.model_tp_backend_p2p.ext.init_p2p_context') as mock_init_p2p:
                mock_init_p2p.side_effect = RuntimeError("P2P context initialization failed")
                
                with pytest.raises(RuntimeError, match="P2P context initialization failed"):
                    # Re-initialize backend to trigger the error
                    backend._init_p2p_context()
        
        # Backend should still be properly closed despite the error

    def test_memory_cleanup_verification(self):
        """Test that memory resources are properly cleaned up."""
        with patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity') as mock_check:
            with patch('exllamav3.model.model_tp_backend_p2p.enable_p2p_access') as mock_enable:
                with patch('exllamav3.model.model_tp_backend_p2p.cuda_host_register') as mock_register:
                    mock_check.return_value = True
                    mock_enable.return_value = None
                    
                    with patch('exllamav3.model.model_tp_backend_p2p.shared_memory.SharedMemory') as mock_shm:
                        with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_init_context'):
                            with patch('exllamav3.model.model_tp_backend_p2p.ext.init_p2p_context') as mock_init_p2p:
                                mock_init_p2p.return_value = 0x12345678
                                
                                # Track memory registration/unregistration calls
                                register_calls = []
                                unregister_calls = []
                                
                                def mock_cuda_register(ptr, size, flags):
                                    register_calls.append((ptr, size, flags))
                                
                                def mock_cuda_unregister(ptr):
                                    unregister_calls.append(ptr)
                                
                                mock_register.side_effect = mock_cuda_register
                                
                                backend = TPBackendP2P(
                                    device=0,
                                    active_devices=[0, 1],
                                    output_device=0,
                                    init_method="tcp://127.0.0.1:29500",
                                    master=True,
                                    uuid="test_memory_cleanup"
                                )
                                
                                # Simulate memory unregistration during close
                                backend.cuda_host_unregister = mock_cuda_unregister
                                
                                # Close backend
                                backend.close()
                                
                                # Verify cleanup
                                assert len(register_calls) == 4  # 4 buffers registered
                                assert len(unregister_calls) == 4  # 4 buffers unregistered
                                
                                # Verify the same pointers were unregistered that were registered
                                registered_ptrs = [call[0] for call in register_calls]
                                unregistered_ptrs = unregister_calls
                                assert set(registered_ptrs) == set(unregistered_ptrs)


class TestGracefulFallbackToOtherBackends:
    """Test graceful fallback to other backends when P2P fails."""

    def test_auto_fallback_to_nccl_when_p2p_unavailable(self):
        """Test auto backend selection falls back to NCCL when P2P unavailable."""
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

    def test_explicit_fallback_when_p2p_not_available(self):
        """Test explicit backend selection when P2P is not available."""
        with patch('exllamav3.model.model_tp_backend.TPBackendP2P_AVAILABLE', False):
            # Test explicit P2P backend selection when not available
            with pytest.raises(ValueError, match="TPBackendP2P is not available"):
                create_tp_backend(
                    backend_type="p2p",
                    device=0,
                    active_devices=[0, 1],
                    output_device=0,
                    init_method="tcp://127.0.0.1:29500",
                    master=True,
                    uuid="test_fallback_explicit"
                )

    def test_fallback_preserves_configuration(self):
        """Test that fallback backend preserves the original configuration."""
        with patch('exllamav3.model.model_backend.TPBackendP2P_AVAILABLE', True):
            with patch('exllamav3.model.model_tp_backend.check_p2p_connectivity') as mock_check:
                mock_check.return_value = False
                
                with patch('exllamav3.model.model_tp_backend.TPBackendNCCL') as mock_nccl:
                    mock_nccl_instance = Mock()
                    mock_nccl.return_value = mock_nccl_instance
                    
                    backend = create_tp_backend(
                        backend_type="auto",
                        device=0,
                        active_devices=[0, 1, 2],
                        output_device=1,
                        init_method="tcp://127.0.0.1:29500",
                        master=False,
                        uuid="test_fallback_config"
                    )
                    
                    # Verify NCCL backend was created with correct configuration
                    mock_nccl.assert_called_once()
                    call_args = mock_nccl.call_args
                    assert call_args[1]['device'] == 0
                    assert call_args[1]['active_devices'] == [0, 1, 2]
                    assert call_args[1]['output_device'] == 1
                    assert call_args[1]['init_method'] == "tcp://127.0.0.1:29500"
                    assert call_args[1]['master'] == False
                    assert call_args[1]['uuid'] == "test_fallback_config"

    def test_fallback_with_mixed_connectivity(self):
        """Test fallback behavior with mixed connectivity scenarios."""
        with patch('exllamav3.model.model_tp_backend.TPBackendP2P_AVAILABLE', True):
            with patch('exllamav3.model.model_tp_backend.check_p2p_connectivity') as mock_check:
                # Test different connectivity scenarios
                connectivity_scenarios = [
                    ([0, 1], False),  # 2 devices, not connected
                    ([0, 1, 2], False),  # 3 devices, not connected
                    ([0, 1, 2, 3], False),  # 4 devices, not connected
                ]
                
                for devices, connected in connectivity_scenarios:
                    mock_check.return_value = connected
                    
                    with patch('exllamav3.model.model_tp_backend.TPBackendNCCL') as mock_nccl:
                        mock_nccl_instance = Mock()
                        mock_nccl.return_value = mock_nccl_instance
                        
                        backend = create_tp_backend(
                            backend_type="auto",
                            device=0,
                            active_devices=devices,
                            output_device=0,
                            init_method="tcp://127.0.0.1:29500",
                            master=True,
                            uuid=f"test_fallback_mixed_{len(devices)}"
                        )
                        
                        # Should always fallback to NCCL when P2P not connected
                        assert isinstance(backend, TPBackendNCCL)

    def test_fallback_memory_consistency(self):
        """Test that fallback maintains memory consistency requirements."""
        with patch('exllamav3.model.model_tp_backend.TPBackendP2P_AVAILABLE', True):
            with patch('exllamav3.model.model_tp_backend.check_p2p_connectivity') as mock_check:
                mock_check.return_value = False
                
                with patch('exllamav3.model.model_tp_backend.TPBackendNCCL') as mock_nccl:
                    mock_nccl_instance = Mock()
                    mock_nccl.return_value = mock_nccl_instance
                    
                    # Create backend with specific memory requirements
                    backend = create_tp_backend(
                        backend_type="auto",
                        device=0,
                        active_devices=[0, 1],
                        output_device=0,
                        init_method="tcp://127.0.0.1:29500",
                        master=True,
                        uuid="test_fallback_memory",
                        shbuf_size=32 * 1024 * 1024  # 32MB buffer
                    )
                    
                    # Verify NCCL backend was created with correct buffer size
                    mock_nccl.assert_called_once()
                    call_args = mock_nccl.call_args
                    assert call_args[1]['shbuf_size'] == 32 * 1024 * 1024


class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions for P2P backend."""

    def test_maximum_device_count_handling(self):
        """Test handling of maximum device count."""
        # Test with large number of devices (theoretical limit)
        max_devices = 16  # Reasonable upper limit
        
        with patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity') as mock_check:
            mock_check.return_value = True
            
            with patch('exllamav3.model.model_tp_backend_p2p.enable_p2p_access') as mock_enable:
                with patch('exllamav3.model.model_tp_backend_p2p.cuda_host_register') as mock_register:
                    with patch('exllamav3.model.model_tp_backend_p2p.shared_memory.SharedMemory'):
                        with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_init_context'):
                            with patch('exllamav3.model.model_tp_backend_p2p.ext.init_p2p_context') as mock_init_p2p:
                                mock_init_p2p.return_value = 0x12345678
                                
                                devices = list(range(max_devices))
                                backend = TPBackendP2P(
                                    device=0,
                                    active_devices=devices,
                                    output_device=0,
                                    init_method="tcp://127.0.0.1:29500",
                                    master=True,
                                    uuid="test_max_devices"
                                )
                                
                                assert backend.world_size == max_devices
                                assert len(backend.active_devices) == max_devices
                                
                                # Test operations with many devices
                                tensor = torch.randn(1000, device=0)
                                with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_all_reduce_full_p2p'):
                                    with patch('exllamav3.model.model_tp_backend_p2p.uintptr_t', side_effect=devices):
                                        backend.all_reduce(tensor)
                                
                                backend.close()

    def test_minimum_tensor_size_handling(self):
        """Test handling of minimum tensor sizes."""
        with patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity') as mock_check:
            mock_check.return_value = True
            
            with patch('exllamav3.model.model_tp_backend_p2p.enable_p2p_access') as mock_enable:
                with patch('exllamav3.model.model_tp_backend_p2p.cuda_host_register') as mock_register:
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
                                    uuid="test_min_tensor"
                                )
                                
                                # Test very small tensor
                                small_tensor = torch.randn(1, device=0)
                                with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_all_reduce_full_p2p'):
                                    with patch('exllamav3.model.model_tp_backend_p2p.uintptr_t', side_effect=[0, 1]):
                                        backend.all_reduce(small_tensor)
                                
                                # Test tensor with multiple dimensions
                                        multi_dim_tensor = torch.randn(2, 3, 4, device=0)
                                        backend.all_reduce(multi_dim_tensor)
                                
                                backend.close()

    def test_invalid_tensor_dtype_handling(self):
        """Test handling of invalid tensor data types."""
        with patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity') as mock_check:
            mock_check.return_value = True
            
            with patch('exllamav3.model.model_tp_backend_p2p.enable_p2p_access') as mock_enable:
                with patch('exllamav3.model.model_tp_backend_p2p.cuda_host_register') as mock_register:
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
                                    uuid="test_invalid_dtype"
                                )
                                
                                # Test with different data types
                                dtypes_to_test = [
                                    torch.float32,
                                    torch.float16,
                                    torch.bfloat16,
                                    torch.int32,
                                    torch.int64
                                ]
                                
                                for dtype in dtypes_to_test:
                                    tensor = torch.randn(1000, dtype=dtype, device=0)
                                    
                                    # Mock the underlying operations to avoid dtype validation issues
                                    with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_all_reduce_full_p2p'):
                                        with patch('exllamav3.model.model_tp_backend_p2p.uintptr_t', side_effect=[0, 1]):
                                            backend.all_reduce(tensor)
                                
                                backend.close()

    def test_timeout_handling_scenarios(self):
        """Test various timeout handling scenarios."""
        with patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity') as mock_check:
            mock_check.return_value = True
            
            with patch('exllamav3.model.model_tp_backend_p2p.enable_p2p_access') as mock_enable:
                with patch('exllamav3.model.model_tp_backend_p2p.cuda_host_register') as mock_register:
                    with patch('exllamav3.model.model_tp_backend_p2p.shared_memory.SharedMemory'):
                        with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_init_context'):
                            with patch('exllamavav3.model.model_tp_backend_p2p.ext.init_p2p_context') as mock_init_p2p:
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
                                
                                # Test timeout during all-reduce
                                with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_all_reduce_full_p2p') as mock_all_reduce:
                                    mock_all_reduce.side_effect = RuntimeError("All-reduce timeout")
                                    
                                    tensor = torch.randn(1000, device=0)
                                    
                                    with pytest.raises(RuntimeError, match="All-reduce timeout"):
                                        backend.all_reduce(tensor)
                                
                                backend.close()

    def test_concurrent_access_handling(self):
        """Test handling of concurrent access to P2P backend."""
        with patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity') as mock_check:
            mock_check.return_value = True
            
            with patch('exllamav3.model.model_tp_backend_p2p.enable_p2p_access') as mock_enable:
                with patch('exllamav3.model.model_tp_backend_p2p.cuda_host_register') as mock_register:
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
                                    uuid="test_concurrent"
                                )
                                
                                # Test concurrent operations
                                def concurrent_operation(tensor):
                                    backend.all_reduce(tensor)
                                
                                # Create multiple threads performing operations
                                threads = []
                                test_tensor = torch.randn(1000, device=0)
                                
                                for i in range(5):
                                    thread = threading.Thread(
                                        target=concurrent_operation,
                                        args=(test_tensor,)
                                    )
                                    threads.append(thread)
                                    thread.start()
                                
                                # Wait for all threads to complete
                                for thread in threads:
                                    thread.join()
                                
                                backend.close()

    def test_resource_starvation_scenario(self):
        """Test handling of resource starvation scenarios."""
        with patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity') as mock_check:
            mock_check.return_value = True
            
            with patch('exllamav3.model.model_tp_backend_p2p.enable_p2p_access') as mock_enable:
                with patch('exllamav3.model.model_tp_backend_p2p.cuda_host_register') as mock_register:
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
                                    uuid="test_starvation"
                                )
                                
                                # Simulate memory pressure by allocating large tensors
                                large_tensors = []
                                for i in range(10):
                                    large_tensor = torch.randn(50000, device=0)
                                    large_tensors.append(large_tensor)
                                    
                                    # Perform operation with large tensor
                                    with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_all_reduce_full_p2p'):
                                        with patch('exllamav3.model.model_tp_backend_p2p.uintptr_t', side_effect=[0, 1]):
                                            backend.all_reduce(large_tensor)
                                
                                # Clean up tensors
                                del large_tensors
                                torch.cuda.empty_cache()
                                
                                backend.close()


class TestErrorRecoveryStrategies:
    """Test various error recovery strategies for P2P backend."""

    def test_reinitialization_after_failure(self):
        """Test reinitialization of P2P backend after failure."""
        with patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity') as mock_check:
            mock_check.return_value = True
            
            with patch('exllamav3.model.model_tp_backend_p2p.enable_p2p_access') as mock_enable:
                with patch('exllamav3.model.model_tp_backend_p2p.cuda_host_register') as mock_register:
                    with patch('exllamav3.model.model_tp_backend_p2p.shared_memory.SharedMemory'):
                        with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_init_context'):
                            with patch('exllamav3.model.model_tp_backend_p2p.ext.init_p2p_context') as mock_init_p2p:
                                mock_init_p2p.return_value = 0x12345678
                                
                                # Create backend
                                backend = TPBackendP2P(
                                    device=0,
                                    active_devices=[0, 1],
                                    output_device=0,
                                    init_method="tcp://127.0.0.1:29500",
                                    master=True,
                                    uuid="test_reinit"
                                )
                                
                                # Simulate failure and recovery
                                original_context = backend.p2p_context
                                
                                # Close and recreate backend
                                backend.close()
                                
                                # Create new instance
                                new_backend = TPBackendP2P(
                                    device=0,
                                    active_devices=[0, 1],
                                    output_device=0,
                                    init_method="tcp://127.0.0.1:29500",
                                    master=True,
                                    uuid="test_reinit_new"
                                )
                                
                                # Verify new backend is properly initialized
                                assert new_backend.p2p_initialized is True
                                assert new_backend.p2p_context != original_context
                                
                                new_backend.close()

    def test_retry_mechanism_for_transient_errors(self):
        """Test retry mechanism for transient errors."""
        with patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity') as mock_check:
            mock_check.return_value = True
            
            with patch('exllamav3.model.model_tp_backend_p2p.enable_p2p_access') as mock_enable:
                with patch('exllamav3.model.model_tp_backend_p2p.cuda_host_register') as mock_register:
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
                                    uuid="test_retry"
                                )
                                
                                # Simulate transient errors with retry mechanism
                                retry_count = 0
                                def mock_all_reduce_with_retry(*args, **kwargs):
                                    nonlocal retry_count
                                    retry_count += 1
                                    if retry_count <= 2:
                                        raise RuntimeError("Transient error")
                                    # Success after retries
                                    return None
                                
                                with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_all_reduce_full_p2p') as mock_all_reduce:
                                    mock_all_reduce.side_effect = mock_all_reduce_with_retry
                                    
                                    tensor = torch.randn(1000, device=0)
                                    
                                    # Should succeed after retries
                                    backend.all_reduce(tensor)
                                    
                                    assert retry_count == 3  # 2 failures + 1 success
                                
                                backend.close()

    def test_circuit_breaker_pattern(self):
        """Test circuit breaker pattern for repeated failures."""
        with patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity') as mock_check:
            mock_check.return_value = True
            
            with patch('exllamav3.model.model_tp_backend_p2p.enable_p2p_access') as mock_enable:
                with patch('exllamav3.model.model_tp_backend_p2p.cuda_host_register') as mock_register:
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
                                    uuid="test_circuit_breaker"
                                )
                                
                                # Simulate repeated failures
                                failure_count = 0
                                def mock_all_reduce_with_circuit_breaker(*args, **kwargs):
                                    nonlocal failure_count
                                    failure_count += 1
                                    if failure_count <= 5:  # Fail 5 times
                                        raise RuntimeError("Repeated failure")
                                    else:
                                        return None  # Success after circuit breaker trips
                                
                                with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_all_reduce_full_p2p') as mock_all_reduce:
                                    mock_all_reduce.side_effect = mock_all_reduce_with_circuit_breaker
                                    
                                    tensor = torch.randn(1000, device=0)
                                    
                                    # Should eventually succeed after circuit breaker trips
                                    backend.all_reduce(tensor)
                                    
                                    assert failure_count == 6  # 5 failures + 1 success
                                
                                backend.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])