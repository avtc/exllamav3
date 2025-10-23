"""
Unit tests for P2P backend initialization and detection functionality.

This module tests:
- P2P connectivity detection functions
- TPBackendP2P initialization with valid/invalid configurations
- P2P memory management utilities
- Backend selection logic and automatic detection
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock, call
import sys
import os
import tempfile
import multiprocessing as mp
from multiprocessing import shared_memory

# Add the project root to sys.path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from exllamav3.model.model_tp_backend_p2p import TPBackendP2P
from exllamav3.model.model_tp_backend import create_tp_backend, get_available_backends
from exllamav3.model.model_tp_cuda import (
    check_p2p_connectivity,
    enable_p2p_access,
    disable_p2p_access,
    cuda_host_register,
    cuda_host_unregister,
    CUDA_HOST_REGISTER_PORTABLE
)


class TestP2PConnectivityDetection:
    """Test P2P connectivity detection functions."""

    def test_check_p2p_connectivity_single_device(self, mocker):
        """Test connectivity check with single device."""
        mock_lib = Mock()
        mock_cuda_get_device_count = Mock(return_value=1)
        mock_cuda_can_access_peer = Mock(return_value=1)
        
        mock_lib.cudaGetDeviceCount = mock_cuda_get_device_count
        mock_lib.cudaDeviceCanAccessPeer = mock_cuda_can_access_peer
        mock_lib.cudaSetDevice = Mock(return_value=0)
        
        mocker.patch('exllamav3.model.model_tp_cuda._cudart', return_value=mock_lib)
        
        result = check_p2p_connectivity([0])
        assert result is True
        mock_cuda_get_device_count.assert_called_once()
        # Should not call can_access_peer for single device

    def test_check_p2p_connectivity_multiple_devices_connected(self, mocker):
        """Test connectivity check with multiple fully connected devices."""
        mock_lib = Mock()
        mock_cuda_get_device_count = Mock(return_value=2)
        mock_cuda_can_access_peer = Mock(return_value=1)
        mock_cuda_set_device = Mock(return_value=0)
        
        mock_lib.cudaGetDeviceCount = mock_cuda_get_device_count
        mock_lib.cudaDeviceCanAccessPeer = mock_cuda_can_access_peer
        mock_lib.cudaSetDevice = mock_cuda_set_device
        
        mocker.patch('exllamav3.model.model_tp_cuda._cudart', return_value=mock_lib)
        
        result = check_p2p_connectivity([0, 1])
        assert result is True
        # Should check P2P access in one direction for each unique pair
        assert mock_cuda_set_device.call_count == 2  # 2 unique device pairs, 1 device set each
        assert mock_cuda_can_access_peer.call_count == 2

    def test_check_p2p_connectivity_multiple_devices_not_connected(self, mocker):
        """Test connectivity check with partially connected devices."""
        mock_lib = Mock()
        mock_cuda_get_device_count = Mock(return_value=2)
        mock_cuda_can_access_peer = Mock()  # One direction returns 0 (not accessible)
        mock_cuda_set_device = Mock(return_value=0)
        
        # First call returns 1 (accessible), second returns 0 (not accessible)
        mock_cuda_can_access_peer.side_effect = [1, 0]
        
        mock_lib.cudaGetDeviceCount = mock_cuda_get_device_count
        mock_lib.cudaDeviceCanAccessPeer = mock_cuda_can_access_peer
        mock_lib.cudaSetDevice = mock_cuda_set_device
        
        mocker.patch('exllamav3.model.model_tp_cuda._cudart', return_value=mock_lib)
        
        result = check_p2p_connectivity([0, 1])
        assert result is False

    def test_check_p2p_connectivity_cuda_error(self, mocker):
        """Test connectivity check with CUDA API error."""
        mock_lib = Mock()
        mock_cuda_get_device_count = Mock(side_effect=RuntimeError("CUDA error"))
        
        mock_lib.cudaGetDeviceCount = mock_cuda_get_device_count
        
        mocker.patch('exllamav3.model.model_tp_cuda._cudart', return_value=mock_lib)
        
        with pytest.raises(RuntimeError, match="Failed to get device count"):
            check_p2p_connectivity([0, 1])

    def test_enable_p2p_access(self, mocker):
        """Test enabling P2P access between devices."""
        mock_lib = Mock()
        mock_cuda_set_device = Mock(return_value=0)
        mock_cuda_enable_peer_access = Mock(return_value=0)  # Success
        
        mock_lib.cudaSetDevice = mock_cuda_set_device
        mock_lib.cudaDeviceEnablePeerAccess = mock_cuda_enable_peer_access
        
        mocker.patch('exllamav3.model.model_tp_cuda._cudart', return_value=mock_lib)
        
        enable_p2p_access([0, 1])
        
        # Should enable P2P access in both directions for each device pair
        assert mock_cuda_set_device.call_count == 2  # 2 unique device pairs, 1 device set each
        assert mock_cuda_enable_peer_access.call_count == 2

    def test_disable_p2p_access(self, mocker):
        """Test disabling P2P access between devices."""
        mock_lib = Mock()
        mock_cuda_set_device = Mock(return_value=0)
        mock_cuda_disable_peer_access = Mock(return_value=0)  # Success
        
        mock_lib.cudaSetDevice = mock_cuda_set_device
        mock_lib.cudaDeviceDisablePeerAccess = mock_cuda_disable_peer_access
        
        mocker.patch('exllamav3.model.model_tp_cuda._cudart', return_value=mock_lib)
        
        disable_p2p_access([0, 1])
        
        # Should disable P2P access in both directions for each device pair
        assert mock_cuda_set_device.call_count == 2  # 2 unique device pairs, 1 device set each
        assert mock_cuda_disable_peer_access.call_count == 2


class TestTPBackendP2PInitialization:
    """Test TPBackendP2P initialization scenarios."""

    @patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity')
    @patch('exllamav3.model.model_tp_backend_p2p.enable_p2p_access')
    @patch('exllamav3.model.model_tp_backend_p2p.cuda_host_register')
    def test_initialization_success(self, mock_cuda_register, mock_enable_p2p, mock_check_p2p):
        """Test successful P2P backend initialization."""
        mock_check_p2p.return_value = True
        mock_enable_p2p.return_value = None
        
        # Mock shared memory creation
        with patch('exllamav3.model.model_tp_backend_p2p.shared_memory.SharedMemory') as mock_shm:
            def mock_shared_memory_side_effect(*args, **kwargs):
                instance = Mock()
                # Determine buffer size based on name parameter
                name = kwargs.get('name', args[1] if len(args) > 1 else '')
                size = kwargs.get('size', args[0] if len(args) > 0 else 0)
                
                # Use the largest size among all expected buffers
                buffer_size = max(size, 17 * 128 * 1024)  # At least 2.25MB
                instance.buf = bytearray(buffer_size)
                return instance
            
            mock_shm.side_effect = mock_shared_memory_side_effect
            
            # Mock torch extension functions
            with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_init_context') as mock_init_context:
                with patch('exllamav3.model.model_tp_backend_p2p.ext.init_p2p_context') as mock_init_p2p:
                    mock_init_p2p.return_value = 0x12345678
                    
                    backend = TPBackendP2P(
                        device=0,
                        active_devices=[0, 1],
                        output_device=0,
                        init_method="tcp://127.0.0.1:29500",
                        master=True,
                        uuid="test_uuid"
                    )
                    
                    assert backend.device == 0
                    assert backend.active_devices == [0, 1]
                    assert backend.world_size == 2
                    assert backend.rank == 0
                    assert backend.p2p_initialized is True
                    assert backend.p2p_context == 0x12345678
                    
                    # Verify P2P checks and setup
                    mock_check_p2p.assert_called_once_with([0, 1])
                    mock_enable_p2p.assert_called_once_with([0, 1])

    @patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity')
    def test_initialization_connectivity_check_fails(self, mock_check_p2p):
        """Test initialization when P2P connectivity check fails."""
        mock_check_p2p.return_value = False
        
        with pytest.raises(RuntimeError, match="P2P backend requires full peer connectivity"):
            TPBackendP2P(
                device=0,
                active_devices=[0, 1],
                output_device=0,
                init_method="tcp://127.0.0.1:29500",
                master=True,
                uuid="test_uuid"
            )

    @patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity')
    @patch('exllamav3.model.model_tp_backend_p2p.enable_p2p_access')
    def test_initialization_enable_p2p_fails(self, mock_enable_p2p, mock_check_p2p):
        """Test initialization when enabling P2P access fails."""
        mock_check_p2p.return_value = True
        mock_enable_p2p.side_effect = RuntimeError("Failed to enable P2P access")
        
        with pytest.raises(RuntimeError, match="Failed to enable P2P access"):
            TPBackendP2P(
                device=0,
                active_devices=[0, 1],
                output_device=0,
                init_method="tcp://127.0.0.1:29500",
                master=True,
                uuid="test_uuid"
            )

    def test_initialization_cpu_process(self):
        """Test initialization for CPU process (should skip)."""
        backend = TPBackendP2P(
            device=-1,
            active_devices=[-1],
            output_device=-1,
            init_method="tcp://127.0.0.1:29500",
            master=True,
            uuid="test_uuid"
        )
        
        assert backend.device == -1
        # Should not raise any errors but skip GPU-specific initialization

    @patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity')
    @patch('exllamav3.model.model_tp_backend_p2p.enable_p2p_access')
    @patch('exllamav3.model.model_tp_backend_p2p.cuda_host_register')
    def test_shared_memory_buffer_creation(self, mock_cuda_register, mock_enable_p2p, mock_check_p2p):
        """Test shared memory buffer creation and management."""
        mock_check_p2p.return_value = True
        mock_enable_p2p.return_value = None
        
        # Mock shared memory creation
        with patch('exllamav3.model.model_tp_backend_p2p.shared_memory.SharedMemory') as mock_shm:
            def mock_shared_memory_side_effect(*args, **kwargs):
                instance = Mock()
                # Determine buffer size based on name parameter
                name = kwargs.get('name', args[1] if len(args) > 1 else '')
                size = kwargs.get('size', args[0] if len(args) > 0 else 0)
                
                # Use the largest size among all expected buffers
                buffer_size = max(size, 17 * 128 * 1024)  # At least 2.25MB
                instance.buf = bytearray(buffer_size)
                return instance
            
            mock_shm.side_effect = mock_shared_memory_side_effect
            
            # Mock torch extension functions
            with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_init_context') as mock_init_context:
                with patch('exllamav3.model.model_tp_backend_p2p.ext.init_p2p_context') as mock_init_p2p:
                    mock_init_p2p.return_value = 0x12345678
                    
                    backend = TPBackendP2P(
                        device=0,
                        active_devices=[0, 1],
                        output_device=0,
                        init_method="tcp://127.0.0.1:29500",
                        master=True,
                        uuid="test_uuid"
                    )
                    
                    # Verify shared memory creation for master
                    expected_calls = [
                        call(create=True, size=128*1024, name="test_uuid_p2p_g"),
                        call(create=True, size=16*1024**2, name="test_uuid_p2p_b"),
                        call(create=True, size=17*128*1024, name="test_uuid_p2p_r"),
                        call(create=True, size=16*1024, name="test_uuid_p2p_s")
                    ]
                    mock_shm.assert_has_calls(expected_calls)

    @patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity')
    @patch('exllamav3.model.model_tp_backend_p2p.enable_p2p_access')
    @patch('exllamav3.model.model_tp_backend_p2p.cuda_host_register')
    def test_p2p_buffer_allocation(self, mock_cuda_register, mock_enable_p2p, mock_check_p2p):
        """Test P2P buffer allocation with appropriate size."""
        mock_check_p2p.return_value = True
        mock_enable_p2p.return_value = None
        
        # Mock shared memory creation
        with patch('exllamav3.model.model_tp_backend_p2p.shared_memory.SharedMemory') as mock_shm:
            def mock_shared_memory_side_effect(*args, **kwargs):
                instance = Mock()
                # Determine buffer size based on name parameter
                name = kwargs.get('name', args[1] if len(args) > 1 else '')
                size = kwargs.get('size', args[0] if len(args) > 0 else 0)
                
                # Use the largest size among all expected buffers (32MB from test parameter)
                buffer_size = max(size, 32 * 1024**2)  # At least 32MB
                instance.buf = bytearray(buffer_size)
                return instance
            
            mock_shm.side_effect = mock_shared_memory_side_effect
            
            # Mock torch extension functions
            with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_init_context') as mock_init_context:
                with patch('exllamav3.model.model_tp_backend_p2p.ext.init_p2p_context') as mock_init_p2p:
                    mock_init_p2p.return_value = 0x12345678
                    
                    backend = TPBackendP2P(
                        device=0,
                        active_devices=[0, 1],
                        output_device=0,
                        init_method="tcp://127.0.0.1:29500",
                        master=True,
                        uuid="test_uuid",
                        shbuf_size=32 * 1024**2  # 32MB buffer
                    )
                    
                    # P2P buffer should be at least 1MB or shbuf_size/4
                    expected_size = max(1024 * 1024, 32 * 1024**2 // 4)
                    assert backend.p2p_buffer_size == expected_size
                    assert backend.p2p_buffer is not None
                    assert backend.p2p_buffer.device == torch.device('cuda:0')


class TestTPBackendP2PMemoryManagement:
    """Test P2P memory management utilities."""

    @patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity')
    @patch('exllamav3.model.model_tp_backend_p2p.enable_p2p_access')
    @patch('exllamav3.model.model_tp_backend_p2p.cuda_host_register')
    def test_host_memory_registration(self, mock_cuda_register, mock_enable_p2p, mock_check_p2p):
        """Test host memory registration for shared buffers."""
        mock_check_p2p.return_value = True
        mock_enable_p2p.return_value = None
        
        # Mock shared memory creation
        with patch('exllamav3.model.model_tp_backend_p2p.shared_memory.SharedMemory') as mock_shm:
            def mock_shared_memory_side_effect(*args, **kwargs):
                instance = Mock()
                # Determine buffer size based on name parameter
                name = kwargs.get('name', args[1] if len(args) > 1 else '')
                size = kwargs.get('size', args[0] if len(args) > 0 else 0)
                
                # Use the largest size among all expected buffers
                buffer_size = max(size, 17 * 128 * 1024)  # At least 2.25MB
                instance.buf = bytearray(buffer_size)
                return instance
            
            mock_shm.side_effect = mock_shared_memory_side_effect
            
            # Mock torch extension functions
            with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_init_context') as mock_init_context:
                with patch('exllamav3.model.model_tp_backend_p2p.ext.init_p2p_context') as mock_init_p2p:
                    mock_init_p2p.return_value = 0x12345678
                    
                    backend = TPBackendP2P(
                        device=0,
                        active_devices=[0, 1],
                        output_device=0,
                        init_method="tcp://127.0.0.1:29500",
                        master=True,
                        uuid="test_uuid"
                    )
                    
                    # Verify all four buffers are registered
                    expected_register_calls = [
                        call(backend.ptr_g, backend.tensor_g.numel(), flags=CUDA_HOST_REGISTER_PORTABLE),
                        call(backend.ptr_b, backend.tensor_b.numel(), flags=CUDA_HOST_REGISTER_PORTABLE),
                        call(backend.ptr_r, backend.tensor_r.numel(), flags=CUDA_HOST_REGISTER_PORTABLE),
                        call(backend.ptr_s, backend.tensor_s.numel(), flags=CUDA_HOST_REGISTER_PORTABLE)
                    ]
                    mock_cuda_register.assert_has_calls(expected_register_calls)

    @patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity')
    @patch('exllamav3.model.model_tp_backend_p2p.enable_p2p_access')
    @patch('exllamav3.model.model_tp_backend_p2p.cuda_host_register')
    @patch('exllamav3.model.model_tp_backend_p2p.cuda_host_unregister')
    def test_host_memory_unregistration(self, mock_cuda_unregister, mock_cuda_register, mock_enable_p2p, mock_check_p2p):
        """Test host memory unregistration during cleanup."""
        mock_check_p2p.return_value = True
        mock_enable_p2p.return_value = None
        
        # Mock shared memory creation
        with patch('exllamav3.model.model_tp_backend_p2p.shared_memory.SharedMemory') as mock_shm:
            def mock_shared_memory_side_effect(*args, **kwargs):
                instance = Mock()
                # Determine buffer size based on name parameter
                name = kwargs.get('name', args[1] if len(args) > 1 else '')
                size = kwargs.get('size', args[0] if len(args) > 0 else 0)
                
                # Use the largest size among all expected buffers
                buffer_size = max(size, 17 * 128 * 1024)  # At least 2.25MB
                instance.buf = bytearray(buffer_size)
                return instance
            
            mock_shm.side_effect = mock_shared_memory_side_effect
            
            # Mock torch extension functions
            with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_init_context') as mock_init_context:
                with patch('exllamav3.model.model_tp_backend_p2p.ext.init_p2p_context') as mock_init_p2p:
                    with patch('exllamav3.model.model_tp_backend_p2p.ext.destroy_p2p_context') as mock_destroy_p2p:
                        mock_init_p2p.return_value = 0x12345678
                        
                        backend = TPBackendP2P(
                            device=0,
                            active_devices=[0, 1],
                            output_device=0,
                            init_method="tcp://127.0.0.1:29500",
                            master=True,
                            uuid="test_uuid"
                        )
                        
                        # Test cleanup
                        backend.close()
                        
                        # Verify P2P context destruction
                        mock_destroy_p2p.assert_called_once_with(0x12345678)
                        
                        # Verify all buffers are unregistered
                        expected_unregister_calls = [
                            call(backend.ptr_g),
                            call(backend.ptr_b),
                            call(backend.ptr_r),
                            call(backend.ptr_s)
                        ]
                        mock_cuda_unregister.assert_has_calls(expected_unregister_calls)

    @patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity')
    @patch('exllamav3.model.model_tp_backend_p2p.enable_p2p_access')
    @patch('exllamav3.model.model_tp_backend_p2p.cuda_host_register')
    @patch('exllamav3.model.model_tp_backend_p2p.cuda_host_unregister')
    def test_shared_memory_cleanup(self, mock_cuda_unregister, mock_cuda_register, mock_enable_p2p, mock_check_p2p):
        """Test shared memory cleanup during backend destruction."""
        mock_check_p2p.return_value = True
        mock_enable_p2p.return_value = None
        
        # Mock shared memory creation
        created_instances = []
        with patch('exllamav3.model.model_tp_backend_p2p.shared_memory.SharedMemory') as mock_shm:
            def mock_shared_memory_side_effect(*args, **kwargs):
                instance = Mock()
                # Determine buffer size based on name parameter
                name = kwargs.get('name', args[1] if len(args) > 1 else '')
                size = kwargs.get('size', args[0] if len(args) > 0 else 0)
                
                # Use the largest size among all expected buffers
                buffer_size = max(size, 17 * 128 * 1024)  # At least 2.25MB
                instance.buf = bytearray(buffer_size)
                created_instances.append(instance)
                return instance
            
            mock_shm.side_effect = mock_shared_memory_side_effect
            
            # Mock torch extension functions
            with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_init_context') as mock_init_context:
                with patch('exllamav3.model.model_tp_backend_p2p.ext.init_p2p_context') as mock_init_p2p:
                    mock_init_p2p.return_value = 0x12345678
                    
                    backend = TPBackendP2P(
                        device=0,
                        active_devices=[0, 1],
                        output_device=0,
                        init_method="tcp://127.0.0.1:29500",
                        master=True,
                        uuid="test_uuid"
                    )
                    
                    # Test cleanup
                    backend.close()
                    
                    # Verify all shared memory buffers are closed
                    assert len(created_instances) == 4  # Should create 4 shared memory objects
                    
                    # Check that close was called on each shared memory object
                    for shm_instance in created_instances:
                        shm_instance.close.assert_called_once()
                    
                    # Verify unlink is called for master process
                    if backend.master:
                        for shm_instance in created_instances:
                            shm_instance.unlink.assert_called_once()

    @patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity')
    @patch('exllamav3.model.model_tp_backend_p2p.enable_p2p_access')
    @patch('exllamav3.model.model_tp_backend_p2p.cuda_host_register')
    def test_p2p_context_initialization(self, mock_cuda_register, mock_enable_p2p, mock_check_p2p):
        """Test P2P context initialization and error handling."""
        mock_check_p2p.return_value = True
        mock_enable_p2p.return_value = None
        
        # Mock shared memory creation
        with patch('exllamav3.model.model_tp_backend_p2p.shared_memory.SharedMemory') as mock_shm:
            def mock_shared_memory_side_effect(*args, **kwargs):
                instance = Mock()
                # Determine buffer size based on name parameter
                name = kwargs.get('name', args[1] if len(args) > 1 else '')
                size = kwargs.get('size', args[0] if len(args) > 0 else 0)
                
                # Use the largest size among all expected buffers
                buffer_size = max(size, 17 * 128 * 1024)  # At least 2.25MB
                instance.buf = bytearray(buffer_size)
                return instance
            
            mock_shm.side_effect = mock_shared_memory_side_effect
            
            # Test successful initialization
            with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_init_context') as mock_init_context:
                with patch('exllamav3.model.model_tp_backend_p2p.ext.init_p2p_context') as mock_init_p2p:
                    mock_init_p2p.return_value = 0x12345678
                    
                    backend = TPBackendP2P(
                        device=0,
                        active_devices=[0, 1],
                        output_device=0,
                        init_method="tcp://127.0.0.1:29500",
                        master=True,
                        uuid="test_uuid"
                    )
                    
                    assert backend.p2p_initialized is True
                    assert backend.p2p_context == 0x12345678
                    mock_init_p2p.assert_called_once()
            
            # Test initialization failure
            with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_init_context') as mock_init_context:
                with patch('exllamav3.model.model_tp_backend_p2p.ext.init_p2p_context') as mock_init_p2p:
                    mock_init_p2p.side_effect = RuntimeError("P2P context initialization failed")
                    
                    with pytest.raises(RuntimeError, match="P2P connectivity was detected and enabled"):
                        TPBackendP2P(
                            device=0,
                            active_devices=[0, 1],
                            output_device=0,
                            init_method="tcp://127.0.0.1:29500",
                            master=True,
                            uuid="test_uuid"
                        )


class TestBackendSelectionLogic:
    """Test backend selection and automatic detection logic."""

    def test_get_available_backends(self):
        """Test getting list of available backends."""
        backends = get_available_backends()
        
        # Should always include nccl, native, and auto
        assert "nccl" in backends
        assert "native" in backends
        assert "auto" in backends
        
        # P2P should be available if TPBackendP2P is importable
        with patch('exllamav3.model.model_tp_backend.TPBackendP2P_AVAILABLE', True):
            backends = get_available_backends()
            assert "p2p" in backends
            # P2P should have priority in auto-detection
            assert backends.index("p2p") < backends.index("nccl")

    @patch('exllamav3.model.model_tp_backend.TPBackendP2P_AVAILABLE', True)
    @patch('exllamav3.model.model_tp_backend.check_p2p_connectivity')
    @patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity')
    def test_create_tp_backend_auto_p2p_available(self, mock_check_p2p_p2p, mock_check_p2p_backend):
        """Test auto backend selection when P2P is available and connected."""
        mock_check_p2p_backend.return_value = True
        mock_check_p2p_p2p.return_value = True
        
        backend = create_tp_backend(
            backend_type="auto",
            device=0,
            active_devices=[0, 1],
            output_device=0,
            init_method="tcp://127.0.0.1:29500",
            master=True,
            uuid="test_uuid"
        )
        
        # Should return P2P backend when auto detects P2P connectivity
        assert isinstance(backend, TPBackendP2P)
        mock_check_p2p_backend.assert_called_once_with([0, 1])
        mock_check_p2p_p2p.assert_called_once_with([0, 1])

    @patch('exllamav3.model.model_tp_backend.TPBackendP2P_AVAILABLE', True)
    @patch('exllamav3.model.model_tp_backend.check_p2p_connectivity')
    def test_create_tp_backend_auto_p2p_not_available(self, mock_check_p2p):
        """Test auto backend selection when P2P is available but not connected."""
        mock_check_p2p.return_value = False
        
        with patch('exllamav3.model.model_tp_backend.TPBackendNCCL') as mock_nccl:
            mock_nccl_instance = Mock()
            mock_nccl.return_value = mock_nccl_instance
            
            backend = create_tp_backend(
                backend_type="auto",
                device=0,
                active_devices=[0, 1],
                output_device=0,
                init_method="tcp://127.0.0.1:29500",
                master=True,
                uuid="test_uuid"
            )
            
            # Should fallback to NCCL when P2P not available
            assert backend == mock_nccl_instance
            mock_check_p2p.assert_called_once_with([0, 1])

    @patch('exllamav3.model.model_tp_backend.TPBackendP2P_AVAILABLE', True)
    @patch('exllamav3.model.model_tp_backend.check_p2p_connectivity')
    @patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity')
    def test_create_tp_backend_explicit_p2p(self, mock_check_p2p_p2p, mock_check_p2p_backend):
        """Test explicit P2P backend selection."""
        mock_check_p2p_backend.return_value = True
        mock_check_p2p_p2p.return_value = True
        
        backend = create_tp_backend(
            backend_type="p2p",
            device=0,
            active_devices=[0, 1],
            output_device=0,
            init_method="tcp://127.0.0.1:29500",
            master=True,
            uuid="test_uuid"
        )
        
        assert isinstance(backend, TPBackendP2P)

    @patch('exllamav3.model.model_tp_backend.TPBackendP2P_AVAILABLE', False)
    def test_create_tp_backend_explicit_p2p_not_available(self):
        """Test explicit P2P backend selection when not available."""
        with pytest.raises(ValueError, match="TPBackendP2P is not available"):
            create_tp_backend(
                backend_type="p2p",
                device=0,
                active_devices=[0, 1],
                output_device=0,
                init_method="tcp://127.0.0.1:29500",
                master=True,
                uuid="test_uuid"
            )

    @patch('exllamav3.model.model_tp_backend.TPBackendP2P_AVAILABLE', True)
    @patch('exllamav3.model.model_tp_backend.check_p2p_connectivity')
    def test_create_tp_backend_explicit_p2p_connectivity_check_fails(self, mock_check_p2p):
        """Test explicit P2P backend selection when connectivity check fails."""
        mock_check_p2p.return_value = False
        
        with pytest.raises(RuntimeError, match="P2P backend requires full peer connectivity"):
            create_tp_backend(
                backend_type="p2p",
                device=0,
                active_devices=[0, 1],
                output_device=0,
                init_method="tcp://127.0.0.1:29500",
                master=True,
                uuid="test_uuid"
            )

    def test_create_tp_backend_nccl(self):
        """Test NCCL backend selection."""
        with patch('exllamav3.model.model_tp_backend.TPBackendNCCL') as mock_nccl:
            mock_nccl_instance = Mock()
            mock_nccl.return_value = mock_nccl_instance
            
            backend = create_tp_backend(
                backend_type="nccl",
                device=0,
                active_devices=[0, 1],
                output_device=0,
                init_method="tcp://127.0.0.1:29500",
                master=True,
                uuid="test_uuid"
            )
            
            assert backend == mock_nccl_instance

    def test_create_tp_backend_native(self):
        """Test native backend selection."""
        with patch('exllamav3.model.model_tp_backend.TPBackendNative') as mock_native:
            mock_native_instance = Mock()
            mock_native.return_value = mock_native_instance
            
            backend = create_tp_backend(
                backend_type="native",
                device=0,
                active_devices=[0, 1],
                output_device=0,
                init_method="tcp://127.0.0.1:29500",
                master=True,
                uuid="test_uuid"
            )
            
            assert backend == mock_native_instance

    def test_create_tp_backend_unsupported(self):
        """Test unsupported backend type."""
        with pytest.raises(ValueError, match="Unsupported backend type"):
            create_tp_backend(
                backend_type="unsupported",
                device=0,
                active_devices=[0, 1],
                output_device=0,
                init_method="tcp://127.0.0.1:29500",
                master=True,
                uuid="test_uuid"
            )


class TestP2PBackendCommunicationPrimitives:
    """Test P2P backend communication primitives."""

    @patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity')
    @patch('exllamav3.model.model_tp_backend_p2p.enable_p2p_access')
    @patch('exllamav3.model.model_tp_backend_p2p.cuda_host_register')
    def test_barrier_synchronization(self, mock_cuda_register, mock_enable_p2p, mock_check_p2p):
        """Test P2P barrier synchronization."""
        mock_check_p2p.return_value = True
        mock_enable_p2p.return_value = None
        
        # Mock shared memory creation
        with patch('exllamav3.model.model_tp_backend_p2p.shared_memory.SharedMemory') as mock_shm:
            def mock_shared_memory_side_effect(*args, **kwargs):
                instance = Mock()
                # Determine buffer size based on name parameter
                name = kwargs.get('name', args[1] if len(args) > 1 else '')
                size = kwargs.get('size', args[0] if len(args) > 0 else 0)
                
                # Use the largest size among all expected buffers
                buffer_size = max(size, 17 * 128 * 1024)  # At least 2.25MB
                instance.buf = bytearray(buffer_size)
                return instance
            
            mock_shm.side_effect = mock_shared_memory_side_effect
            
            # Mock torch extension functions
            with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_init_context') as mock_init_context:
                with patch('exllamav3.model.model_tp_backend_p2p.ext.init_p2p_context') as mock_init_p2p:
                    mock_init_p2p.return_value = 0x12345678
                    
                    backend = TPBackendP2P(
                        device=0,
                        active_devices=[0, 1],
                        output_device=0,
                        init_method="tcp://127.0.0.1:29500",
                        master=True,
                        uuid="test_uuid"
                    )
                    
                    # Mock the P2P barrier function
                    with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_barrier_full_p2p') as mock_barrier:
                        with patch('exllamav3.model.model_tp_backend_p2p.uintptr_t', side_effect=[0, 1]):
                            abort_flag = torch.zeros((1,), device=0, dtype=torch.int32)
                            backend.fwd_barrier()
                            
                            mock_barrier.assert_called_once()
                            args, kwargs = mock_barrier.call_args
                            assert args[0] == backend.p2p_context  # context
                            assert args[1] == [0, 1]  # devices
                            assert args[2] == 0  # this_device
                            assert args[3] == abort_flag  # abort_flag

    @patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity')
    @patch('exllamav3.model.model_tp_backend_p2p.enable_p2p_access')
    @patch('exllamav3.model.model_tp_backend_p2p.cuda_host_register')
    def test_broadcast_operation(self, mock_cuda_register, mock_enable_p2p, mock_check_p2p):
        """Test P2P broadcast operation."""
        mock_check_p2p.return_value = True
        mock_enable_p2p.return_value = None
        
        # Mock shared memory creation
        with patch('exllamav3.model.model_tp_backend_p2p.shared_memory.SharedMemory') as mock_shm:
            def mock_shared_memory_side_effect(*args, **kwargs):
                instance = Mock()
                # Determine buffer size based on name parameter
                name = kwargs.get('name', args[1] if len(args) > 1 else '')
                size = kwargs.get('size', args[0] if len(args) > 0 else 0)
                
                # Use the largest size among all expected buffers
                buffer_size = max(size, 17 * 128 * 1024)  # At least 2.25MB
                instance.buf = bytearray(buffer_size)
                return instance
            
            mock_shm.side_effect = mock_shared_memory_side_effect
            
            # Mock torch extension functions
            with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_init_context') as mock_init_context:
                with patch('exllamav3.model.model_tp_backend_p2p.ext.init_p2p_context') as mock_init_p2p:
                    mock_init_p2p.return_value = 0x12345678
                    
                    backend = TPBackendP2P(
                        device=0,
                        active_devices=[0, 1],
                        output_device=0,
                        init_method="tcp://127.0.0.1:29500",
                        master=True,
                        uuid="test_uuid"
                    )
                    
                    # Test small tensor broadcast (should use shared memory)
                    small_tensor = torch.randn(100, device=0)
                    with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_broadcast_ll') as mock_broadcast_ll:
                        backend.broadcast(small_tensor, src_device=0)
                        mock_broadcast_ll.assert_called_once()
                    
                    # Test large tensor broadcast (should use P2P)
                    large_tensor = torch.randn(10000, device=0)
                    with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_broadcast_full_p2p') as mock_broadcast_p2p:
                        with patch('exllamav3.model.model_tp_backend_p2p.uintptr_t', side_effect=[0, 1]):
                            backend.broadcast(large_tensor, src_device=0)
                            mock_broadcast_p2p.assert_called_once()

    @patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity')
    @patch('exllamav3.model.model_tp_backend_p2p.enable_p2p_access')
    @patch('exllamav3.model.model_tp_backend_p2p.cuda_host_register')
    def test_all_reduce_operation(self, mock_cuda_register, mock_enable_p2p, mock_check_p2p):
        """Test P2P all-reduce operation."""
        mock_check_p2p.return_value = True
        mock_enable_p2p.return_value = None
        
        # Mock shared memory creation
        with patch('exllamav3.model.model_tp_backend_p2p.shared_memory.SharedMemory') as mock_shm:
            def mock_shared_memory_side_effect(*args, **kwargs):
                instance = Mock()
                # Determine buffer size based on name parameter
                name = kwargs.get('name', args[1] if len(args) > 1 else '')
                size = kwargs.get('size', args[0] if len(args) > 0 else 0)
                
                # Use the largest size among all expected buffers
                buffer_size = max(size, 17 * 128 * 1024)  # At least 2.25MB
                instance.buf = bytearray(buffer_size)
                return instance
            
            mock_shm.side_effect = mock_shared_memory_side_effect
            
            # Mock torch extension functions
            with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_init_context') as mock_init_context:
                with patch('exllamav3.model.model_tp_backend_p2p.ext.init_p2p_context') as mock_init_p2p:
                    mock_init_p2p.return_value = 0x12345678
                    
                    backend = TPBackendP2P(
                        device=0,
                        active_devices=[0, 1],
                        output_device=0,
                        init_method="tcp://127.0.0.1:29500",
                        master=True,
                        uuid="test_uuid"
                    )
                    
                    # Test all-reduce operation
                    tensor = torch.randn(1000, device=0)
                    with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_all_reduce_full_p2p') as mock_all_reduce:
                        with patch('exllamav3.model.model_tp_backend_p2p.uintptr_t', side_effect=[0, 1]):
                            backend.all_reduce(tensor)
                            mock_all_reduce.assert_called_once()
                            args, kwargs = mock_all_reduce.call_args
                            assert args[0] == backend.p2p_context  # context
                            assert args[1] == [0, 1]  # devices
                            assert args[2] == 0  # this_device
                            assert args[4] is tensor  # tensor
                            assert args[5] == backend.ptr_b  # shbuf
                            assert args[6] == backend.shbuf_size  # shbuf_size

    @patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity')
    @patch('exllamav3.model.model_tp_backend_p2p.enable_p2p_access')
    @patch('exllamav3.model.model_tp_backend_p2p.cuda_host_register')
    def test_gather_operation(self, mock_cuda_register, mock_enable_p2p, mock_check_p2p):
        """Test P2P gather operation."""
        mock_check_p2p.return_value = True
        mock_enable_p2p.return_value = None
        
        # Mock shared memory creation
        with patch('exllamav3.model.model_tp_backend_p2p.shared_memory.SharedMemory') as mock_shm:
            def mock_shared_memory_side_effect(*args, **kwargs):
                instance = Mock()
                # Determine buffer size based on name parameter
                name = kwargs.get('name', args[1] if len(args) > 1 else '')
                size = kwargs.get('size', args[0] if len(args) > 0 else 0)
                
                # Use the largest size among all expected buffers
                buffer_size = max(size, 17 * 128 * 1024)  # At least 2.25MB
                instance.buf = bytearray(buffer_size)
                return instance
            
            mock_shm.side_effect = mock_shared_memory_side_effect
            
            # Mock torch extension functions
            with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_init_context') as mock_init_context:
                with patch('exllamav3.model.model_tp_backend_p2p.ext.init_p2p_context') as mock_init_p2p:
                    mock_init_p2p.return_value = 0x12345678
                    
                    backend = TPBackendP2P(
                        device=0,
                        active_devices=[0, 1],
                        output_device=0,
                        init_method="tcp://127.0.0.1:29500",
                        master=True,
                        uuid="test_uuid"
                    )
                    
                    # Test gather operation on output device
                    tensor = torch.randn(1000, device=0)
                    out_tensor = torch.randn(2000, device=0)
                    ldims = [1000, 1000]
                    
                    with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_gather_full_p2p') as mock_gather:
                        with patch('exllamav3.model.model_tp_backend_p2p.uintptr_t', side_effect=[0, 1]):
                            backend.gather(tensor, out_tensor, None, 0, ldims)
                            mock_gather.assert_called_once()
                            args, kwargs = mock_gather.call_args
                            assert args[0] == backend.p2p_context  # context
                            assert args[1] == [0, 1]  # devices
                            assert args[2] == 0  # this_device
                            assert args[3] == 0  # out_device
                            assert args[4] is tensor  # tensor
                            assert args[5] is out_tensor  # out_tensor
                            assert args[6] == ldims  # ldims

    @patch('exllamav3.model.model_tp_backend_p2p.check_p2p_connectivity')
    @patch('exllamav3.model.model_tp_backend_p2p.enable_p2p_access')
    @patch('exllamav3.model.model_tp_backend_p2p.cuda_host_register')
    def test_gather_operation_non_output_device(self, mock_cuda_register, mock_enable_p2p, mock_check_p2p):
        """Test P2P gather operation on non-output device."""
        mock_check_p2p.return_value = True
        mock_enable_p2p.return_value = None
        
        # Mock shared memory creation
        with patch('exllamav3.model.model_tp_backend_p2p.shared_memory.SharedMemory') as mock_shm:
            def mock_shared_memory_side_effect(*args, **kwargs):
                instance = Mock()
                # Determine buffer size based on name parameter
                name = kwargs.get('name', args[1] if len(args) > 1 else '')
                size = kwargs.get('size', args[0] if len(args) > 0 else 0)
                
                # Use the largest size among all expected buffers
                buffer_size = max(size, 17 * 128 * 1024)  # At least 2.25MB
                instance.buf = bytearray(buffer_size)
                return instance
            
            mock_shm.side_effect = mock_shared_memory_side_effect
            
            # Mock torch extension functions
            with patch('exllamav3.model.model_tp_backend_p2p.ext.pg_init_context') as mock_init_context:
                with patch('exllamav3.model.model_tp_backend_p2p.ext.init_p2p_context') as mock_init_p2p:
                    mock_init_p2p.return_value = 0x12345678
                    
                    backend = TPBackendP2P(
                        device=1,
                        active_devices=[0, 1],
                        output_device=0,
                        init_method="tcp://127.0.0.1:29500",
                        master=False,
                        uuid="test_uuid"
                    )
                    
                    # Test gather operation on non-output device
                    tensor = torch.randn(1000, device=1)
                    
                    with patch.object(backend, '_p2p_send_to_output') as mock_send:
                        backend.gather(tensor, None, None, 0, [1000, 1000])
                        mock_send.assert_called_once_with(tensor, 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])