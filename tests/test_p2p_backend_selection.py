"""
Backend selection logic tests for P2P functionality.

This module tests:
- Automatic backend selection logic
- Manual backend specification
- Mixed connectivity scenarios
- Configuration validation
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock, call
import sys
import os

# Add the project root to sys.path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from exllamav3.model.model_tp_backend_p2p import TPBackendP2P
from exllamav3.model.model_tp_backend import (
    create_tp_backend,
    get_available_backends,
    TPBackendNCCL,
    TPBackendNative,
    TPBackendP2P_AVAILABLE
)


class TestAutomaticBackendSelectionLogic:
    """Test automatic backend selection logic."""

    def test_auto_selection_with_p2p_available_and_connected(self):
        """Test auto selection when P2P is available and connected."""
        with patch('exllamav3.model.model_tp_backend.TPBackendP2P_AVAILABLE', True):
            with patch('exllamav3.model.model_tp_backend.check_p2p_connectivity') as mock_check:
                # P2P is available and connected
                mock_check.return_value = True
                
                backend = create_tp_backend(
                    backend_type="auto",
                    device=0,
                    active_devices=[0, 1],
                    output_device=0,
                    init_method="tcp://127.0.0.1:29500",
                    master=True,
                    uuid="test_auto_p2p"
                )
                
                # Should select P2P backend
                assert isinstance(backend, TPBackendP2P)
                mock_check.assert_called_once_with([0, 1])

    def test_auto_selection_with_p2p_available_but_not_connected(self):
        """Test auto selection when P2P is available but not connected."""
        with patch('exllamav3.model.model_tp_backend.TPBackendP2P_AVAILABLE', True):
            with patch('exllamav3.model.model_tp_backend.check_p2p_connectivity') as mock_check:
                # P2P is available but not connected
                mock_check.return_value = False
                
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
                        uuid="test_auto_nccl"
                    )
                    
                    # Should fallback to NCCL
                    assert backend == mock_nccl_instance
                    mock_check.assert_called_once_with([0, 1])

    def test_auto_selection_with_p2p_not_available(self):
        """Test auto selection when P2P is not available."""
        with patch('exllamav3.model.model_tp_backend.TPBackendP2P_AVAILABLE', False):
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
                    uuid="test_auto_no_p2p"
                )
                
                # Should select NCCL when P2P not available
                assert backend == mock_nccl_instance

    def test_auto_selection_single_device(self):
        """Test auto selection with single device."""
        with patch('exllamav3.model.model_tp_backend.TPBackendP2P_AVAILABLE', True):
            with patch('exllamav3.model.model_tp_backend.check_p2p_connectivity') as mock_check:
                # Single device should not trigger P2P check
                backend = create_tp_backend(
                    backend_type="auto",
                    device=0,
                    active_devices=[0],
                    output_device=0,
                    init_method="tcp://127.0.0.1:29500",
                    master=True,
                    uuid="test_auto_single"
                )
                
                # Should select NCCL for single device
                assert isinstance(backend, TPBackendNCCL)
                mock_check.assert_not_called()

    def test_auto_selection_cpu_process(self):
        """Test auto selection for CPU process."""
        with patch('exllamav3.model.model_tp_backend.TPBackendP2P_AVAILABLE', True):
            with patch('exllamav3.model.model_tp_backend.TPBackendNCCL') as mock_nccl:
                mock_nccl_instance = Mock()
                mock_nccl.return_value = mock_nccl_instance
                
                backend = create_tp_backend(
                    backend_type="auto",
                    device=-1,  # CPU process
                    active_devices=[-1],
                    output_device=-1,
                    init_method="tcp://127.0.0.1:29500",
                    master=True,
                    uuid="test_auto_cpu"
                )
                
                # Should select NCCL for CPU process
                assert backend == mock_nccl_instance

    def test_auto_selection_with_multiple_device_configs(self):
        """Test auto selection with different device configurations."""
        test_configs = [
            ([0, 1], True),   # 2 devices, connected
            ([0, 1, 2], False),  # 3 devices, not connected
            ([0, 1, 2, 3], True),  # 4 devices, connected
            ([0, 1, 2, 3, 4], False),  # 5 devices, not connected
        ]
        
        with patch('exllamav3.model.model_tp_backend.TPBackendP2P_AVAILABLE', True):
            for devices, connected in test_configs:
                with patch('exllamav3.model.model_tp_backend.check_p2p_connectivity') as mock_check:
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
                            uuid=f"test_auto_multi_{len(devices)}"
                        )
                        
                        if connected:
                            # Should select P2P when connected
                            assert isinstance(backend, TPBackendP2P)
                        else:
                            # Should fallback to NCCL when not connected
                            assert backend == mock_nccl_instance
                        
                        mock_check.assert_called_once_with(devices)


class TestManualBackendSpecification:
    """Test manual backend specification."""

    def test_explicit_p2p_selection(self):
        """Test explicit P2P backend selection."""
        with patch('exllamav3.model.model_tp_backend.TPBackendP2P_AVAILABLE', True):
            with patch('exllamav3.model.model_tp_backend.check_p2p_connectivity') as mock_check:
                mock_check.return_value = True
                
                backend = create_tp_backend(
                    backend_type="p2p",
                    device=0,
                    active_devices=[0, 1, 2],
                    output_device=0,
                    init_method="tcp://127.0.0.1:29500",
                    master=True,
                    uuid="test_explicit_p2p"
                )
                
                # Should select P2P backend
                assert isinstance(backend, TPBackendP2P)
                mock_check.assert_called_once_with([0, 1, 2])

    def test_explicit_p2p_selection_fails_when_not_connected(self):
        """Test explicit P2P selection fails when not connected."""
        with patch('exllamav3.model.model_tp_backend.TPBackendP2P_AVAILABLE', True):
            with patch('exllamav3.model.model_tp_backend.check_p2p_connectivity') as mock_check:
                mock_check.return_value = False
                
                with pytest.raises(RuntimeError, match="P2P backend requires full peer connectivity"):
                    create_tp_backend(
                        backend_type="p2p",
                        device=0,
                        active_devices=[0, 1],
                        output_device=0,
                        init_method="tcp://127.0.0.1:29500",
                        master=True,
                        uuid="test_explicit_p2p_fail"
                    )

    def test_explicit_p2p_selection_fails_when_not_available(self):
        """Test explicit P2P selection fails when not available."""
        with patch('exllamav3.model.model_tp_backend.TPBackendP2P_AVAILABLE', False):
            with pytest.raises(ValueError, match="TPBackendP2P is not available"):
                create_tp_backend(
                    backend_type="p2p",
                    device=0,
                    active_devices=[0, 1],
                    output_device=0,
                    init_method="tcp://127.0.0.1:29500",
                    master=True,
                    uuid="test_explicit_p2p_unavailable"
                )

    def test_explicit_nccl_selection(self):
        """Test explicit NCCL backend selection."""
        with patch('exllamav3.model.model_tp_backend.TPBackendNCCL') as mock_nccl:
            mock_nccl_instance = Mock()
            mock_nccl.return_value = mock_nccl_instance
            
            backend = create_tp_backend(
                backend_type="nccl",
                device=0,
                active_devices=[0, 1, 2],
                output_device=1,
                init_method="tcp://127.0.0.1:29500",
                master=False,
                uuid="test_explicit_nccl"
            )
            
            # Should select NCCL backend
            assert backend == mock_nccl_instance
            mock_nccl.assert_called_once()

    def test_explicit_native_selection(self):
        """Test explicit native backend selection."""
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
                uuid="test_explicit_native"
            )
            
            # Should select native backend
            assert backend == mock_native_instance
            mock_native.assert_called_once()

    def test_explicit_unsupported_selection(self):
        """Test explicit unsupported backend selection."""
        with pytest.raises(ValueError, match="Unsupported backend type"):
            create_tp_backend(
                backend_type="unsupported",
                device=0,
                active_devices=[0, 1],
                output_device=0,
                init_method="tcp://127.0.0.1:29500",
                master=True,
                uuid="test_explicit_unsupported"
            )


class TestMixedConnectivityScenarios:
    """Test mixed connectivity scenarios."""

    def test_mixed_connectivity_detection(self):
        """Test detection of mixed connectivity scenarios."""
        # Simulate different connectivity patterns
        connectivity_scenarios = [
            # (devices, connectivity_matrix, expected_result)
            ([0, 1], [[True, True], [True, True]], True),  # Fully connected
            ([0, 1], [[True, False], [False, True]], False),  # Not connected
            ([0, 1, 2], [[True, True, True], [True, True, True], [True, True, True]], True),  # Fully connected
            ([0, 1, 2], [[True, True, False], [True, True, False], [False, False, True]], False),  # Partially connected
        ]
        
        with patch('exllamav3.model.model_tp_backend.TPBackendP2P_AVAILABLE', True):
            for devices, matrix, expected_connected in connectivity_scenarios:
                with patch('exllamav3.model.model_tp_backend.check_p2p_connectivity') as mock_check:
                    mock_check.return_value = expected_connected
                    
                    if expected_connected:
                        # Should be able to create P2P backend
                        backend = create_tp_backend(
                            backend_type="p2p",
                            device=0,
                            active_devices=devices,
                            output_device=0,
                            init_method="tcp://127.0.0.1:29500",
                            master=True,
                            uuid=f"test_mixed_{len(devices)}"
                        )
                        assert isinstance(backend, TPBackendP2P)
                    else:
                        # Should fail to create P2P backend
                        with pytest.raises(RuntimeError, match="P2P backend requires full peer connectivity"):
                            create_tp_backend(
                                backend_type="p2p",
                                device=0,
                                active_devices=devices,
                                output_device=0,
                                init_method="tcp://127.0.0.1:29500",
                                master=True,
                                uuid=f"test_mixed_fail_{len(devices)}"
                            )
                    
                    mock_check.assert_called_once_with(devices)

    def test_graceful_degradation(self):
        """Test graceful degradation when P2P connectivity is partial."""
        with patch('exllamav3.model.model_tp_backend.TPBackendP2P_AVAILABLE', True):
            with patch('exllamav3.model.model_tp_backend.check_p2p_connectivity') as mock_check:
                # Simulate connectivity check that sometimes fails
                call_count = 0
                def mock_connectivity_check(devices):
                    nonlocal call_count
                    call_count += 1
                    # Fail on first call, succeed on second
                    return call_count > 1
                
                mock_check.side_effect = mock_connectivity_check
                
                # First attempt should fail
                with pytest.raises(RuntimeError, match="P2P backend requires full peer connectivity"):
                    create_tp_backend(
                        backend_type="p2p",
                        device=0,
                        active_devices=[0, 1],
                        output_device=0,
                        init_method="tcp://127.0.0.1:29500",
                        master=True,
                        uuid="test_degradation_1"
                    )
                
                # Reset call counter and try auto selection
                call_count = 0
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
                        uuid="test_degradation_2"
                    )
                    
                    # Should fallback to NCCL
                    assert backend == mock_nccl_instance

    def test_connectivity_verification(self):
        """Test connectivity verification in mixed scenarios."""
        with patch('exllamav3.model.model_tp_backend.TPBackendP2P_AVAILABLE', True):
            with patch('exllamav3.model.model_tp_backend.check_p2p_connectivity') as mock_check:
                # Test different device configurations
                test_configs = [
                    ([0, 1], True),   # 2 devices, connected
                    ([0, 1, 2], False),  # 3 devices, not connected
                    ([0, 1, 2, 3], True),  # 4 devices, connected
                    ([0, 1, 2, 3, 4], False),  # 5 devices, not connected
                ]
                
                for devices, connected in test_configs:
                    # Reset mock for each test
                    mock_check.reset_mock()
                    mock_check.return_value = connected
                    
                    if connected:
                        # Should succeed with P2P
                        backend = create_tp_backend(
                            backend_type="p2p",
                            device=0,
                            active_devices=devices,
                            output_device=0,
                            init_method="tcp://127.0.0.1:29500",
                            master=True,
                            uuid=f"test_verify_{len(devices)}"
                        )
                        assert isinstance(backend, TPBackendP2P)
                    else:
                        # Should fail with P2P
                        with pytest.raises(RuntimeError, match="P2P backend requires full peer connectivity"):
                            create_tp_backend(
                                backend_type="p2p",
                                device=0,
                                active_devices=devices,
                                output_device=0,
                                init_method="tcp://127.0.0.1:29500",
                                master=True,
                                uuid=f"test_verify_fail_{len(devices)}"
                            )
                    
                    # Verify connectivity check was called with correct devices
                    mock_check.assert_called_once_with(devices)


class TestConfigurationValidation:
    """Test configuration validation for backend selection."""

    def test_device_configuration_validation(self):
        """Test validation of device configurations."""
        with patch('exllamav3.model.model_tp_backend.TPBackendP2P_AVAILABLE', True):
            with patch('exllamav3.model.model_tp_backend.check_p2p_connectivity') as mock_check:
                mock_check.return_value = True
                
                # Test various device configurations
                device_configs = [
                    ([0], True),      # Single device
                    ([0, 1], True),   # Two devices
                    ([0, 1, 2], True), # Three devices
                    ([0, 1, 2, 3], True), # Four devices
                    ([0, 2, 4], True), # Non-contiguous devices
                ]
                
                for devices, should_succeed in device_configs:
                    try:
                        backend = create_tp_backend(
                            backend_type="p2p",
                            device=devices[0],  # Use first device as current device
                            active_devices=devices,
                            output_device=devices[0],
                            init_method="tcp://127.0.0.1:29500",
                            master=True,
                            uuid=f"test_device_config_{len(devices)}"
                        )
                        
                        if should_succeed:
                            assert isinstance(backend, TPBackendP2P)
                            assert backend.device == devices[0]
                            assert backend.active_devices == devices
                            assert backend.world_size == len(devices)
                            assert backend.rank == 0
                        
                    except Exception as e:
                        if should_succeed:
                            raise AssertionError(f"Configuration {devices} should succeed but failed: {e}")

    def test_output_device_validation(self):
        """Test validation of output device configuration."""
        with patch('exllamav3.model.model_tp_backend.TPBackendP2P_AVAILABLE', True):
            with patch('exllamav3.model.model_tp_backend.check_p2p_connectivity') as mock_check:
                mock_check.return_value = True
                
                devices = [0, 1, 2]
                
                # Test different output device configurations
                for output_device in devices:
                    backend = create_tp_backend(
                        backend_type="p2p",
                        device=0,
                        active_devices=devices,
                        output_device=output_device,
                        init_method="tcp://127.0.0.1:29500",
                        master=True,
                        uuid=f"test_output_device_{output_device}"
                    )
                    
                    assert isinstance(backend, TPBackendP2P)
                    assert backend.output_device == output_device

    def test_init_method_validation(self):
        """Test validation of initialization methods."""
        with patch('exllamav3.model.model_tp_backend.TPBackendP2P_AVAILABLE', True):
            with patch('exllamav3.model.model_tp_backend.check_p2p_connectivity') as mock_check:
                mock_check.return_value = True
                
                init_methods = [
                    "tcp://127.0.0.1:29500",
                    "tcp://192.168.1.100:29500",
                    "env://",
                ]
                
                for init_method in init_methods:
                    backend = create_tp_backend(
                        backend_type="p2p",
                        device=0,
                        active_devices=[0, 1],
                        output_device=0,
                        init_method=init_method,
                        master=True,
                        uuid=f"test_init_method_{init_method}"
                    )
                    
                    assert isinstance(backend, TPBackendP2P)
                    assert backend.init_method == init_method

    def test_master_slave_configuration(self):
        """Test master/slave configuration validation."""
        with patch('exllamav3.model.model_tp_backend.TPBackendP2P_AVAILABLE', True):
            with patch('exllamav3.model.model_tp_backend.check_p2p_connectivity') as mock_check:
                mock_check.return_value = True
                
                # Test both master and slave configurations
                for master in [True, False]:
                    backend = create_tp_backend(
                        backend_type="p2p",
                        device=0,
                        active_devices=[0, 1],
                        output_device=0,
                        init_method="tcp://127.0.0.1:29500",
                        master=master,
                        uuid=f"test_master_{master}"
                    )
                    
                    assert isinstance(backend, TPBackendP2P)
                    assert backend.master == master

    def test_uuid_configuration(self):
        """Test UUID configuration validation."""
        with patch('exllamav3.model.model_tp_backend.TPBackendP2P_AVAILABLE', True):
            with patch('exllamav3.model.model_tp_backend.check_p2p_connectivity') as mock_check:
                mock_check.return_value = True
                
                uuids = [
                    "test_uuid_1",
                    "test-uuid-2",
                    "test.uuid.3",
                    "1234567890",
                ]
                
                for uuid in uuids:
                    backend = create_tp_backend(
                        backend_type="p2p",
                        device=0,
                        active_devices=[0, 1],
                        output_device=0,
                        init_method="tcp://127.0.0.1:29500",
                        master=True,
                        uuid=uuid
                    )
                    
                    assert isinstance(backend, TPBackendP2P)
                    assert backend.uuid == uuid

    def test_shbuf_size_configuration(self):
        """Test shared buffer size configuration."""
        with patch('exllamav3.model.model_tp_backend.TPBackendP2P_AVAILABLE', True):
            with patch('exllamav3.model.model_tp_backend.check_p2p_connectivity') as mock_check:
                mock_check.return_value = True
                
                buffer_sizes = [
                    16 * 1024 * 1024,    # 16MB
                    32 * 1024 * 1024,    # 32MB
                    64 * 1024 * 1024,    # 64MB
                    128 * 1024 * 1024,   # 128MB
                ]
                
                for shbuf_size in buffer_sizes:
                    backend = create_tp_backend(
                        backend_type="p2p",
                        device=0,
                        active_devices=[0, 1],
                        output_device=0,
                        init_method="tcp://127.0.0.1:29500",
                        master=True,
                        uuid="test_buffer_size",
                        shbuf_size=shbuf_size
                    )
                    
                    assert isinstance(backend, TPBackendP2P)
                    assert backend.shbuf_size == shbuf_size


class TestBackendPriorityAndFallback:
    """Test backend priority and fallback logic."""

    def test_backend_priority_order(self):
        """Test that backends are selected in the correct priority order."""
        with patch('exllamav3.model.model_tp_backend.TPBackendP2P_AVAILABLE', True):
            with patch('exllamav3.model.model_tp_backend.check_p2p_connectivity') as mock_check:
                mock_check.return_value = True
                
                # Test available backends
                backends = get_available_backends()
                
                # P2P should have priority when available
                assert "p2p" in backends
                assert backends.index("p2p") < backends.index("nccl")
                assert "auto" in backends  # Auto should always be available
                assert "native" in backends  # Native should always be available

    def test_fallback_chain(self):
        """Test the complete fallback chain."""
        with patch('exllamav3.model.model_tp_backend.TPBackendP2P_AVAILABLE', True):
            with patch('exllamav3.model.model_tp_backend.check_p2p_connectivity') as mock_check:
                # Test fallback chain: P2P -> NCCL -> Native
                fallback_scenarios = [
                    (True, "p2p"),   # P2P available and connected
                    (False, "nccl"), # P2P available but not connected -> NCCL
                ]
                
                for connected, expected_backend in fallback_scenarios:
                    mock_check.return_value = connected
                    
                    with patch('exllamav3.model.model_tp_backend.TPBackendNCCL') as mock_nccl:
                        with patch('exllamav3.model.model_tp_backend.TPBackendNative') as mock_native:
                            mock_nccl_instance = Mock()
                            mock_nccl.return_value = mock_nccl_instance
                            
                            mock_native_instance = Mock()
                            mock_native.return_value = mock_native_instance
                            
                            backend = create_tp_backend(
                                backend_type="auto",
                                device=0,
                                active_devices=[0, 1],
                                output_device=0,
                                init_method="tcp://127.0.0.1:29500",
                                master=True,
                                uuid="test_fallback_chain"
                            )
                            
                            if expected_backend == "p2p":
                                assert isinstance(backend, TPBackendP2P)
                            elif expected_backend == "nccl":
                                assert backend == mock_nccl_instance

    def test_auto_selection_with_preference(self):
        """Test auto selection with user preferences."""
        with patch('exllamav3.model.model_tp_backend.TPBackendP2P_AVAILABLE', True):
            with patch('exllamav3.model.model_tp_backend.check_p2p_connectivity') as mock_check:
                mock_check.return_value = False
                
                # Even when P2P is available but not connected, auto should prefer NCCL
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
                        uuid="test_preference"
                    )
                    
                    # Should prefer NCCL over native
                    assert backend == mock_nccl_instance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])