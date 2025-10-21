"""
Tests for P2P Monitoring Tools

This module contains tests for the P2P monitoring, profiling, and debugging tools.
"""

import unittest
import torch
import numpy as np
import time
import tempfile
import shutil
import os
from unittest.mock import Mock, patch

# Add the exllamav3 directory to the path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from exllamav3.util.p2p_monitor import P2PMonitor, P2POperationMetrics, P2PDeviceMetrics, P2PTopologyMetrics
    from exllamav3.util.p2p_profiler import P2PProfiler, P2PProfileResult, P2PComparisonResult
    from exllamav3.util.p2p_debug import P2PDebugger, P2PDebugEvent, P2PErrorInfo, P2PCommunicationTrace
    from exllamav3.util.p2p_config import P2PConfig, P2PConfigManager
    from exllamav3.model.model_tp_p2p import P2PTopology
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False


@unittest.skipUnless(MONITORING_AVAILABLE, "P2P monitoring tools not available")
class TestP2PMonitor(unittest.TestCase):
    """Test cases for P2P Monitor."""
    
    def setUp(self):
        """Set up test environment."""
        self.active_devices = [0, 1]
        self.temp_dir = tempfile.mkdtemp()
        self.monitor = P2PMonitor(
            active_devices=self.active_devices,
            monitoring_level="basic",
            max_history_size=100,
            enable_real_time=False,
            output_dir=self.temp_dir
        )
    
    def tearDown(self):
        """Clean up test environment."""
        self.monitor.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_monitor_initialization(self):
        """Test monitor initialization."""
        self.assertEqual(self.monitor.active_devices, self.active_devices)
        self.assertEqual(self.monitor.num_devices, 2)
        self.assertEqual(self.monitor.monitoring_level, "basic")
        self.assertFalse(self.monitor.enable_real_time)
    
    def test_record_operation(self):
        """Test recording operations."""
        tensor = torch.randn(100, 100)
        start_time = time.time()
        time.sleep(0.01)  # Small delay
        end_time = time.time()
        
        self.monitor.record_operation(
            operation_type="broadcast",
            device_id=0,
            peer_device=1,
            tensor=tensor,
            start_time=start_time,
            end_time=end_time,
            algorithm="ring",
            success=True
        )
        
        # Check operation history
        operations = self.monitor.get_operation_history()
        self.assertEqual(len(operations), 1)
        
        op = operations[0]
        self.assertEqual(op.operation_type, "broadcast")
        self.assertEqual(op.device_id, 0)
        self.assertEqual(op.peer_device, 1)
        self.assertEqual(op.algorithm, "ring")
        self.assertTrue(op.success)
    
    def test_device_metrics(self):
        """Test device metrics tracking."""
        # Record some operations
        tensor = torch.randn(100, 100)
        for i in range(5):
            self.monitor.record_operation(
                operation_type="all_reduce",
                device_id=0,
                peer_device=None,
                tensor=tensor,
                start_time=time.time(),
                end_time=time.time() + 0.01,
                algorithm="binary_tree",
                success=True
            )
        
        # Check device metrics
        metrics = self.monitor.get_device_metrics(0)
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.total_operations, 5)
        self.assertEqual(metrics.successful_operations, 5)
        self.assertEqual(metrics.failed_operations, 0)
    
    def test_performance_summary(self):
        """Test performance summary generation."""
        # Record some operations
        tensor = torch.randn(100, 100)
        for i in range(3):
            self.monitor.record_operation(
                operation_type="broadcast",
                device_id=0,
                peer_device=1,
                tensor=tensor,
                start_time=time.time(),
                end_time=time.time() + 0.01,
                algorithm="ring",
                success=True
            )
        
        summary = self.monitor.get_performance_summary()
        self.assertIn("monitoring_info", summary)
        self.assertIn("device_metrics", summary)
        self.assertIn("operation_stats", summary)
        self.assertIn("performance_analysis", summary)
    
    def test_bottleneck_identification(self):
        """Test bottleneck identification."""
        # Record some operations with varying performance
        tensor = torch.randn(100, 100)
        
        # Fast operation
        self.monitor.record_operation(
            operation_type="broadcast",
            device_id=0,
            peer_device=1,
            tensor=tensor,
            start_time=time.time(),
            end_time=time.time() + 0.001,
            algorithm="ring",
            success=True
        )
        
        # Slow operation
        self.monitor.record_operation(
            operation_type="all_reduce",
            device_id=0,
            peer_device=None,
            tensor=tensor,
            start_time=time.time(),
            end_time=time.time() + 0.1,
            algorithm="binary_tree",
            success=True
        )
        
        bottlenecks = self.monitor.identify_bottlenecks()
        self.assertIsInstance(bottlenecks, dict)
        self.assertIn("slow_operations", bottlenecks)
    
    def test_optimization_suggestions(self):
        """Test optimization suggestions."""
        # Record some operations
        tensor = torch.randn(100, 100)
        self.monitor.record_operation(
            operation_type="broadcast",
            device_id=0,
            peer_device=1,
            tensor=tensor,
            start_time=time.time(),
            end_time=time.time() + 0.01,
            algorithm="ring",
            success=True
        )
        
        suggestions = self.monitor.get_optimization_suggestions()
        self.assertIsInstance(suggestions, list)
    
    def test_export_metrics(self):
        """Test metrics export."""
        # Record some operations
        tensor = torch.randn(100, 100)
        self.monitor.record_operation(
            operation_type="broadcast",
            device_id=0,
            peer_device=1,
            tensor=tensor,
            start_time=time.time(),
            end_time=time.time() + 0.01,
            algorithm="ring",
            success=True
        )
        
        export_path = self.monitor.export_metrics()
        self.assertTrue(os.path.exists(export_path))
        
        # Check file content
        with open(export_path, 'r') as f:
            content = f.read()
            self.assertIn("summary", content)
            self.assertIn("operation_history", content)


@unittest.skipUnless(MONITORING_AVAILABLE, "P2P monitoring tools not available")
class TestP2PProfiler(unittest.TestCase):
    """Test cases for P2P Profiler."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.profiler = P2PProfiler(
            output_dir=self.temp_dir
        )
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_profiler_initialization(self):
        """Test profiler initialization."""
        self.assertIsNotNone(self.profiler)
        self.assertEqual(self.profiler.output_dir, self.temp_dir)
    
    def test_profile_session(self):
        """Test profiling session."""
        tensor = torch.randn(100, 100)
        
        with self.profiler.profile_session("test_session", "broadcast", "ring"):
            self.profiler.record_operation(
                tensor=tensor,
                device_id=0,
                peer_device=1,
                duration_ms=10.0,
                bandwidth_gbps=5.0,
                success=True
            )
        
        # Check session result
        result = self.profiler.get_session("test_session")
        self.assertIsNotNone(result)
        self.assertEqual(result.session_id, "test_session")
        self.assertEqual(result.operation_type, "broadcast")
        self.assertEqual(result.algorithm, "ring")
    
    def test_profile_operation(self):
        """Test profiling an operation."""
        def mock_operation(tensor):
            time.sleep(0.01)
            return tensor
        
        tensor = torch.randn(100, 100)
        result = self.profiler.profile_operation(
            operation_func=mock_operation,
            operation_type="broadcast",
            algorithm="ring",
            num_iterations=3,
            warmup_iterations=1,
            tensor=tensor,
            device_id=0
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.operation_type, "broadcast")
        self.assertEqual(result.algorithm, "ring")
        self.assertEqual(len(result.durations_ms), 3)
    
    def test_compare_algorithms(self):
        """Test algorithm comparison."""
        # Create mock sessions
        with self.profiler.profile_session("session1", "all_reduce", "ring"):
            pass
        
        with self.profiler.profile_session("session2", "all_reduce", "binary_tree"):
            pass
        
        comparison = self.profiler.compare_algorithms(
            comparison_id="test_comparison",
            operation_type="all_reduce",
            algorithms=["ring", "binary_tree"],
            sessions={"ring": "session1", "binary_tree": "session2"}
        )
        
        self.assertIsNotNone(comparison)
        self.assertEqual(comparison.comparison_id, "test_comparison")
        self.assertEqual(comparison.operation_type, "all_reduce")
    
    def test_analyze_communication_patterns(self):
        """Test communication pattern analysis."""
        # Create some mock sessions
        with self.profiler.profile_session("session1", "broadcast", "ring"):
            pass
        
        patterns = self.profiler.analyze_communication_patterns()
        self.assertIsInstance(patterns, dict)
        self.assertIn("device_pair_frequency", patterns)
        self.assertIn("tensor_size_distribution", patterns)
        self.assertIn("operation_frequency", patterns)


@unittest.skipUnless(MONITORING_AVAILABLE, "P2P monitoring tools not available")
class TestP2PDebugger(unittest.TestCase):
    """Test cases for P2P Debugger."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.debugger = P2PDebugger(
            debug_level="basic",
            max_events=100,
            output_dir=self.temp_dir
        )
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_debugger_initialization(self):
        """Test debugger initialization."""
        self.assertIsNotNone(self.debugger)
        self.assertEqual(self.debugger.debug_level, "basic")
        self.assertEqual(self.debugger.max_events, 100)
    
    def test_log_event(self):
        """Test event logging."""
        tensor = torch.randn(100, 100)
        
        self.debugger.log_event(
            event_type="operation_start",
            operation_type="broadcast",
            device_id=0,
            peer_device=1,
            tensor=tensor,
            message="Test event"
        )
        
        # Check event history
        events = self.debugger.get_events()
        self.assertEqual(len(events), 1)
        
        event = events[0]
        self.assertEqual(event.event_type, "operation_start")
        self.assertEqual(event.operation_type, "broadcast")
        self.assertEqual(event.device_id, 0)
        self.assertEqual(event.peer_device, 1)
        self.assertEqual(event.message, "Test event")
    
    def test_log_error(self):
        """Test error logging."""
        self.debugger.log_error(
            error_type="communication_error",
            operation_type="broadcast",
            device_id=0,
            peer_device=1,
            error_message="Test error",
            context={"test": "context"}
        )
        
        # Check error history
        errors = self.debugger.get_errors()
        self.assertEqual(len(errors), 1)
        
        error = errors[0]
        self.assertEqual(error.error_type, "communication_error")
        self.assertEqual(error.operation_type, "broadcast")
        self.assertEqual(error.device_id, 0)
        self.assertEqual(error.peer_device, 1)
        self.assertEqual(error.error_message, "Test error")
    
    def test_communication_trace(self):
        """Test communication tracing."""
        trace_id = "test_trace"
        participants = [0, 1]
        
        # Start trace
        self.debugger.start_trace(trace_id, "broadcast", participants)
        
        # Add trace step
        self.debugger.add_trace_step(
            trace_id=trace_id,
            step_type="data_transfer",
            device_id=0,
            peer_device=1,
            message="Data transfer"
        )
        
        # End trace
        self.debugger.end_trace(trace_id, success=True)
        
        # Check trace
        trace = self.debugger.get_trace(trace_id)
        self.assertIsNotNone(trace)
        self.assertEqual(trace.trace_id, trace_id)
        self.assertEqual(trace.operation_type, "broadcast")
        self.assertTrue(trace.success)
    
    def test_diagnose_issues(self):
        """Test issue diagnosis."""
        # Log some errors
        self.debugger.log_error(
            error_type="communication_error",
            operation_type="broadcast",
            device_id=0,
            peer_device=1,
            error_message="Test error"
        )
        
        diagnosis = self.debugger.diagnose_issues()
        self.assertIsInstance(diagnosis, dict)
        self.assertIn("issues", diagnosis)
        self.assertIn("recommendations", diagnosis)
    
    def test_export_debug_report(self):
        """Test debug report export."""
        # Log some events
        self.debugger.log_event(
            event_type="operation_start",
            operation_type="broadcast",
            device_id=0,
            message="Test event"
        )
        
        report_path = self.debugger.export_debug_report()
        self.assertTrue(os.path.exists(report_path))
        
        # Check file content
        with open(report_path, 'r') as f:
            content = f.read()
            self.assertIn("statistics", content)
            self.assertIn("diagnosis", content)


@unittest.skipUnless(MONITORING_AVAILABLE, "P2P monitoring tools not available")
class TestP2PConfig(unittest.TestCase):
    """Test cases for P2P Configuration."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "test_config.json")
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_config_manager_initialization(self):
        """Test configuration manager initialization."""
        config_manager = P2PConfigManager(self.config_file)
        self.assertIsNotNone(config_manager)
        self.assertIsNotNone(config_manager.get_config())
    
    def test_config_save_load(self):
        """Test configuration save and load."""
        config_manager = P2PConfigManager(self.config_file)
        
        # Modify config
        config_manager.update_monitoring_config(monitoring_level="detailed")
        
        # Save config
        config_manager.save_config()
        
        # Load config in new manager
        new_config_manager = P2PConfigManager(self.config_file)
        config = new_config_manager.get_config()
        
        self.assertEqual(config.monitoring.monitoring_level, "detailed")
    
    def test_config_updates(self):
        """Test configuration updates."""
        config_manager = P2PConfigManager(self.config_file)
        
        # Update monitoring config
        config_manager.update_monitoring_config(
            enable_monitoring=False,
            monitoring_level="comprehensive"
        )
        
        config = config_manager.get_config()
        self.assertFalse(config.monitoring.enable_monitoring)
        self.assertEqual(config.monitoring.monitoring_level, "comprehensive")
        
        # Update profiling config
        config_manager.update_profiling_config(
            enable_profiling=False,
            profiling_level="detailed"
        )
        
        self.assertFalse(config.profiling.enable_profiling)
        self.assertEqual(config.profiling.profiling_level, "detailed")
    
    def test_environment_config_loading(self):
        """Test loading configuration from environment."""
        # Set environment variables
        os.environ["P2P_MONITORING_LEVEL"] = "detailed"
        os.environ["P2P_ENABLE_MONITORING"] = "false"
        
        try:
            config_manager = P2PConfigManager(self.config_file)
            load_config_from_env()
            
            config = config_manager.get_config()
            self.assertEqual(config.monitoring.monitoring_level, "detailed")
            self.assertFalse(config.monitoring.enable_monitoring)
        finally:
            # Clean up environment variables
            os.environ.pop("P2P_MONITORING_LEVEL", None)
            os.environ.pop("P2P_ENABLE_MONITORING", None)


@unittest.skipUnless(MONITORING_AVAILABLE, "P2P monitoring tools not available")
class TestP2PTopologyIntegration(unittest.TestCase):
    """Test cases for P2P topology integration with monitoring."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.active_devices = [0, 1]
        
        # Mock P2P topology
        self.topology = Mock(spec=P2PTopology)
        self.topology.active_devices = self.active_devices
        self.topology.num_devices = 2
        self.topology.get_topology_summary.return_value = {
            "num_devices": 2,
            "topology_type": "fully_connected",
            "connectivity_ratio": 1.0,
            "is_fully_connected": True
        }
        
        # Create monitor with topology
        self.monitor = P2PMonitor(
            active_devices=self.active_devices,
            output_dir=self.temp_dir
        )
        self.monitor.set_topology(self.topology)
    
    def tearDown(self):
        """Clean up test environment."""
        self.monitor.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_topology_integration(self):
        """Test topology integration with monitoring."""
        # Check topology metrics
        self.assertIsNotNone(self.monitor.topology_metrics)
        self.assertEqual(self.monitor.topology_metrics.num_devices, 2)
        self.assertEqual(self.monitor.topology_metrics.topology_type, "fully_connected")
        self.assertTrue(self.monitor.topology_metrics.is_fully_connected)
    
    def test_topology_aware_monitoring(self):
        """Test topology-aware monitoring."""
        tensor = torch.randn(100, 100)
        
        # Record operation with topology context
        self.monitor.record_operation(
            operation_type="broadcast",
            device_id=0,
            peer_device=1,
            tensor=tensor,
            start_time=time.time(),
            end_time=time.time() + 0.01,
            algorithm="ring",
            success=True
        )
        
        # Check that topology info is included
        operations = self.monitor.get_operation_history()
        self.assertEqual(len(operations), 1)
        
        op = operations[0]
        self.assertIsNotNone(op.topology_info)
        self.assertIn("topology_type", op.topology_info)


class TestMonitoringIntegration(unittest.TestCase):
    """Test cases for monitoring integration with P2P backend."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @unittest.skipUnless(MONITORING_AVAILABLE, "P2P monitoring tools not available")
    @patch('exllamav3.model.model_tp_backend.MONITORING_AVAILABLE', True)
    def test_backend_monitoring_integration(self):
        """Test monitoring integration in P2P backend."""
        # This test would require actual P2P backend implementation
        # For now, we'll test the integration points
        
        from exllamav3.model.model_tp_backend import TPBackendP2P
        
        # Mock the monitoring tools
        with patch('exllamav3.model.model_tp_backend.get_global_monitor') as mock_monitor, \
             patch('exllamav3.model.model_tp_backend.get_global_profiler') as mock_profiler, \
             patch('exllamav3.model.model_tp_backend.get_global_debugger') as mock_debugger:
            
            mock_monitor.return_value = Mock()
            mock_profiler.return_value = Mock()
            mock_debugger.return_value = Mock()
            
            # Create backend (this would normally require actual GPU devices)
            # backend = TPBackendP2P(
            #     device=0,
            #     active_devices=[0, 1],
            #     output_device=0,
            #     init_method="tcp://localhost:12345",
            #     master=True,
            #     uuid="test_uuid"
            # )
            
            # Verify that monitoring tools are initialized
            mock_monitor.assert_called_once()
            mock_profiler.assert_called_once()
            mock_debugger.assert_called_once()


if __name__ == "__main__":
    unittest.main()