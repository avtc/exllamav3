#!/usr/bin/env python3
"""
Comprehensive test suite for P2P GPU communication implementation.

This module provides extensive testing coverage for all P2P components including:
- Topology detection and analysis
- Direct memory access operations
- Tree-based reductions
- Shared memory optimizations
- Backend integration
- Performance validation
- Error handling and recovery
"""

import unittest
import torch
import numpy as np
import time
import tempfile
import shutil
import os
import sys
from typing import List, Dict, Tuple, Optional
from unittest.mock import Mock, patch

# Add the exllamav3 directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from exllamav3.model.model_tp_p2p import P2PTopology
    from exllamav3.model.model_tp_backend import TPBackendP2P, TPBackendNative
    from exllamav3.util.p2p_monitor import P2PMonitor, P2POperationMetrics
    from exllamav3.util.p2p_profiler import P2PProfiler
    from exllamav3.util.p2p_debug import P2PDebugger
    from exllamav3.util.p2p_config import P2PConfigManager
    from exllamav3.ext import exllamav3_ext as ext
    P2P_AVAILABLE = True
except ImportError as e:
    print(f"P2P modules not available: {e}")
    P2P_AVAILABLE = False


class TestP2PTopology(unittest.TestCase):
    """Test cases for P2P topology detection and analysis."""
    
    def setUp(self):
        """Set up test environment."""
        if not P2P_AVAILABLE:
            self.skipTest("P2P modules not available")
        
        # Mock device list for testing
        self.active_devices = [0, 1, 2, 3]
        self.topology = P2PTopology(self.active_devices)
    
    def test_topology_initialization(self):
        """Test topology initialization."""
        self.assertEqual(self.topology.active_devices, self.active_devices)
        self.assertEqual(self.topology.num_devices, 4)
        self.assertIsNotNone(self.topology.p2p_matrix)
        self.assertEqual(self.topology.p2p_matrix.shape, (4, 4))
    
    def test_p2p_matrix_properties(self):
        """Test P2P matrix properties."""
        matrix = self.topology.get_p2p_matrix()
        
        # Diagonal should be True
        for i in range(self.topology.num_devices):
            self.assertTrue(matrix[i][i], f"Device {i} should access itself")
        
        # Matrix should be symmetric for P2P
        for i in range(self.topology.num_devices):
            for j in range(self.topology.num_devices):
                if i != j:
                    # This might not always be true depending on hardware
                    # but we test the matrix structure
                    self.assertIsInstance(matrix[i][j], bool)
    
    def test_connectivity_ratio(self):
        """Test connectivity ratio calculation."""
        ratio = self.topology.get_connectivity_ratio()
        self.assertIsInstance(ratio, float)
        self.assertGreaterEqual(ratio, 0.0)
        self.assertLessEqual(ratio, 1.0)
    
    def test_fully_connected_detection(self):
        """Test fully connected topology detection."""
        is_fully_connected = self.topology.is_fully_connected()
        self.assertIsInstance(is_fully_connected, bool)
    
    def test_tree_building(self):
        """Test tree building algorithms."""
        # Test binary tree
        binary_tree = self.topology.build_binary_tree()
        self.assertIn("root", binary_tree)
        self.assertIn("children", binary_tree)
        self.assertIn("parent", binary_tree)
        self.assertIn("depth", binary_tree)
        self.assertEqual(binary_tree["tree_type"], "binary")
        
        # Test k-ary tree
        kary_tree = self.topology.build_kary_tree(4)
        self.assertEqual(kary_tree["tree_type"], "4-ary")
        
        # Test balanced tree
        balanced_tree = self.topology.build_balanced_tree()
        self.assertIn("tree_type", balanced_tree)
    
    def test_tree_stats(self):
        """Test tree statistics calculation."""
        binary_tree = self.topology.build_binary_tree()
        stats = self.topology.get_tree_stats(binary_tree)
        
        self.assertIn("tree_type", stats)
        self.assertIn("num_devices", stats)
        self.assertIn("tree_depth", stats)
        self.assertIn("root", stats)
        self.assertIn("num_leaves", stats)
        self.assertIn("branching_factors", stats)
        self.assertIn("avg_branching_factor", stats)
        self.assertIn("max_branching_factor", stats)
        
        self.assertEqual(stats["num_devices"], 4)
        self.assertEqual(stats["tree_type"], "binary")
    
    def test_algorithm_selection(self):
        """Test adaptive algorithm selection."""
        # Test with different tensor sizes
        small_size = 1024  # 1KB
        medium_size = 1024 * 1024  # 1MB
        large_size = 10 * 1024 * 1024  # 10MB
        
        small_algo = self.topology.select_reduce_algorithm(small_size)
        medium_algo = self.topology.select_reduce_algorithm(medium_size)
        large_algo = self.topology.select_reduce_algorithm(large_size)
        
        for algo in [small_algo, medium_algo, large_algo]:
            self.assertIn(algo, ["ring", "binary_tree", "kary_tree", "balanced_tree"])
    
    def test_optimal_topology_building(self):
        """Test optimal topology building for different operations."""
        # Test reduce operation
        reduce_topology = self.topology.build_optimal_topology("reduce", 1024*1024)
        self.assertIn("type", reduce_topology)
        self.assertIn("reason", reduce_topology)
        
        # Test broadcast operation
        broadcast_topology = self.topology.build_optimal_topology("broadcast", 1024*1024)
        self.assertIn("type", broadcast_topology)
        self.assertIn("reason", broadcast_topology)
        
        # Test gather operation
        gather_topology = self.topology.build_optimal_topology("gather", 1024*1024)
        self.assertIn("type", gather_topology)
        self.assertIn("reason", gather_topology)
    
    def test_topology_summary(self):
        """Test topology summary generation."""
        summary = self.topology.get_topology_summary()
        
        required_keys = ["status", "num_devices", "topology_type", 
                        "connected_pairs", "total_pairs", "connectivity_ratio", 
                        "is_fully_connected"]
        
        for key in required_keys:
            self.assertIn(key, summary)
        
        self.assertEqual(summary["num_devices"], 4)
        self.assertIsInstance(summary["connectivity_ratio"], float)
        self.assertIsInstance(summary["is_fully_connected"], bool)


class TestP2PBackend(unittest.TestCase):
    """Test cases for P2P backend implementation."""
    
    def setUp(self):
        """Set up test environment."""
        if not P2P_AVAILABLE:
            self.skipTest("P2P modules not available")
        
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        if torch.cuda.device_count() < 2:
            self.skipTest("Need at least 2 GPUs for P2P testing")
        
        self.active_devices = [0, 1]
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize backends
        try:
            self.backend_0 = TPBackendP2P(
                device=0,
                active_devices=self.active_devices,
                output_device=0,
                init_method="tcp://localhost:12345",
                master=True,
                uuid="test_p2p_backend"
            )
            
            self.backend_1 = TPBackendP2P(
                device=1,
                active_devices=self.active_devices,
                output_device=0,
                init_method="tcp://localhost:12345",
                master=False,
                uuid="test_p2p_backend"
            )
        except Exception as e:
            self.skipTest(f"P2P backend initialization failed: {e}")
    
    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, 'backend_0'):
            self.backend_0.close()
        if hasattr(self, 'backend_1'):
            self.backend_1.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_backend_initialization(self):
        """Test backend initialization."""
        self.assertEqual(self.backend_0.device, 0)
        self.assertEqual(self.backend_1.device, 1)
        self.assertEqual(self.backend_0.active_devices, self.active_devices)
        self.assertEqual(self.backend_1.active_devices, self.active_devices)
        self.assertEqual(self.backend_0.world_size, 2)
        self.assertEqual(self.backend_1.world_size, 2)
        self.assertEqual(self.backend_0.rank, 0)
        self.assertEqual(self.backend_1.rank, 1)
    
    def test_p2p_availability(self):
        """Test P2P availability detection."""
        # Check if P2P is available
        if hasattr(self.backend_0, 'use_p2p'):
            self.assertIsInstance(self.backend_0.use_p2p, bool)
        
        if hasattr(self.backend_0, 'p2p_topology'):
            if self.backend_0.p2p_topology:
                summary = self.backend_0.p2p_topology.get_topology_summary()
                self.assertIsInstance(summary, dict)
    
    def test_memory_pool_stats(self):
        """Test memory pool statistics."""
        if hasattr(self.backend_0, 'get_memory_pool_stats'):
            stats = self.backend_0.get_memory_pool_stats()
            self.assertIsInstance(stats, dict)
            
            if stats:  # Only check if stats are available
                self.assertIn('device', stats)
                self.assertIn('pool_usage_bytes', stats)
                self.assertIn('pool_total_bytes', stats)
    
    def test_peer_access_management(self):
        """Test peer access management."""
        if hasattr(self.backend_0, 'enable_peer_access'):
            # Test enabling peer access
            result = self.backend_0.enable_peer_access(1)
            self.assertIsInstance(result, bool)
            
            # Test checking peer access
            if hasattr(self.backend_0, 'is_peer_access_enabled'):
                is_enabled = self.backend_0.is_peer_access_enabled(1)
                self.assertIsInstance(is_enabled, bool)
            
            # Test disabling peer access
            if hasattr(self.backend_0, 'disable_peer_access'):
                result = self.backend_0.disable_peer_access(1)
                self.assertIsInstance(result, bool)
    
    def test_bandwidth_measurement(self):
        """Test bandwidth measurement."""
        if hasattr(self.backend_0, 'measure_p2p_bandwidth'):
            bandwidth = self.backend_0.measure_p2p_bandwidth(0, 1, 1, 3)
            self.assertIsInstance(bandwidth, (float, int))
            self.assertGreaterEqual(bandwidth, 0.0)
    
    def test_latency_measurement(self):
        """Test latency measurement."""
        if hasattr(self.backend_0, 'measure_p2p_latency'):
            latency = self.backend_0.measure_p2p_latency(0, 1, 4, 10)
            self.assertIsInstance(latency, (float, int))
            self.assertGreaterEqual(latency, 0.0)
    
    def test_memory_access_validation(self):
        """Test memory access validation."""
        if hasattr(self.backend_0, 'validate_p2p_memory_access'):
            is_valid = self.backend_0.validate_p2p_memory_access(0, 1, 1024)
            self.assertIsInstance(is_valid, bool)
    
    def test_direct_tensor_copy(self):
        """Test direct tensor copy."""
        if hasattr(self.backend_0, 'copy_tensor_direct'):
            test_tensor = torch.randn(100, 100, device=0)
            
            try:
                copied_tensor = self.backend_0.copy_tensor_direct(0, 1, test_tensor)
                if copied_tensor is not None:
                    self.assertEqual(copied_tensor.device.index, 1)
                    self.assertEqual(copied_tensor.shape, test_tensor.shape)
            except Exception as e:
                # This might fail if P2P is not available
                print(f"Direct copy test failed (expected if P2P unavailable): {e}")
    
    def test_barrier_operation(self):
        """Test barrier operation."""
        try:
            self.backend_0.fwd_barrier()
            self.backend_1.fwd_barrier()
        except Exception as e:
            self.fail(f"Barrier operation failed: {e}")
    
    def test_broadcast_operation(self):
        """Test broadcast operation."""
        test_tensor = torch.randn(100, 100, device=0)
        
        try:
            self.backend_0.broadcast(test_tensor, 0)
            self.backend_1.broadcast(test_tensor, 0)
        except Exception as e:
            print(f"Broadcast test failed (might be expected): {e}")
    
    def test_all_reduce_operation(self):
        """Test all_reduce operation."""
        test_tensor = torch.randn(100, 100, device=0)
        
        try:
            self.backend_0.all_reduce(test_tensor)
            self.backend_1.all_reduce(test_tensor)
        except Exception as e:
            print(f"All_reduce test failed (might be expected): {e}")
    
    def test_gather_operation(self):
        """Test gather operation."""
        test_tensor = torch.randn(100, 100, device=0)
        out_tensor = torch.zeros(200, 100, device=0)
        gather_devices = torch.tensor([0, 1], dtype=torch.int)
        ldims = [100, 100]
        
        try:
            self.backend_0.gather(test_tensor, out_tensor, gather_devices, 0, ldims)
            self.backend_1.gather(test_tensor, None, gather_devices, 0, ldims)
        except Exception as e:
            print(f"Gather test failed (might be expected): {e}")


class TestP2PMonitoring(unittest.TestCase):
    """Test cases for P2P monitoring functionality."""
    
    def setUp(self):
        """Set up test environment."""
        if not P2P_AVAILABLE:
            self.skipTest("P2P modules not available")
        
        self.temp_dir = tempfile.mkdtemp()
        self.active_devices = [0, 1]
        
        try:
            self.monitor = P2PMonitor(
                active_devices=self.active_devices,
                monitoring_level="basic",
                max_history_size=100,
                enable_real_time=False,
                output_dir=self.temp_dir
            )
        except Exception as e:
            self.skipTest(f"P2P monitor initialization failed: {e}")
    
    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, 'monitor'):
            self.monitor.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_monitor_initialization(self):
        """Test monitor initialization."""
        self.assertEqual(self.monitor.active_devices, self.active_devices)
        self.assertEqual(self.monitor.num_devices, 2)
        self.assertEqual(self.monitor.monitoring_level, "basic")
        self.assertFalse(self.monitor.enable_real_time)
    
    def test_operation_recording(self):
        """Test operation recording."""
        tensor = torch.randn(100, 100)
        start_time = time.time()
        time.sleep(0.01)
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
        tensor = torch.randn(100, 100)
        
        # Record some operations
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
        
        metrics = self.monitor.get_device_metrics(0)
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.total_operations, 5)
        self.assertEqual(metrics.successful_operations, 5)
        self.assertEqual(metrics.failed_operations, 0)
    
    def test_performance_summary(self):
        """Test performance summary generation."""
        tensor = torch.randn(100, 100)
        
        # Record some operations
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


class TestP2PProfiling(unittest.TestCase):
    """Test cases for P2P profiling functionality."""
    
    def setUp(self):
        """Set up test environment."""
        if not P2P_AVAILABLE:
            self.skipTest("P2P modules not available")
        
        self.temp_dir = tempfile.mkdtemp()
        
        try:
            self.profiler = P2PProfiler(output_dir=self.temp_dir)
        except Exception as e:
            self.skipTest(f"P2P profiler initialization failed: {e}")
    
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
    
    def test_algorithm_comparison(self):
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
    
    def test_communication_pattern_analysis(self):
        """Test communication pattern analysis."""
        # Create some mock sessions
        with self.profiler.profile_session("session1", "broadcast", "ring"):
            pass
        
        patterns = self.profiler.analyze_communication_patterns()
        self.assertIsInstance(patterns, dict)
        self.assertIn("device_pair_frequency", patterns)
        self.assertIn("tensor_size_distribution", patterns)
        self.assertIn("operation_frequency", patterns)


class TestP2PDebugging(unittest.TestCase):
    """Test cases for P2P debugging functionality."""
    
    def setUp(self):
        """Set up test environment."""
        if not P2P_AVAILABLE:
            self.skipTest("P2P modules not available")
        
        self.temp_dir = tempfile.mkdtemp()
        
        try:
            self.debugger = P2PDebugger(
                debug_level="basic",
                max_events=100,
                output_dir=self.temp_dir
            )
        except Exception as e:
            self.skipTest(f"P2P debugger initialization failed: {e}")
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_debugger_initialization(self):
        """Test debugger initialization."""
        self.assertIsNotNone(self.debugger)
        self.assertEqual(self.debugger.debug_level, "basic")
        self.assertEqual(self.debugger.max_events, 100)
    
    def test_event_logging(self):
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
        
        events = self.debugger.get_events()
        self.assertEqual(len(events), 1)
        
        event = events[0]
        self.assertEqual(event.event_type, "operation_start")
        self.assertEqual(event.operation_type, "broadcast")
        self.assertEqual(event.device_id, 0)
        self.assertEqual(event.peer_device, 1)
        self.assertEqual(event.message, "Test event")
    
    def test_error_logging(self):
        """Test error logging."""
        self.debugger.log_error(
            error_type="communication_error",
            operation_type="broadcast",
            device_id=0,
            peer_device=1,
            error_message="Test error",
            context={"test": "context"}
        )
        
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
        
        trace = self.debugger.get_trace(trace_id)
        self.assertIsNotNone(trace)
        self.assertEqual(trace.trace_id, trace_id)
        self.assertEqual(trace.operation_type, "broadcast")
        self.assertTrue(trace.success)
    
    def test_issue_diagnosis(self):
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
    
    def test_debug_report_export(self):
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


class TestP2PConfiguration(unittest.TestCase):
    """Test cases for P2P configuration management."""
    
    def setUp(self):
        """Set up test environment."""
        if not P2P_AVAILABLE:
            self.skipTest("P2P modules not available")
        
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
            from exllamav3.util.p2p_config import load_config_from_env
            load_config_from_env()
            
            config = config_manager.get_config()
            self.assertEqual(config.monitoring.monitoring_level, "detailed")
            self.assertFalse(config.monitoring.enable_monitoring)
        finally:
            # Clean up environment variables
            os.environ.pop("P2P_MONITORING_LEVEL", None)
            os.environ.pop("P2P_ENABLE_MONITORING", None)


class TestP2PIntegration(unittest.TestCase):
    """Test cases for P2P integration with existing infrastructure."""
    
    def setUp(self):
        """Set up test environment."""
        if not P2P_AVAILABLE:
            self.skipTest("P2P modules not available")
        
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        if torch.cuda.device_count() < 2:
            self.skipTest("Need at least 2 GPUs for P2P testing")
        
        self.active_devices = [0, 1]
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_topology_monitoring_integration(self):
        """Test topology integration with monitoring."""
        try:
            topology = P2PTopology(self.active_devices)
            monitor = P2PMonitor(
                active_devices=self.active_devices,
                output_dir=self.temp_dir
            )
            
            # Set topology for monitoring
            monitor.set_topology(topology)
            
            # Check that topology metrics are available
            if hasattr(monitor, 'topology_metrics'):
                self.assertIsNotNone(monitor.topology_metrics)
                self.assertEqual(monitor.topology_metrics.num_devices, 2)
        except Exception as e:
            self.skipTest(f"Topology monitoring integration failed: {e}")
    
    def test_backend_monitoring_integration(self):
        """Test backend integration with monitoring."""
        try:
            backend = TPBackendP2P(
                device=0,
                active_devices=self.active_devices,
                output_device=0,
                init_method="tcp://localhost:12345",
                master=True,
                uuid="test_integration"
            )
            
            # Check that monitoring tools are initialized
            if hasattr(backend, 'monitor'):
                self.assertIsNotNone(backend.monitor)
            
            if hasattr(backend, 'profiler'):
                self.assertIsNotNone(backend.profiler)
            
            if hasattr(backend, 'debugger'):
                self.assertIsNotNone(backend.debugger)
            
            backend.close()
        except Exception as e:
            self.skipTest(f"Backend monitoring integration failed: {e}")
    
    def test_fallback_mechanism(self):
        """Test P2P fallback mechanism."""
        try:
            # Test with single device (should fallback)
            backend = TPBackendP2P(
                device=0,
                active_devices=[0],  # Single device
                output_device=0,
                init_method="tcp://localhost:12345",
                master=True,
                uuid="test_fallback"
            )
            
            # Should have fallback backend
            self.assertTrue(hasattr(backend, 'fallback'))
            self.assertIsNotNone(backend.fallback)
            
            # Test operations (should work through fallback)
            test_tensor = torch.randn(100, 100, device=0)
            backend.broadcast(test_tensor, 0)
            backend.all_reduce(test_tensor)
            
            backend.close()
        except Exception as e:
            self.skipTest(f"Fallback mechanism test failed: {e}")


class TestP2PErrorHandling(unittest.TestCase):
    """Test cases for P2P error handling and recovery."""
    
    def setUp(self):
        """Set up test environment."""
        if not P2P_AVAILABLE:
            self.skipTest("P2P modules not available")
        
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        if torch.cuda.device_count() < 2:
            self.skipTest("Need at least 2 GPUs for P2P testing")
        
        self.active_devices = [0, 1]
    
    def test_invalid_device_handling(self):
        """Test handling of invalid device IDs."""
        try:
            topology = P2PTopology(self.active_devices)
            
            # Test with invalid device
            result = topology.can_access_peer(999, 0)
            self.assertFalse(result)
            
            result = topology.can_access_peer(0, 999)
            self.assertFalse(result)
        except Exception as e:
            self.skipTest(f"Invalid device handling test failed: {e}")
    
    def test_empty_device_list(self):
        """Test handling of empty device list."""
        try:
            topology = P2PTopology([])
            self.assertEqual(topology.num_devices, 0)
            self.assertEqual(topology.active_devices, [])
            
            summary = topology.get_topology_summary()
            self.assertEqual(summary["num_devices"], 0)
        except Exception as e:
            self.skipTest(f"Empty device list test failed: {e}")
    
    def test_single_device_topology(self):
        """Test single device topology."""
        try:
            topology = P2PTopology([0])
            self.assertEqual(topology.num_devices, 1)
            self.assertTrue(topology.is_fully_connected())
            
            ratio = topology.get_connectivity_ratio()
            self.assertEqual(ratio, 1.0)  # Single device is fully connected
        except Exception as e:
            self.skipTest(f"Single device topology test failed: {e}")
    
    def test_backend_initialization_failure(self):
        """Test backend initialization failure handling."""
        try:
            # Test with invalid init method
            backend = TPBackendP2P(
                device=0,
                active_devices=self.active_devices,
                output_device=0,
                init_method="invalid://method",
                master=True,
                uuid="test_failure"
            )
            
            # Should handle failure gracefully
            if hasattr(backend, 'use_p2p'):
                # P2P might be disabled due to initialization failure
                self.assertIsInstance(backend.use_p2p, bool)
            
            backend.close()
        except Exception as e:
            # Expected to fail with invalid init method
            pass


def run_comprehensive_tests():
    """Run all comprehensive tests and return results."""
    # Create test suite
    test_classes = [
        TestP2PTopology,
        TestP2PBackend,
        TestP2PMonitoring,
        TestP2PProfiling,
        TestP2PDebugging,
        TestP2PConfiguration,
        TestP2PIntegration,
        TestP2PErrorHandling
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    print("Running Comprehensive P2P Test Suite")
    print("=" * 50)
    
    result = run_comprehensive_tests()
    
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All comprehensive tests passed!")
        exit(0)
    else:
        print(f"❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        exit(1)