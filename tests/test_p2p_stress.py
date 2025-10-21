#!/usr/bin/env python3
"""
Stress testing suite for P2P GPU communication implementation.

This module provides comprehensive stress testing including:
- High-load scenarios
- Long-running stability tests
- Resource exhaustion scenarios
- Concurrent operation testing
- Error handling and recovery under stress
"""

import unittest
import torch
import numpy as np
import time
import tempfile
import shutil
import os
import sys
import threading
import queue
import random
import gc
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil

# Add the exllamav3 directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from exllamav3.model.model_tp_p2p import P2PTopology
    from exllamav3.model.model_tp_backend import TPBackendP2P, TPBackendNative
    from exllamav3.util.p2p_monitor import P2PMonitor
    from exllamav3.util.p2p_debug import P2PDebugger
    from exllamav3.ext import exllamav3_ext as ext
    P2P_AVAILABLE = True
except ImportError as e:
    print(f"P2P modules not available: {e}")
    P2P_AVAILABLE = False


class P2PStressTester:
    """Stress tester for P2P operations."""
    
    def __init__(self, active_devices: List[int], output_dir: str = "./p2p_stress"):
        """
        Initialize stress tester.
        
        Args:
            active_devices: List of active GPU device IDs
            output_dir: Directory to store stress test results
        """
        self.active_devices = active_devices
        self.num_devices = len(active_devices)
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.topology = None
        self.backends = {}
        self.monitor = None
        self.debugger = None
        
        # Stress test configuration
        self.max_concurrent_ops = 100
        self.stress_duration = 300  # 5 minutes
        self.memory_pressure_mb = 1024  # 1GB
        
        # Results tracking
        self.stress_results = []
        self.error_count = 0
        self.operation_count = 0
        
        if P2P_AVAILABLE:
            try:
                self.topology = P2PTopology(active_devices)
                self.monitor = P2PMonitor(active_devices=active_devices, output_dir=output_dir)
                self.debugger = P2PDebugger(output_dir=output_dir)
            except Exception as e:
                print(f"Failed to initialize P2P components: {e}")
    
    def initialize_backends(self):
        """Initialize P2P backends for testing."""
        if not P2P_AVAILABLE:
            return False
        
        try:
            for device in self.active_devices:
                backend = TPBackendP2P(
                    device=device,
                    active_devices=self.active_devices,
                    output_device=self.active_devices[0],
                    init_method="tcp://localhost:12345",
                    master=(device == self.active_devices[0]),
                    uuid="p2p_stress_test"
                )
                self.backends[device] = backend
            return True
        except Exception as e:
            print(f"Failed to initialize backends: {e}")
            return False
    
    def cleanup_backends(self):
        """Clean up P2P backends."""
        for device, backend in self.backends.items():
            try:
                backend.close()
            except Exception:
                pass
        self.backends.clear()
    
    def stress_test_concurrent_operations(self, num_threads: int = 10, 
                                       ops_per_thread: int = 100) -> Dict:
        """
        Stress test with concurrent operations.
        
        Args:
            num_threads: Number of concurrent threads
            ops_per_thread: Number of operations per thread
            
        Returns:
            Dictionary with stress test results
        """
        if not self.backends:
            return {"error": "Backends not initialized"}
        
        results = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "total_time": 0.0,
            "avg_time_per_op": 0.0,
            "errors": []
        }
        
        def worker_thread(thread_id: int, num_ops: int):
            """Worker thread for stress testing."""
            thread_results = {
                "thread_id": thread_id,
                "operations": 0,
                "successful": 0,
                "failed": 0,
                "errors": []
            }
            
            device = self.active_devices[thread_id % len(self.active_devices)]
            backend = self.backends[device]
            
            for i in range(num_ops):
                try:
                    # Random operation type
                    op_type = random.choice(["broadcast", "all_reduce", "gather"])
                    
                    # Random tensor size
                    tensor_size = random.choice([1024, 1024*1024, 4*1024*1024])
                    
                    # Create test tensor
                    tensor = torch.randn(tensor_size, dtype=torch.float32, device=device)
                    
                    start_time = time.time()
                    
                    # Execute operation
                    if op_type == "broadcast":
                        backend.broadcast(tensor, device)
                    elif op_type == "all_reduce":
                        test_tensor = tensor.clone()
                        backend.all_reduce(test_tensor)
                    elif op_type == "gather":
                        out_tensor = torch.zeros(tensor_size * self.num_devices, dtype=torch.float32, device=device)
                        gather_devices = torch.tensor(self.active_devices, dtype=torch.int)
                        ldims = [tensor_size] * self.num_devices
                        backend.gather(tensor, out_tensor, gather_devices, device, ldims)
                    
                    end_time = time.time()
                    
                    thread_results["operations"] += 1
                    thread_results["successful"] += 1
                    
                    # Record operation
                    if self.monitor:
                        self.monitor.record_operation(
                            operation_type=op_type,
                            device_id=device,
                            peer_device=None,
                            tensor=tensor,
                            start_time=start_time,
                            end_time=end_time,
                            algorithm="stress_test",
                            success=True
                        )
                    
                except Exception as e:
                    thread_results["failed"] += 1
                    thread_results["errors"].append(str(e))
                    
                    # Log error
                    if self.debugger:
                        self.debugger.log_error(
                            error_type="stress_test_error",
                            operation_type=op_type,
                            device_id=device,
                            error_message=str(e),
                            context={"thread_id": thread_id, "operation": i}
                        )
            
            return thread_results
        
        # Start stress test
        start_time = time.time()
        
        # Create and start threads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for i in range(num_threads):
                future = executor.submit(worker_thread, i, ops_per_thread)
                futures.append(future)
            
            # Wait for all threads to complete
            thread_results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    thread_results.append(result)
                except Exception as e:
                    results["errors"].append(f"Thread failed: {e}")
        
        end_time = time.time()
        
        # Aggregate results
        for thread_result in thread_results:
            results["total_operations"] += thread_result["operations"]
            results["successful_operations"] += thread_result["successful"]
            results["failed_operations"] += thread_result["failed"]
            results["errors"].extend(thread_result["errors"])
        
        results["total_time"] = end_time - start_time
        if results["total_operations"] > 0:
            results["avg_time_per_op"] = results["total_time"] / results["total_operations"]
        
        return results
    
    def stress_test_memory_pressure(self, duration_seconds: int = 60) -> Dict:
        """
        Stress test under memory pressure.
        
        Args:
            duration_seconds: Duration of stress test in seconds
            
        Returns:
            Dictionary with stress test results
        """
        if not self.backends:
            return {"error": "Backends not initialized"}
        
        results = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "memory_allocated_mb": 0,
            "peak_memory_mb": 0,
            "errors": []
        }
        
        device = self.active_devices[0]
        backend = self.backends[device]
        
        # Allocate memory to create pressure
        memory_tensors = []
        try:
            # Allocate memory tensors
            for i in range(10):
                tensor = torch.randn(256*1024*1024 // 4, dtype=torch.float32, device=device)  # 256MB each
                memory_tensors.append(tensor)
            
            start_time = time.time()
            peak_memory = 0
            
            while time.time() - start_time < duration_seconds:
                try:
                    # Get current memory usage
                    current_memory = torch.cuda.memory_allocated(device) / (1024**2)
                    peak_memory = max(peak_memory, current_memory)
                    
                    # Perform operation
                    tensor_size = random.choice([1024, 1024*1024])
                    test_tensor = torch.randn(tensor_size, dtype=torch.float32, device=device)
                    
                    op_start = time.time()
                    backend.all_reduce(test_tensor)
                    op_end = time.time()
                    
                    results["total_operations"] += 1
                    results["successful_operations"] += 1
                    
                    # Record operation
                    if self.monitor:
                        self.monitor.record_operation(
                            operation_type="all_reduce",
                            device_id=device,
                            peer_device=None,
                            tensor=test_tensor,
                            start_time=op_start,
                            end_time=op_end,
                            algorithm="memory_pressure",
                            success=True
                        )
                    
                    # Small delay to prevent overwhelming the system
                    time.sleep(0.01)
                    
                except Exception as e:
                    results["failed_operations"] += 1
                    results["errors"].append(str(e))
                    
                    if self.debugger:
                        self.debugger.log_error(
                            error_type="memory_pressure_error",
                            operation_type="all_reduce",
                            device_id=device,
                            error_message=str(e),
                            context={"memory_usage_mb": current_memory}
                        )
            
            results["memory_allocated_mb"] = torch.cuda.memory_allocated(device) / (1024**2)
            results["peak_memory_mb"] = peak_memory
            
        finally:
            # Clean up memory tensors
            for tensor in memory_tensors:
                del tensor
            torch.cuda.empty_cache()
        
        return results
    
    def stress_test_long_running(self, duration_seconds: int = 300) -> Dict:
        """
        Long-running stability test.
        
        Args:
            duration_seconds: Duration of test in seconds
            
        Returns:
            Dictionary with stress test results
        """
        if not self.backends:
            return {"error": "Backends not initialized"}
        
        results = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "total_time": 0.0,
            "avg_time_per_op": 0.0,
            "performance_degradation": 0.0,
            "errors": []
        }
        
        device = self.active_devices[0]
        backend = self.backends[device]
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        # Track performance over time
        performance_samples = []
        
        while time.time() < end_time:
            try:
                # Test operation
                tensor_size = random.choice([1024, 1024*1024, 4*1024*1024])
                test_tensor = torch.randn(tensor_size, dtype=torch.float32, device=device)
                
                op_start = time.time()
                backend.broadcast(test_tensor, device)
                op_end = time.time()
                
                op_time = op_end - op_start
                results["total_operations"] += 1
                results["successful_operations"] += 1
                
                # Record performance sample
                performance_samples.append({
                    "timestamp": time.time() - start_time,
                    "operation_time": op_time,
                    "tensor_size": tensor_size
                })
                
                # Record operation
                if self.monitor:
                    self.monitor.record_operation(
                        operation_type="broadcast",
                        device_id=device,
                        peer_device=None,
                        tensor=test_tensor,
                        start_time=op_start,
                        end_time=op_end,
                        algorithm="long_running",
                        success=True
                    )
                
                # Check for performance degradation
                if len(performance_samples) > 100:
                    recent_samples = performance_samples[-100:]
                    early_samples = performance_samples[:100]
                    
                    recent_avg = np.mean([s["operation_time"] for s in recent_samples])
                    early_avg = np.mean([s["operation_time"] for s in early_samples])
                    
                    degradation = (recent_avg - early_avg) / early_avg * 100
                    results["performance_degradation"] = max(results["performance_degradation"], degradation)
                
                # Small delay
                time.sleep(0.1)
                
            except Exception as e:
                results["failed_operations"] += 1
                results["errors"].append(str(e))
                
                if self.debugger:
                    self.debugger.log_error(
                        error_type="long_running_error",
                        operation_type="broadcast",
                        device_id=device,
                        error_message=str(e),
                        context={"operation_count": results["total_operations"]}
                    )
        
        results["total_time"] = time.time() - start_time
        if results["total_operations"] > 0:
            results["avg_time_per_op"] = results["total_time"] / results["total_operations"]
        
        return results
    
    def stress_test_resource_exhaustion(self) -> Dict:
        """
        Test behavior under resource exhaustion.
        
        Returns:
            Dictionary with stress test results
        """
        if not self.backends:
            return {"error": "Backends not initialized"}
        
        results = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "max_concurrent_ops": 0,
            "resource_errors": [],
            "recovery_time": 0.0
        }
        
        device = self.active_devices[0]
        backend = self.backends[device]
        
        # Test with increasing concurrent operations
        max_concurrent = 0
        recovery_start = None
        
        for concurrent_ops in range(10, 200, 10):
            try:
                # Create multiple operations concurrently
                tensors = []
                futures = []
                
                with ThreadPoolExecutor(max_workers=concurrent_ops) as executor:
                    for i in range(concurrent_ops):
                        tensor = torch.randn(1024*1024, dtype=torch.float32, device=device)
                        tensors.append(tensor)
                        
                        future = executor.submit(backend.broadcast, tensor, device)
                        futures.append(future)
                    
                    # Wait for operations to complete
                    for future in as_completed(futures):
                        try:
                            future.result(timeout=10)  # 10 second timeout
                            results["successful_operations"] += 1
                        except Exception as e:
                            results["failed_operations"] += 1
                            results["resource_errors"].append(str(e))
                            
                            if recovery_start is None:
                                recovery_start = time.time()
                
                max_concurrent = concurrent_ops
                results["total_operations"] += concurrent_ops
                
                # Clean up tensors
                for tensor in tensors:
                    del tensor
                torch.cuda.empty_cache()
                
                # Small delay
                time.sleep(0.1)
                
            except Exception as e:
                results["resource_errors"].append(f"Concurrent ops {concurrent_ops}: {e}")
                if recovery_start is None:
                    recovery_start = time.time()
                break
        
        results["max_concurrent_ops"] = max_concurrent
        
        # Test recovery
        if recovery_start:
            recovery_time = 0
            try:
                # Try to perform a simple operation
                test_tensor = torch.randn(1024, dtype=torch.float32, device=device)
                start_time = time.time()
                backend.broadcast(test_tensor, device)
                recovery_time = time.time() - start_time
                results["successful_operations"] += 1
            except Exception as e:
                results["resource_errors"].append(f"Recovery failed: {e}")
            
            results["recovery_time"] = recovery_time
        
        return results
    
    def stress_test_error_handling(self) -> Dict:
        """
        Test error handling under stress conditions.
        
        Returns:
            Dictionary with stress test results
        """
        if not self.backends:
            return {"error": "Backends not initialized"}
        
        results = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "error_types": {},
            "recovery_success_rate": 0.0,
            "errors": []
        }
        
        device = self.active_devices[0]
        backend = self.backends[device]
        
        # Test various error conditions
        error_conditions = [
            "invalid_device",
            "oversized_tensor",
            "null_tensor",
            "wrong_dtype",
            "non_contiguous"
        ]
        
        for condition in error_conditions:
            condition_results = {
                "condition": condition,
                "attempts": 0,
                "failures": 0,
                "recoveries": 0
            }
            
            for attempt in range(10):
                condition_results["attempts"] += 1
                results["total_operations"] += 1
                
                try:
                    if condition == "invalid_device":
                        # Try to use invalid device
                        invalid_tensor = torch.randn(100, dtype=torch.float32, device=device)
                        backend.broadcast(invalid_tensor, 999)  # Invalid device
                        
                    elif condition == "oversized_tensor":
                        # Try to use oversized tensor
                        oversized_tensor = torch.randn(100*1024*1024, dtype=torch.float32, device=device)
                        backend.broadcast(oversized_tensor, device)
                        
                    elif condition == "null_tensor":
                        # Try to use null tensor
                        backend.broadcast(None, device)
                        
                    elif condition == "wrong_dtype":
                        # Try to use wrong dtype
                        wrong_tensor = torch.randn(100, dtype=torch.int32, device=device)
                        backend.broadcast(wrong_tensor, device)
                        
                    elif condition == "non_contiguous":
                        # Try to use non-contiguous tensor
                        base_tensor = torch.randn(200, dtype=torch.float32, device=device)
                        non_contiguous = base_tensor[::2]  # Every other element
                        backend.broadcast(non_contiguous, device)
                    
                    # If we get here, the operation didn't fail as expected
                    results["successful_operations"] += 1
                    
                except Exception as e:
                    condition_results["failures"] += 1
                    results["failed_operations"] += 1
                    
                    error_type = type(e).__name__
                    if error_type not in results["error_types"]:
                        results["error_types"][error_type] = 0
                    results["error_types"][error_type] += 1
                    
                    results["errors"].append(f"{condition}: {e}")
                    
                    # Test recovery
                    try:
                        recovery_tensor = torch.randn(100, dtype=torch.float32, device=device)
                        backend.broadcast(recovery_tensor, device)
                        condition_results["recoveries"] += 1
                        results["successful_operations"] += 1
                    except Exception as recovery_error:
                        results["errors"].append(f"Recovery failed for {condition}: {recovery_error}")
            
            # Calculate recovery success rate
            if condition_results["failures"] > 0:
                recovery_rate = condition_results["recoveries"] / condition_results["failures"]
                results["recovery_success_rate"] += recovery_rate
        
        # Average recovery success rate
        if len(error_conditions) > 0:
            results["recovery_success_rate"] /= len(error_conditions)
        
        return results
    
    def generate_stress_report(self) -> str:
        """
        Generate a comprehensive stress test report.
        
        Returns:
            Path to generated report file
        """
        report_path = os.path.join(self.output_dir, "stress_test_report.json")
        
        report_data = {
            "timestamp": time.time(),
            "num_devices": self.num_devices,
            "active_devices": self.active_devices,
            "stress_results": self.stress_results,
            "summary": {
                "total_tests": len(self.stress_results),
                "total_operations": sum(r.get("total_operations", 0) for r in self.stress_results),
                "total_successful": sum(r.get("successful_operations", 0) for r in self.stress_results),
                "total_failed": sum(r.get("failed_operations", 0) for r in self.stress_results),
                "total_errors": sum(len(r.get("errors", [])) for r in self.stress_results)
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        return report_path


class TestP2PStress(unittest.TestCase):
    """Test cases for P2P stress testing."""
    
    def setUp(self):
        """Set up test environment."""
        if not P2P_AVAILABLE:
            self.skipTest("P2P modules not available")
        
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        if torch.cuda.device_count() < 2:
            self.skipTest("Need at least 2 GPUs for P2P stress testing")
        
        self.active_devices = list(range(min(4, torch.cuda.device_count())))
        self.temp_dir = tempfile.mkdtemp()
        
        self.stress_tester = P2PStressTester(
            active_devices=self.active_devices,
            output_dir=self.temp_dir
        )
        
        # Initialize backends
        if not self.stress_tester.initialize_backends():
            self.skipTest("Failed to initialize P2P backends")
    
    def tearDown(self):
        """Clean up test environment."""
        self.stress_tester.cleanup_backends()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_concurrent_operations_stress(self):
        """Test concurrent operations under stress."""
        # Reduced parameters for unit testing
        num_threads = 4
        ops_per_thread = 10
        
        result = self.stress_tester.stress_test_concurrent_operations(
            num_threads=num_threads,
            ops_per_thread=ops_per_thread
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn("total_operations", result)
        self.assertIn("successful_operations", result)
        self.assertIn("failed_operations", result)
        self.assertIn("total_time", result)
        self.assertIn("avg_time_per_op", result)
        self.assertIn("errors", result)
        
        # Check that operations were performed
        expected_ops = num_threads * ops_per_thread
        self.assertEqual(result["total_operations"], expected_ops)
        
        # Check success rate (should be reasonably high)
        if result["total_operations"] > 0:
            success_rate = result["successful_operations"] / result["total_operations"]
            self.assertGreaterEqual(success_rate, 0.5, "Success rate too low")
        
        # Store result for report
        self.stress_tester.stress_results.append({
            "test": "concurrent_operations",
            "result": result
        })
    
    def test_memory_pressure_stress(self):
        """Test operations under memory pressure."""
        # Reduced duration for unit testing
        duration_seconds = 10
        
        result = self.stress_tester.stress_test_memory_pressure(duration_seconds)
        
        self.assertIsInstance(result, dict)
        self.assertIn("total_operations", result)
        self.assertIn("successful_operations", result)
        self.assertIn("failed_operations", result)
        self.assertIn("memory_allocated_mb", result)
        self.assertIn("peak_memory_mb", result)
        self.assertIn("errors", result)
        
        # Check that operations were performed
        self.assertGreater(result["total_operations"], 0)
        
        # Check memory usage
        self.assertGreater(result["memory_allocated_mb"], 0)
        self.assertGreaterEqual(result["peak_memory_mb"], result["memory_allocated_mb"])
        
        # Store result for report
        self.stress_tester.stress_results.append({
            "test": "memory_pressure",
            "result": result
        })
    
    def test_long_running_stress(self):
        """Test long-running stability."""
        # Reduced duration for unit testing
        duration_seconds = 15
        
        result = self.stress_tester.stress_test_long_running(duration_seconds)
        
        self.assertIsInstance(result, dict)
        self.assertIn("total_operations", result)
        self.assertIn("successful_operations", result)
        self.assertIn("failed_operations", result)
        self.assertIn("total_time", result)
        self.assertIn("avg_time_per_op", result)
        self.assertIn("performance_degradation", result)
        self.assertIn("errors", result)
        
        # Check that operations were performed
        self.assertGreater(result["total_operations"], 0)
        
        # Check that test ran for expected duration
        self.assertGreaterEqual(result["total_time"], duration_seconds * 0.9)
        
        # Check performance degradation (should be reasonable)
        self.assertLessEqual(result["performance_degradation"], 50.0, "Performance degradation too high")
        
        # Store result for report
        self.stress_tester.stress_results.append({
            "test": "long_running",
            "result": result
        })
    
    def test_resource_exhaustion_stress(self):
        """Test behavior under resource exhaustion."""
        result = self.stress_tester.stress_test_resource_exhaustion()
        
        self.assertIsInstance(result, dict)
        self.assertIn("total_operations", result)
        self.assertIn("successful_operations", result)
        self.assertIn("failed_operations", result)
        self.assertIn("max_concurrent_ops", result)
        self.assertIn("resource_errors", result)
        self.assertIn("recovery_time", result)
        
        # Check that some operations were performed
        self.assertGreater(result["total_operations"], 0)
        
        # Check that we found a limit for concurrent operations
        self.assertGreater(result["max_concurrent_ops"], 0)
        
        # Store result for report
        self.stress_tester.stress_results.append({
            "test": "resource_exhaustion",
            "result": result
        })
    
    def test_error_handling_stress(self):
        """Test error handling under stress."""
        result = self.stress_tester.stress_test_error_handling()
        
        self.assertIsInstance(result, dict)
        self.assertIn("total_operations", result)
        self.assertIn("successful_operations", result)
        self.assertIn("failed_operations", result)
        self.assertIn("error_types", result)
        self.assertIn("recovery_success_rate", result)
        self.assertIn("errors", result)
        
        # Check that operations were performed
        self.assertGreater(result["total_operations"], 0)
        
        # Check that we encountered some errors (expected)
        self.assertGreater(result["failed_operations"], 0)
        
        # Check error types
        self.assertIsInstance(result["error_types"], dict)
        self.assertGreater(len(result["error_types"]), 0)
        
        # Check recovery success rate
        self.assertGreaterEqual(result["recovery_success_rate"], 0.0)
        self.assertLessEqual(result["recovery_success_rate"], 1.0)
        
        # Store result for report
        self.stress_tester.stress_results.append({
            "test": "error_handling",
            "result": result
        })
    
    def test_stress_report_generation(self):
        """Test stress report generation."""
        # Run some stress tests to generate results
        self.stress_tester.stress_test_concurrent_operations(2, 5)
        self.stress_tester.stress_test_memory_pressure(5)
        self.stress_tester.stress_test_long_running(5)
        
        # Generate report
        report_path = self.stress_tester.generate_stress_report()
        
        self.assertTrue(os.path.exists(report_path))
        
        # Check report content
        with open(report_path, 'r') as f:
            report_data = json.load(f)
        
        self.assertIn("timestamp", report_data)
        self.assertIn("num_devices", report_data)
        self.assertIn("active_devices", report_data)
        self.assertIn("stress_results", report_data)
        self.assertIn("summary", report_data)
        
        # Check that we have results
        self.assertGreater(len(report_data["stress_results"]), 0)
        
        # Check summary
        summary = report_data["summary"]
        self.assertIn("total_tests", summary)
        self.assertIn("total_operations", summary)
        self.assertIn("total_successful", summary)
        self.assertIn("total_failed", summary)
        self.assertIn("total_errors", summary)
        
        self.assertGreater(summary["total_tests"], 0)
        self.assertGreater(summary["total_operations"], 0)
    
    def test_system_resource_monitoring(self):
        """Test system resource monitoring during stress."""
        # Get initial system stats
        initial_cpu = psutil.cpu_percent()
        initial_memory = psutil.virtual_memory().percent
        
        # Run stress test
        result = self.stress_tester.stress_test_concurrent_operations(4, 10)
        
        # Get final system stats
        final_cpu = psutil.cpu_percent()
        final_memory = psutil.virtual_memory().percent
        
        # Check that system resources were used
        self.assertGreaterEqual(final_cpu, initial_cpu)
        self.assertGreaterEqual(final_memory, initial_memory)
        
        # Check that stress test completed
        self.assertGreater(result["total_operations"], 0)


def run_stress_tests():
    """Run all stress tests and return results."""
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestP2PStress)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    print("Running P2P Stress Tests")
    print("=" * 50)
    
    result = run_stress_tests()
    
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All stress tests passed!")
        exit(0)
    else:
        print(f"❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        exit(1)