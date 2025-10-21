"""
P2P Monitoring Validation Tool

This script validates the accuracy and performance impact of the P2P monitoring tools.
"""

import os
import sys
import time
import json
import argparse
import tempfile
import shutil
from typing import Dict, List, Any, Optional
import numpy as np
import torch

# Add the exllamav3 directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from exllamav3.util.p2p_monitor import P2PMonitor, initialize_global_monitor, get_global_monitor
    from exllamav3.util.p2p_profiler import P2PProfiler, initialize_global_profiler, get_global_profiler
    from exllamav3.util.p2p_debug import P2PDebugger, initialize_global_debugger, get_global_debugger
    from exllamav3.util.p2p_config import P2PConfigManager, get_global_config
    from exllamav3.model.model_tp_p2p import P2PTopology
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    print("Warning: P2P monitoring tools not available")


class P2PMonitoringValidator:
    """Validator for P2P monitoring tools."""
    
    def __init__(self, output_dir: str = "./monitoring_validation"):
        """
        Initialize validator.
        
        Args:
            output_dir: Directory to save validation results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Validation results
        self.results = {
            "accuracy_tests": {},
            "performance_impact": {},
            "overhead_analysis": {},
            "reliability_tests": {},
            "scalability_tests": {},
            "summary": {}
        }
        
        # Test devices (mock for validation)
        self.active_devices = [0, 1] if torch.cuda.is_available() else []
        
        if not self.active_devices:
            print("Warning: No CUDA devices available. Using mock devices for validation.")
            self.active_devices = [0, 1]  # Mock devices
    
    def run_all_validations(self) -> Dict[str, Any]:
        """
        Run all validation tests.
        
        Returns:
            Dictionary with validation results
        """
        print("Starting P2P monitoring validation...")
        print(f"Output directory: {self.output_dir}")
        
        if not MONITORING_AVAILABLE:
            print("Error: P2P monitoring tools not available")
            return self.results
        
        # Run validation tests
        self._test_monitoring_accuracy()
        self._test_performance_impact()
        self._test_overhead_analysis()
        self._test_reliability()
        self._test_scalability()
        
        # Generate summary
        self._generate_summary()
        
        # Save results
        self._save_results()
        
        print("Validation complete!")
        return self.results
    
    def _test_monitoring_accuracy(self):
        """Test monitoring accuracy."""
        print("\nTesting monitoring accuracy...")
        
        # Initialize monitoring tools
        monitor = P2PMonitor(
            active_devices=self.active_devices,
            monitoring_level="comprehensive",
            enable_real_time=False,
            output_dir=os.path.join(self.output_dir, "accuracy_test")
        )
        
        profiler = P2PProfiler(
            monitor=monitor,
            output_dir=os.path.join(self.output_dir, "accuracy_test")
        )
        
        debugger = P2PDebugger(
            monitor=monitor,
            debug_level="detailed",
            output_dir=os.path.join(self.output_dir, "accuracy_test")
        )
        
        # Create mock topology
        topology = P2PTopology(self.active_devices)
        monitor.set_topology(topology)
        
        accuracy_results = {
            "operation_tracking": {},
            "metrics_calculation": {},
            "topology_analysis": {},
            "error_detection": {}
        }
        
        try:
            # Test operation tracking accuracy
            print("  Testing operation tracking...")
            tensor_sizes = [100, 1000, 10000]
            operations = ["broadcast", "all_reduce", "gather", "direct_copy"]
            
            for size in tensor_sizes:
                for op in operations:
                    tensor = torch.randn(size, size)
                    
                    # Record operation with known parameters
                    start_time = time.time()
                    time.sleep(0.001)  # Small delay
                    end_time = time.time()
                    
                    monitor.record_operation(
                        operation_type=op,
                        device_id=self.active_devices[0],
                        peer_device=self.active_devices[1] if len(self.active_devices) > 1 else None,
                        tensor=tensor,
                        start_time=start_time,
                        end_time=end_time,
                        algorithm="test_algorithm",
                        success=True
                    )
                    
                    # Verify tracking
                    recorded_ops = monitor.get_operation_history(operation_type=op, limit=1)
                    if recorded_ops:
                        recorded_op = recorded_ops[0]
                        
                        # Check accuracy
                        duration_diff = abs(recorded_op.duration_ms - (end_time - start_time) * 1000)
                        size_match = recorded_op.tensor_size_bytes == tensor.numel() * tensor.element_size()
                        
                        accuracy_results["operation_tracking"][f"{op}_{size}"] = {
                            "duration_error_ms": duration_diff,
                            "size_match": size_match,
                            "algorithm_match": recorded_op.algorithm == "test_algorithm",
                            "success_match": recorded_op.success
                        }
            
            # Test metrics calculation accuracy
            print("  Testing metrics calculation...")
            
            # Record multiple operations for metrics calculation
            for i in range(10):
                tensor = torch.randn(1000, 1000)
                start_time = time.time()
                time.sleep(0.001)
                end_time = time.time()
                
                monitor.record_operation(
                    operation_type="all_reduce",
                    device_id=self.active_devices[0],
                    tensor=tensor,
                    start_time=start_time,
                    end_time=end_time,
                    algorithm="test",
                    success=True
                )
            
            # Get device metrics
            device_metrics = monitor.get_device_metrics(self.active_devices[0])
            if device_metrics:
                # Calculate expected values
                expected_ops = 10
                expected_success_rate = 1.0
                
                accuracy_results["metrics_calculation"] = {
                    "operation_count_match": device_metrics.total_operations == expected_ops,
                    "success_rate_match": abs(device_metrics.successful_operations / device_metrics.total_operations - expected_success_rate) < 0.01,
                    "bandwidth_calculated": device_metrics.average_bandwidth_gbps > 0
                }
            
            # Test topology analysis accuracy
            print("  Testing topology analysis...")
            topology_summary = topology.get_topology_summary()
            topology_metrics = monitor.topology_metrics
            
            if topology_metrics:
                accuracy_results["topology_analysis"] = {
                    "device_count_match": topology_metrics.num_devices == topology_summary.get("num_devices", 0),
                    "connectivity_match": abs(topology_metrics.connectivity_ratio - topology_summary.get("connectivity_ratio", 0)) < 0.01,
                    "topology_type_match": topology_metrics.topology_type == topology_summary.get("topology_type", "")
                }
            
            # Test error detection accuracy
            print("  Testing error detection...")
            
            # Log a test error
            debugger.log_error(
                error_type="test_error",
                operation_type="test_operation",
                device_id=self.active_devices[0],
                error_message="Test error message",
                context={"test": "context"}
            )
            
            # Check error detection
            errors = debugger.get_errors(error_type="test_error", limit=1)
            accuracy_results["error_detection"] = {
                "error_detected": len(errors) > 0,
                "error_type_match": errors[0].error_type == "test_error" if errors else False,
                "error_message_match": "Test error message" in errors[0].error_message if errors else False
            }
            
        except Exception as e:
            print(f"  Error in accuracy testing: {e}")
            accuracy_results["error"] = str(e)
        
        finally:
            # Cleanup
            monitor.close()
        
        self.results["accuracy_tests"] = accuracy_results
        print("  Monitoring accuracy testing complete.")
    
    def _test_performance_impact(self):
        """Test performance impact of monitoring."""
        print("\nTesting performance impact...")
        
        impact_results = {
            "monitoring_overhead": {},
            "profiling_overhead": {},
            "debugging_overhead": {},
            "memory_usage": {}
        }
        
        try:
            # Test monitoring overhead
            print("  Testing monitoring overhead...")
            
            # Baseline test (no monitoring)
            baseline_times = []
            for _ in range(10):
                tensor = torch.randn(1000, 1000)
                start_time = time.time()
                
                # Simulate P2P operation
                time.sleep(0.001)
                
                end_time = time.time()
                baseline_times.append(end_time - start_time)
            
            baseline_avg = np.mean(baseline_times)
            
            # Test with monitoring
            monitor = P2PMonitor(
                active_devices=self.active_devices,
                monitoring_level="comprehensive",
                enable_real_time=False,
                output_dir=os.path.join(self.output_dir, "impact_test")
            )
            
            monitored_times = []
            for i in range(10):
                tensor = torch.randn(1000, 1000)
                start_time = time.time()
                
                # Record operation
                monitor.record_operation(
                    operation_type="test",
                    device_id=self.active_devices[0],
                    tensor=tensor,
                    start_time=start_time,
                    end_time=time.time(),
                    success=True
                )
                
                # Simulate P2P operation
                time.sleep(0.001)
                
                end_time = time.time()
                monitored_times.append(end_time - start_time)
            
            monitored_avg = np.mean(monitored_times)
            overhead_percent = ((monitored_avg - baseline_avg) / baseline_avg) * 100
            
            impact_results["monitoring_overhead"] = {
                "baseline_time_ms": baseline_avg * 1000,
                "monitored_time_ms": monitored_avg * 1000,
                "overhead_percent": overhead_percent,
                "acceptable_overhead": overhead_percent < 5.0  # Less than 5% overhead is acceptable
            }
            
            # Test profiling overhead
            print("  Testing profiling overhead...")
            profiler = P2PProfiler(
                monitor=monitor,
                output_dir=os.path.join(self.output_dir, "impact_test")
            )
            
            # Profile operation
            def mock_operation(tensor):
                time.sleep(0.001)
                return tensor
            
            tensor = torch.randn(1000, 1000)
            start_time = time.time()
            
            result = profiler.profile_operation(
                operation_func=mock_operation,
                operation_type="test",
                num_iterations=5,
                warmup_iterations=2,
                tensor=tensor,
                device_id=self.active_devices[0]
            )
            
            end_time = time.time()
            profiling_time = end_time - start_time
            
            impact_results["profiling_overhead"] = {
                "profiling_time_ms": profiling_time * 1000,
                "iterations": len(result.durations_ms),
                "avg_duration_ms": np.mean(result.durations_ms) if result.durations_ms else 0
            }
            
            # Test debugging overhead
            print("  Testing debugging overhead...")
            debugger = P2PDebugger(
                monitor=monitor,
                debug_level="detailed",
                output_dir=os.path.join(self.output_dir, "impact_test")
            )
            
            # Log debug events
            start_time = time.time()
            for i in range(10):
                debugger.log_event(
                    event_type="test",
                    operation_type="test",
                    device_id=self.active_devices[0],
                    message="Test event"
                )
            end_time = time.time()
            
            debugging_time = end_time - start_time
            
            impact_results["debugging_overhead"] = {
                "debugging_time_ms": debugging_time * 1000,
                "events_logged": 10,
                "avg_time_per_event_ms": (debugging_time * 1000) / 10
            }
            
            # Test memory usage
            print("  Testing memory usage...")
            
            # Get memory before monitoring
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                memory_before = torch.cuda.memory_allocated(self.active_devices[0])
            else:
                memory_before = 0
            
            # Use monitoring tools
            for i in range(100):
                tensor = torch.randn(100, 100)
                monitor.record_operation(
                    operation_type="test",
                    device_id=self.active_devices[0],
                    tensor=tensor,
                    start_time=time.time(),
                    end_time=time.time(),
                    success=True
                )
            
            # Get memory after monitoring
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated(self.active_devices[0])
            else:
                memory_after = 0
            
            impact_results["memory_usage"] = {
                "memory_before_bytes": memory_before,
                "memory_after_bytes": memory_after,
                "memory_increase_bytes": memory_after - memory_before,
                "memory_per_operation_bytes": (memory_after - memory_before) / 100 if memory_after > memory_before else 0
            }
            
        except Exception as e:
            print(f"  Error in performance impact testing: {e}")
            impact_results["error"] = str(e)
        
        finally:
            # Cleanup
            if 'monitor' in locals():
                monitor.close()
        
        self.results["performance_impact"] = impact_results
        print("  Performance impact testing complete.")
    
    def _test_overhead_analysis(self):
        """Test overhead analysis of monitoring tools."""
        print("\nTesting overhead analysis...")
        
        overhead_results = {
            "monitoring_levels": {},
            "real_time_monitoring": {},
            "history_size_impact": {}
        }
        
        try:
            # Test different monitoring levels
            print("  Testing monitoring levels...")
            levels = ["basic", "detailed", "comprehensive"]
            
            for level in levels:
                monitor = P2PMonitor(
                    active_devices=self.active_devices,
                    monitoring_level=level,
                    enable_real_time=False,
                    output_dir=os.path.join(self.output_dir, f"overhead_test_{level}")
                )
                
                # Record operations
                start_time = time.time()
                for i in range(50):
                    tensor = torch.randn(100, 100)
                    monitor.record_operation(
                        operation_type="test",
                        device_id=self.active_devices[0],
                        tensor=tensor,
                        start_time=time.time(),
                        end_time=time.time(),
                        success=True
                    )
                end_time = time.time()
                
                overhead_results["monitoring_levels"][level] = {
                    "operations_recorded": 50,
                    "total_time_ms": (end_time - start_time) * 1000,
                    "avg_time_per_operation_ms": ((end_time - start_time) * 1000) / 50
                }
                
                monitor.close()
            
            # Test real-time monitoring overhead
            print("  Testing real-time monitoring overhead...")
            
            # Test without real-time monitoring
            monitor_no_rt = P2PMonitor(
                active_devices=self.active_devices,
                monitoring_level="basic",
                enable_real_time=False,
                output_dir=os.path.join(self.output_dir, "overhead_test_no_rt")
            )
            
            start_time = time.time()
            for i in range(20):
                tensor = torch.randn(100, 100)
                monitor_no_rt.record_operation(
                    operation_type="test",
                    device_id=self.active_devices[0],
                    tensor=tensor,
                    start_time=time.time(),
                    end_time=time.time(),
                    success=True
                )
            end_time = time.time()
            
            no_rt_time = (end_time - start_time) * 1000
            monitor_no_rt.close()
            
            # Test with real-time monitoring
            monitor_rt = P2PMonitor(
                active_devices=self.active_devices,
                monitoring_level="basic",
                enable_real_time=True,
                output_dir=os.path.join(self.output_dir, "overhead_test_rt")
            )
            
            start_time = time.time()
            for i in range(20):
                tensor = torch.randn(100, 100)
                monitor_rt.record_operation(
                    operation_type="test",
                    device_id=self.active_devices[0],
                    tensor=tensor,
                    start_time=time.time(),
                    end_time=time.time(),
                    success=True
                )
            end_time = time.time()
            
            rt_time = (end_time - start_time) * 1000
            monitor_rt.close()
            
            overhead_results["real_time_monitoring"] = {
                "no_real_time_ms": no_rt_time,
                "with_real_time_ms": rt_time,
                "overhead_ms": rt_time - no_rt_time,
                "overhead_percent": ((rt_time - no_rt_time) / no_rt_time) * 100 if no_rt_time > 0 else 0
            }
            
            # Test history size impact
            print("  Testing history size impact...")
            history_sizes = [100, 1000, 10000]
            
            for size in history_sizes:
                monitor = P2PMonitor(
                    active_devices=self.active_devices,
                    monitoring_level="basic",
                    max_history_size=size,
                    enable_real_time=False,
                    output_dir=os.path.join(self.output_dir, f"overhead_test_hist_{size}")
                )
                
                # Fill history
                start_time = time.time()
                for i in range(min(size, 100)):  # Limit to 100 for testing
                    tensor = torch.randn(100, 100)
                    monitor.record_operation(
                        operation_type="test",
                        device_id=self.active_devices[0],
                        tensor=tensor,
                        start_time=time.time(),
                        end_time=time.time(),
                        success=True
                    )
                end_time = time.time()
                
                overhead_results["history_size_impact"][size] = {
                    "history_size": size,
                    "operations_recorded": min(size, 100),
                    "total_time_ms": (end_time - start_time) * 1000,
                    "avg_time_per_operation_ms": ((end_time - start_time) * 1000) / min(size, 100)
                }
                
                monitor.close()
            
        except Exception as e:
            print(f"  Error in overhead analysis: {e}")
            overhead_results["error"] = str(e)
        
        self.results["overhead_analysis"] = overhead_results
        print("  Overhead analysis complete.")
    
    def _test_reliability(self):
        """Test reliability of monitoring tools."""
        print("\nTesting reliability...")
        
        reliability_results = {
            "error_handling": {},
            "resource_cleanup": {},
            "concurrent_access": {},
            "data_integrity": {}
        }
        
        try:
            # Test error handling
            print("  Testing error handling...")
            monitor = P2PMonitor(
                active_devices=self.active_devices,
                monitoring_level="basic",
                enable_real_time=False,
                output_dir=os.path.join(self.output_dir, "reliability_test")
            )
            
            # Test with invalid parameters
            try:
                monitor.record_operation(
                    operation_type="invalid_op",
                    device_id=-1,  # Invalid device
                    tensor=None,  # Invalid tensor
                    start_time=time.time(),
                    end_time=time.time(),
                    success=False
                )
                reliability_results["error_handling"]["invalid_parameters"] = "handled"
            except Exception as e:
                reliability_results["error_handling"]["invalid_parameters"] = f"error: {str(e)}"
            
            # Test resource cleanup
            print("  Testing resource cleanup...")
            
            # Create and destroy multiple monitors
            for i in range(5):
                test_monitor = P2PMonitor(
                    active_devices=self.active_devices,
                    monitoring_level="basic",
                    enable_real_time=False,
                    output_dir=os.path.join(self.output_dir, f"reliability_test_{i}")
                )
                
                # Record some operations
                for j in range(10):
                    tensor = torch.randn(10, 10)
                    test_monitor.record_operation(
                        operation_type="test",
                        device_id=self.active_devices[0],
                        tensor=tensor,
                        start_time=time.time(),
                        end_time=time.time(),
                        success=True
                    )
                
                # Close monitor
                test_monitor.close()
            
            reliability_results["resource_cleanup"]["multiple_monitors"] = "success"
            
            # Test concurrent access
            print("  Testing concurrent access...")
            import threading
            
            def worker(worker_id, results_list):
                try:
                    worker_monitor = P2PMonitor(
                        active_devices=self.active_devices,
                        monitoring_level="basic",
                        enable_real_time=False,
                        output_dir=os.path.join(self.output_dir, f"reliability_test_worker_{worker_id}")
                    )
                    
                    for i in range(10):
                        tensor = torch.randn(10, 10)
                        worker_monitor.record_operation(
                            operation_type="test",
                            device_id=self.active_devices[0],
                            tensor=tensor,
                            start_time=time.time(),
                            end_time=time.time(),
                            success=True
                        )
                    
                    results_list.append(worker_id)
                    worker_monitor.close()
                except Exception as e:
                    results_list.append(f"error: {str(e)}")
            
            threads = []
            results_list = []
            
            for i in range(3):
                thread = threading.Thread(target=worker, args=(i, results_list))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            reliability_results["concurrent_access"] = {
                "threads_completed": len([r for r in results_list if isinstance(r, int)]),
                "threads_failed": len([r for r in results_list if isinstance(r, str) and r.startswith("error")]),
                "total_threads": len(threads)
            }
            
            # Test data integrity
            print("  Testing data integrity...")
            
            # Record known operations
            known_operations = []
            for i in range(20):
                tensor = torch.randn(50, 50)
                start_time = time.time()
                end_time = time.time()
                
                monitor.record_operation(
                    operation_type="test",
                    device_id=self.active_devices[0],
                    tensor=tensor,
                    start_time=start_time,
                    end_time=end_time,
                    success=True
                )
                
                known_operations.append({
                    "tensor_size": tensor.numel() * tensor.element_size(),
                    "duration": end_time - start_time,
                    "success": True
                })
            
            # Retrieve and verify operations
            retrieved_operations = monitor.get_operation_history(limit=20)
            
            integrity_issues = 0
            for i, (known, retrieved) in enumerate(zip(known_operations, retrieved_operations)):
                if known["tensor_size"] != retrieved.tensor_size_bytes:
                    integrity_issues += 1
                if abs(known["duration"] - retrieved.duration_ms / 1000) > 0.01:  # 10ms tolerance
                    integrity_issues += 1
                if known["success"] != retrieved.success:
                    integrity_issues += 1
            
            reliability_results["data_integrity"] = {
                "operations_tested": len(known_operations),
                "integrity_issues": integrity_issues,
                "integrity_rate": (len(known_operations) - integrity_issues) / len(known_operations)
            }
            
            monitor.close()
            
        except Exception as e:
            print(f"  Error in reliability testing: {e}")
            reliability_results["error"] = str(e)
        
        self.results["reliability_tests"] = reliability_results
        print("  Reliability testing complete.")
    
    def _test_scalability(self):
        """Test scalability of monitoring tools."""
        print("\nTesting scalability...")
        
        scalability_results = {
            "operation_count": {},
            "device_count": {},
            "history_size": {},
            "memory_scaling": {}
        }
        
        try:
            # Test operation count scalability
            print("  Testing operation count scalability...")
            operation_counts = [100, 1000, 5000]
            
            for count in operation_counts:
                monitor = P2PMonitor(
                    active_devices=self.active_devices,
                    monitoring_level="basic",
                    max_history_size=count,
                    enable_real_time=False,
                    output_dir=os.path.join(self.output_dir, f"scalability_test_ops_{count}")
                )
                
                start_time = time.time()
                for i in range(count):
                    tensor = torch.randn(10, 10)
                    monitor.record_operation(
                        operation_type="test",
                        device_id=self.active_devices[0],
                        tensor=tensor,
                        start_time=time.time(),
                        end_time=time.time(),
                        success=True
                    )
                end_time = time.time()
                
                scalability_results["operation_count"][count] = {
                    "operations": count,
                    "total_time_ms": (end_time - start_time) * 1000,
                    "avg_time_per_operation_ms": ((end_time - start_time) * 1000) / count,
                    "throughput_ops_per_sec": count / (end_time - start_time)
                }
                
                monitor.close()
            
            # Test device count scalability
            print("  Testing device count scalability...")
            device_counts = [[0], [0, 1], [0, 1, 2]]  # Mock device configurations
            
            for devices in device_counts:
                monitor = P2PMonitor(
                    active_devices=devices,
                    monitoring_level="basic",
                    enable_real_time=False,
                    output_dir=os.path.join(self.output_dir, f"scalability_test_dev_{len(devices)}")
                )
                
                start_time = time.time()
                for i in range(100):
                    tensor = torch.randn(10, 10)
                    for device in devices:
                        monitor.record_operation(
                            operation_type="test",
                            device_id=device,
                            tensor=tensor,
                            start_time=time.time(),
                            end_time=time.time(),
                            success=True
                        )
                end_time = time.time()
                
                scalability_results["device_count"][len(devices)] = {
                    "devices": len(devices),
                    "total_operations": 100 * len(devices),
                    "total_time_ms": (end_time - start_time) * 1000,
                    "avg_time_per_operation_ms": ((end_time - start_time) * 1000) / (100 * len(devices))
                }
                
                monitor.close()
            
            # Test history size scalability
            print("  Testing history size scalability...")
            history_sizes = [1000, 10000, 50000]
            
            for size in history_sizes:
                monitor = P2PMonitor(
                    active_devices=self.active_devices,
                    monitoring_level="basic",
                    max_history_size=size,
                    enable_real_time=False,
                    output_dir=os.path.join(self.output_dir, f"scalability_test_hist_{size}")
                )
                
                # Fill history
                start_time = time.time()
                for i in range(min(size, 1000)):  # Limit to 1000 for testing
                    tensor = torch.randn(10, 10)
                    monitor.record_operation(
                        operation_type="test",
                        device_id=self.active_devices[0],
                        tensor=tensor,
                        start_time=time.time(),
                        end_time=time.time(),
                        success=True
                    )
                end_time = time.time()
                
                scalability_results["history_size"][size] = {
                    "history_size": size,
                    "operations_recorded": min(size, 1000),
                    "total_time_ms": (end_time - start_time) * 1000,
                    "avg_time_per_operation_ms": ((end_time - start_time) * 1000) / min(size, 1000)
                }
                
                monitor.close()
            
            # Test memory scaling
            print("  Testing memory scaling...")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                memory_before = torch.cuda.memory_allocated(self.active_devices[0])
            else:
                memory_before = 0
            
            monitor = P2PMonitor(
                active_devices=self.active_devices,
                monitoring_level="comprehensive",
                max_history_size=10000,
                enable_real_time=False,
                output_dir=os.path.join(self.output_dir, "scalability_test_memory")
            )
            
            # Record many operations
            for i in range(1000):
                tensor = torch.randn(100, 100)
                monitor.record_operation(
                    operation_type="test",
                    device_id=self.active_devices[0],
                    tensor=tensor,
                    start_time=time.time(),
                    end_time=time.time(),
                    success=True
                )
            
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated(self.active_devices[0])
            else:
                memory_after = 0
            
            scalability_results["memory_scaling"] = {
                "operations": 1000,
                "memory_before_bytes": memory_before,
                "memory_after_bytes": memory_after,
                "memory_increase_bytes": memory_after - memory_before,
                "memory_per_operation_bytes": (memory_after - memory_before) / 1000 if memory_after > memory_before else 0
            }
            
            monitor.close()
            
        except Exception as e:
            print(f"  Error in scalability testing: {e}")
            scalability_results["error"] = str(e)
        
        self.results["scalability_tests"] = scalability_results
        print("  Scalability testing complete.")
    
    def _generate_summary(self):
        """Generate validation summary."""
        print("\nGenerating validation summary...")
        
        summary = {
            "validation_timestamp": time.time(),
            "overall_status": "success",
            "critical_issues": [],
            "recommendations": [],
            "performance_impact_acceptable": True,
            "monitoring_reliable": True
        }
        
        # Check accuracy tests
        accuracy = self.results.get("accuracy_tests", {})
        if "error" in accuracy:
            summary["critical_issues"].append("Accuracy tests failed")
            summary["overall_status"] = "failed"
        
        # Check performance impact
        performance = self.results.get("performance_impact", {})
        if "monitoring_overhead" in performance:
            overhead = performance["monitoring_overhead"].get("overhead_percent", 0)
            if overhead > 10.0:  # More than 10% overhead is concerning
                summary["critical_issues"].append(f"High monitoring overhead: {overhead:.2f}%")
                summary["performance_impact_acceptable"] = False
        
        # Check reliability
        reliability = self.results.get("reliability_tests", {})
        if "concurrent_access" in reliability:
            failed_threads = reliability["concurrent_access"].get("threads_failed", 0)
            if failed_threads > 0:
                summary["critical_issues"].append(f"Concurrent access issues: {failed_threads} threads failed")
                summary["monitoring_reliable"] = False
        
        # Generate recommendations
        recommendations = []
        
        if performance.get("monitoring_overhead", {}).get("overhead_percent", 0) > 5.0:
            recommendations.append("Consider reducing monitoring level for better performance")
        
        if reliability.get("data_integrity", {}).get("integrity_rate", 1.0) < 0.95:
            recommendations.append("Investigate data integrity issues in monitoring")
        
        if self.results.get("scalability_tests", {}).get("operation_count", {}).get("5000", {}).get("avg_time_per_operation_ms", 0) > 1.0:
            recommendations.append("Consider optimizing monitoring for high operation counts")
        
        summary["recommendations"] = recommendations
        
        self.results["summary"] = summary
        print("  Validation summary generated.")
    
    def _save_results(self):
        """Save validation results."""
        print("\nSaving validation results...")
        
        # Save detailed results
        results_path = os.path.join(self.output_dir, "validation_results.json")
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save summary report
        summary_path = os.path.join(self.output_dir, "validation_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("P2P Monitoring Validation Summary\n")
            f.write("=" * 40 + "\n\n")
            
            summary = self.results.get("summary", {})
            f.write(f"Overall Status: {summary.get('overall_status', 'unknown')}\n")
            f.write(f"Validation Timestamp: {summary.get('validation_timestamp', 0)}\n")
            f.write(f"Performance Impact Acceptable: {summary.get('performance_impact_acceptable', False)}\n")
            f.write(f"Monitoring Reliable: {summary.get('monitoring_reliable', False)}\n\n")
            
            if summary.get("critical_issues"):
                f.write("Critical Issues:\n")
                for issue in summary["critical_issues"]:
                    f.write(f"  - {issue}\n")
                f.write("\n")
            
            if summary.get("recommendations"):
                f.write("Recommendations:\n")
                for rec in summary["recommendations"]:
                    f.write(f"  - {rec}\n")
                f.write("\n")
        
        print(f"  Results saved to: {self.output_dir}")
        print(f"  Detailed results: {results_path}")
        print(f"  Summary report: {summary_path}")


def main():
    """Main function for validation."""
    parser = argparse.ArgumentParser(description="P2P Monitoring Validation Tool")
    parser.add_argument("--output-dir", default="./monitoring_validation",
                        help="Output directory for validation results")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Create validator
    validator = P2PMonitoringValidator(output_dir=args.output_dir)
    
    # Run validations
    results = validator.run_all_validations()
    
    # Print summary
    summary = results.get("summary", {})
    print(f"\nValidation Status: {summary.get('overall_status', 'unknown')}")
    
    if summary.get("critical_issues"):
        print("Critical Issues Found:")
        for issue in summary["critical_issues"]:
            print(f"  - {issue}")
    
    if summary.get("recommendations"):
        print("Recommendations:")
        for rec in summary["recommendations"]:
            print(f"  - {rec}")
    
    # Exit with appropriate code
    exit_code = 0 if summary.get("overall_status") == "success" else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()