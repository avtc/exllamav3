"""
P2P Performance Monitoring for ExLlamaV3

This module provides comprehensive performance monitoring and debugging tools
for P2P GPU communication operations in ExLlamaV3 tensor parallelism.
"""

import time
import threading
import json
import os
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import numpy as np
import torch

from .debug import log_tp
from ..model.model_tp_p2p import P2PTopology


@dataclass
class P2POperationMetrics:
    """Metrics for a single P2P operation."""
    operation_type: str  # "broadcast", "all_reduce", "gather", "direct_copy"
    device_id: int
    peer_device: Optional[int]  # None for collective operations
    tensor_size_bytes: int
    tensor_shape: Tuple[int, ...]
    dtype: str
    start_time: float
    end_time: float
    duration_ms: float
    bandwidth_gbps: float
    algorithm: Optional[str]  # e.g., "ring", "binary_tree", "kary_tree"
    success: bool
    error_message: Optional[str] = None
    topology_info: Optional[Dict] = None


@dataclass
class P2PDeviceMetrics:
    """Metrics for a specific device."""
    device_id: int
    total_operations: int
    successful_operations: int
    failed_operations: int
    total_bytes_transferred: int
    total_time_ms: float
    average_bandwidth_gbps: float
    peak_bandwidth_gbps: float
    memory_pool_usage_bytes: int
    memory_pool_total_bytes: int
    gpu_utilization: float
    gpu_memory_used: int
    gpu_memory_total: int
    peer_access_enabled: List[int]
    last_update: float


@dataclass
class P2PTopologyMetrics:
    """Metrics for P2P topology."""
    num_devices: int
    connectivity_ratio: float
    is_fully_connected: bool
    topology_type: str
    tree_depth: int
    average_branching_factor: float
    communication_patterns: Dict[str, int]


class P2PMonitor:
    """
    Comprehensive P2P performance monitoring system.
    
    This class collects, analyzes, and reports performance metrics for P2P operations
    across multiple GPU devices, providing insights into communication patterns,
    bottlenecks, and optimization opportunities.
    """
    
    def __init__(
        self,
        active_devices: List[int],
        monitoring_level: str = "basic",
        max_history_size: int = 10000,
        enable_real_time: bool = True,
        output_dir: Optional[str] = None
    ):
        """
        Initialize P2P monitor.
        
        Args:
            active_devices: List of active GPU device IDs
            monitoring_level: Level of monitoring ("basic", "detailed", "comprehensive")
            max_history_size: Maximum number of operations to keep in history
            enable_real_time: Enable real-time monitoring
            output_dir: Directory to save monitoring reports
        """
        self.active_devices = active_devices
        self.num_devices = len(active_devices)
        self.monitoring_level = monitoring_level
        self.max_history_size = max_history_size
        self.enable_real_time = enable_real_time
        self.output_dir = output_dir or "./p2p_monitoring"
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Operation history
        self.operation_history: deque = deque(maxlen=max_history_size)
        self.operation_lock = threading.Lock()
        
        # Device metrics
        self.device_metrics: Dict[int, P2PDeviceMetrics] = {}
        self.metrics_lock = threading.Lock()
        
        # Topology information
        self.topology: Optional[P2PTopology] = None
        self.topology_metrics: Optional[P2PTopologyMetrics] = None
        
        # Performance statistics
        self.performance_stats = defaultdict(list)
        self.stats_lock = threading.Lock()
        
        # Real-time monitoring thread
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        
        # Initialize device metrics
        self._initialize_device_metrics()
        
        # Start real-time monitoring if enabled
        if self.enable_real_time:
            self._start_real_time_monitoring()
    
    def _initialize_device_metrics(self):
        """Initialize metrics for each device."""
        for device_id in self.active_devices:
            self.device_metrics[device_id] = P2PDeviceMetrics(
                device_id=device_id,
                total_operations=0,
                successful_operations=0,
                failed_operations=0,
                total_bytes_transferred=0,
                total_time_ms=0.0,
                average_bandwidth_gbps=0.0,
                peak_bandwidth_gbps=0.0,
                memory_pool_usage_bytes=0,
                memory_pool_total_bytes=0,
                gpu_utilization=0.0,
                gpu_memory_used=0,
                gpu_memory_total=0,
                peer_access_enabled=[],
                last_update=time.time()
            )
    
    def set_topology(self, topology: P2PTopology):
        """Set P2P topology information."""
        self.topology = topology
        self._update_topology_metrics()
    
    def _update_topology_metrics(self):
        """Update topology metrics."""
        if not self.topology:
            return
        
        topology_summary = self.topology.get_topology_summary()
        tree_stats = self.topology.get_tree_stats(self.topology.communication_tree) if self.topology.communication_tree else {}
        
        self.topology_metrics = P2PTopologyMetrics(
            num_devices=self.num_devices,
            connectivity_ratio=topology_summary.get("connectivity_ratio", 0.0),
            is_fully_connected=topology_summary.get("is_fully_connected", False),
            topology_type=topology_summary.get("topology_type", "unknown"),
            tree_depth=tree_stats.get("tree_depth", 0),
            average_branching_factor=tree_stats.get("avg_branching_factor", 0.0),
            communication_patterns=defaultdict(int)
        )
    
    def record_operation(
        self,
        operation_type: str,
        device_id: int,
        peer_device: Optional[int],
        tensor: torch.Tensor,
        start_time: float,
        end_time: float,
        algorithm: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ):
        """
        Record a P2P operation.
        
        Args:
            operation_type: Type of operation
            device_id: Device ID performing the operation
            peer_device: Peer device ID (None for collective operations)
            tensor: Tensor being transferred
            start_time: Operation start time
            end_time: Operation end time
            algorithm: Algorithm used (if applicable)
            success: Whether operation was successful
            error_message: Error message (if failed)
        """
        duration_ms = (end_time - start_time) * 1000.0
        tensor_size_bytes = tensor.numel() * tensor.element_size()
        
        # Calculate bandwidth (GB/s)
        bandwidth_gbps = 0.0
        if duration_ms > 0:
            bandwidth_gbps = (tensor_size_bytes / (1024**3)) / (duration_ms / 1000.0)
        
        # Create operation metrics
        metrics = P2POperationMetrics(
            operation_type=operation_type,
            device_id=device_id,
            peer_device=peer_device,
            tensor_size_bytes=tensor_size_bytes,
            tensor_shape=tuple(tensor.shape),
            dtype=str(tensor.dtype),
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            bandwidth_gbps=bandwidth_gbps,
            algorithm=algorithm,
            success=success,
            error_message=error_message,
            topology_info=self.topology.get_topology_summary() if self.topology else None
        )
        
        # Add to history
        with self.operation_lock:
            self.operation_history.append(metrics)
        
        # Update device metrics
        self._update_device_metrics(device_id, metrics)
        
        # Update performance statistics
        self._update_performance_stats(metrics)
        
        # Update topology metrics
        if self.topology_metrics:
            self.topology_metrics.communication_patterns[operation_type] += 1
    
    def _update_device_metrics(self, device_id: int, metrics: P2POperationMetrics):
        """Update metrics for a specific device."""
        with self.metrics_lock:
            if device_id not in self.device_metrics:
                self._initialize_device_metrics()
            
            device_metrics = self.device_metrics[device_id]
            device_metrics.total_operations += 1
            
            if metrics.success:
                device_metrics.successful_operations += 1
                device_metrics.total_bytes_transferred += metrics.tensor_size_bytes
                device_metrics.total_time_ms += metrics.duration_ms
                
                # Update bandwidth metrics
                if device_metrics.total_time_ms > 0:
                    device_metrics.average_bandwidth_gbps = (
                        device_metrics.total_bytes_transferred / (1024**3) / 
                        (device_metrics.total_time_ms / 1000.0)
                    )
                
                device_metrics.peak_bandwidth_gbps = max(
                    device_metrics.peak_bandwidth_gbps, metrics.bandwidth_gbps
                )
            else:
                device_metrics.failed_operations += 1
            
            device_metrics.last_update = time.time()
    
    def _update_performance_stats(self, metrics: P2POperationMetrics):
        """Update performance statistics."""
        with self.stats_lock:
            key = f"{metrics.operation_type}_{metrics.algorithm or 'default'}"
            self.performance_stats[key].append({
                "duration_ms": metrics.duration_ms,
                "bandwidth_gbps": metrics.bandwidth_gbps,
                "tensor_size_bytes": metrics.tensor_size_bytes,
                "timestamp": metrics.end_time
            })
    
    def _start_real_time_monitoring(self):
        """Start real-time monitoring thread."""
        self.monitoring_thread = threading.Thread(
            target=self._real_time_monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
    
    def _real_time_monitoring_loop(self):
        """Real-time monitoring loop."""
        while not self.stop_monitoring.is_set():
            try:
                self._update_gpu_metrics()
                time.sleep(1.0)  # Update every second
            except Exception as e:
                log_tp(None, f"P2P monitoring error: {e}")
    
    def _update_gpu_metrics(self):
        """Update GPU-related metrics."""
        try:
            for device_id in self.active_devices:
                if device_id >= 0:  # Skip CPU device
                    with torch.cuda.device(device_id):
                        # Get GPU memory info
                        memory_used = torch.cuda.memory_allocated(device_id)
                        memory_total = torch.cuda.get_device_properties(device_id).total_memory
                        
                        # Get GPU utilization (simplified)
                        utilization = min(100.0, (memory_used / memory_total) * 100)
                        
                        with self.metrics_lock:
                            if device_id in self.device_metrics:
                                self.device_metrics[device_id].gpu_utilization = utilization
                                self.device_metrics[device_id].gpu_memory_used = memory_used
                                self.device_metrics[device_id].gpu_memory_total = memory_total
        except Exception as e:
            log_tp(None, f"Failed to update GPU metrics: {e}")
    
    def get_device_metrics(self, device_id: int) -> Optional[P2PDeviceMetrics]:
        """Get metrics for a specific device."""
        with self.metrics_lock:
            return self.device_metrics.get(device_id)
    
    def get_all_device_metrics(self) -> Dict[int, P2PDeviceMetrics]:
        """Get metrics for all devices."""
        with self.metrics_lock:
            return self.device_metrics.copy()
    
    def get_operation_history(
        self,
        operation_type: Optional[str] = None,
        device_id: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List[P2POperationMetrics]:
        """
        Get operation history with optional filtering.
        
        Args:
            operation_type: Filter by operation type
            device_id: Filter by device ID
            limit: Maximum number of operations to return
            
        Returns:
            List of operation metrics
        """
        with self.operation_lock:
            history = list(self.operation_history)
        
        # Apply filters
        if operation_type:
            history = [op for op in history if op.operation_type == operation_type]
        
        if device_id is not None:
            history = [op for op in history if op.device_id == device_id]
        
        # Sort by timestamp (newest first)
        history.sort(key=lambda x: x.end_time, reverse=True)
        
        # Apply limit
        if limit:
            history = history[:limit]
        
        return history
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        summary = {
            "monitoring_info": {
                "active_devices": self.active_devices,
                "num_devices": self.num_devices,
                "monitoring_level": self.monitoring_level,
                "total_operations": len(self.operation_history),
                "last_update": time.time()
            },
            "topology": asdict(self.topology_metrics) if self.topology_metrics else None,
            "device_metrics": {},
            "operation_stats": {},
            "performance_analysis": {}
        }
        
        # Add device metrics
        with self.metrics_lock:
            for device_id, metrics in self.device_metrics.items():
                summary["device_metrics"][device_id] = asdict(metrics)
        
        # Add operation statistics
        operation_counts = defaultdict(int)
        operation_success_rates = defaultdict(lambda: {"total": 0, "success": 0})
        
        with self.operation_lock:
            for op in self.operation_history:
                operation_counts[op.operation_type] += 1
                operation_success_rates[op.operation_type]["total"] += 1
                if op.success:
                    operation_success_rates[op.operation_type]["success"] += 1
        
        for op_type, count in operation_counts.items():
            success_rate = (
                operation_success_rates[op_type]["success"] / 
                operation_success_rates[op_type]["total"] * 100.0
            )
            summary["operation_stats"][op_type] = {
                "count": count,
                "success_rate_percent": success_rate
            }
        
        # Add performance analysis
        with self.stats_lock:
            for key, stats in self.performance_stats.items():
                if stats:
                    durations = [s["duration_ms"] for s in stats]
                    bandwidths = [s["bandwidth_gbps"] for s in stats]
                    
                    summary["performance_analysis"][key] = {
                        "avg_duration_ms": np.mean(durations),
                        "min_duration_ms": np.min(durations),
                        "max_duration_ms": np.max(durations),
                        "avg_bandwidth_gbps": np.mean(bandwidths),
                        "min_bandwidth_gbps": np.min(bandwidths),
                        "max_bandwidth_gbps": np.max(bandwidths),
                        "sample_count": len(stats)
                    }
        
        return summary
    
    def export_metrics(self, filename: Optional[str] = None) -> str:
        """
        Export metrics to JSON file.
        
        Args:
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to exported file
        """
        if not filename:
            timestamp = int(time.time())
            filename = f"p2p_metrics_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Prepare export data
        export_data = {
            "export_timestamp": time.time(),
            "summary": self.get_performance_summary(),
            "operation_history": [asdict(op) for op in self.operation_history],
            "performance_stats": dict(self.performance_stats)
        }
        
        # Write to file
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        log_tp(None, f"P2P metrics exported to {filepath}")
        return filepath
    
    def identify_bottlenecks(self) -> Dict[str, Any]:
        """
        Identify performance bottlenecks.
        
        Returns:
            Dictionary with bottleneck analysis
        """
        bottlenecks = {
            "slow_operations": [],
            "low_bandwidth_pairs": [],
            "high_failure_rates": [],
            "memory_issues": [],
            "topology_issues": []
        }
        
        # Analyze slow operations
        with self.operation_lock:
            for op in self.operation_history:
                if not op.success:
                    continue
                
                # Flag operations slower than 2x average
                if op.operation_type in self.performance_stats:
                    stats = self.performance_stats[f"{op.operation_type}_{op.algorithm or 'default'}"]
                    if stats:
                        avg_duration = np.mean([s["duration_ms"] for s in stats])
                        if op.duration_ms > avg_duration * 2:
                            bottlenecks["slow_operations"].append({
                                "operation_type": op.operation_type,
                                "device_id": op.device_id,
                                "peer_device": op.peer_device,
                                "duration_ms": op.duration_ms,
                                "avg_duration_ms": avg_duration,
                                "slowdown_factor": op.duration_ms / avg_duration
                            })
        
        # Analyze low bandwidth pairs
        bandwidth_pairs = defaultdict(list)
        with self.operation_lock:
            for op in self.operation_history:
                if op.success and op.peer_device is not None:
                    pair = tuple(sorted([op.device_id, op.peer_device]))
                    bandwidth_pairs[pair].append(op.bandwidth_gbps)
        
        for pair, bandwidths in bandwidth_pairs.items():
            if bandwidths:
                avg_bandwidth = np.mean(bandwidths)
                if avg_bandwidth < 5.0:  # Less than 5 GB/s
                    bottlenecks["low_bandwidth_pairs"].append({
                        "device_pair": pair,
                        "avg_bandwidth_gbps": avg_bandwidth,
                        "sample_count": len(bandwidths)
                    })
        
        # Analyze high failure rates
        operation_failures = defaultdict(lambda: {"total": 0, "failed": 0})
        with self.operation_lock:
            for op in self.operation_history:
                operation_failures[op.operation_type]["total"] += 1
                if not op.success:
                    operation_failures[op.operation_type]["failed"] += 1
        
        for op_type, counts in operation_failures.items():
            if counts["total"] > 0:
                failure_rate = counts["failed"] / counts["total"] * 100.0
                if failure_rate > 5.0:  # More than 5% failure rate
                    bottlenecks["high_failure_rates"].append({
                        "operation_type": op_type,
                        "failure_rate_percent": failure_rate,
                        "failed_count": counts["failed"],
                        "total_count": counts["total"]
                    })
        
        # Analyze memory issues
        with self.metrics_lock:
            for device_id, metrics in self.device_metrics.items():
                if metrics.gpu_memory_total > 0:
                    memory_usage_percent = (metrics.gpu_memory_used / metrics.gpu_memory_total) * 100.0
                    if memory_usage_percent > 90.0:  # More than 90% memory usage
                        bottlenecks["memory_issues"].append({
                            "device_id": device_id,
                            "memory_usage_percent": memory_usage_percent,
                            "memory_used_gb": metrics.gpu_memory_used / (1024**3),
                            "memory_total_gb": metrics.gpu_memory_total / (1024**3)
                        })
        
        # Analyze topology issues
        if self.topology_metrics:
            if self.topology_metrics.connectivity_ratio < 0.5:
                bottlenecks["topology_issues"].append({
                    "issue": "low_connectivity",
                    "connectivity_ratio": self.topology_metrics.connectivity_ratio,
                    "description": "Low P2P connectivity ratio may impact performance"
                })
            
            if self.topology_metrics.tree_depth > 5:
                bottlenecks["topology_issues"].append({
                    "issue": "deep_tree",
                    "tree_depth": self.topology_metrics.tree_depth,
                    "description": "Deep communication tree may increase latency"
                })
        
        return bottlenecks
    
    def get_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """
        Get optimization suggestions based on performance analysis.
        
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        bottlenecks = self.identify_bottlenecks()
        
        # Suggestions for slow operations
        if bottlenecks["slow_operations"]:
            suggestions.append({
                "category": "performance",
                "priority": "high",
                "suggestion": "Consider using different communication algorithms for slow operations",
                "details": f"Found {len(bottlenecks['slow_operations'])} operations slower than average"
            })
        
        # Suggestions for low bandwidth pairs
        if bottlenecks["low_bandwidth_pairs"]:
            suggestions.append({
                "category": "connectivity",
                "priority": "medium",
                "suggestion": "Check P2P connections between low bandwidth device pairs",
                "details": f"Found {len(bottlenecks['low_bandwidth_pairs'])} low bandwidth pairs"
            })
        
        # Suggestions for high failure rates
        if bottlenecks["high_failure_rates"]:
            suggestions.append({
                "category": "reliability",
                "priority": "high",
                "suggestion": "Investigate and fix operations with high failure rates",
                "details": f"Found {len(bottlenecks['high_failure_rates'])} operation types with >5% failure rate"
            })
        
        # Suggestions for memory issues
        if bottlenecks["memory_issues"]:
            suggestions.append({
                "category": "memory",
                "priority": "high",
                "suggestion": "Optimize memory usage or consider memory pooling",
                "details": f"Found {len(bottlenecks['memory_issues'])} devices with >90% memory usage"
            })
        
        # Suggestions for topology issues
        if bottlenecks["topology_issues"]:
            suggestions.append({
                "category": "topology",
                "priority": "medium",
                "suggestion": "Optimize communication topology",
                "details": f"Found {len(bottlenecks['topology_issues'])} topology issues"
            })
        
        # General suggestions
        if self.topology_metrics and not self.topology_metrics.is_fully_connected:
            suggestions.append({
                "category": "topology",
                "priority": "low",
                "suggestion": "Consider enabling P2P between more GPU pairs for better performance",
                "details": f"Current connectivity ratio: {self.topology_metrics.connectivity_ratio:.2f}"
            })
        
        return suggestions
    
    def reset_metrics(self):
        """Reset all metrics and history."""
        with self.operation_lock:
            self.operation_history.clear()
        
        with self.metrics_lock:
            self._initialize_device_metrics()
        
        with self.stats_lock:
            self.performance_stats.clear()
        
        log_tp(None, "P2P metrics reset")
    
    def close(self):
        """Close monitor and cleanup resources."""
        if self.monitoring_thread:
            self.stop_monitoring.set()
            self.monitoring_thread.join(timeout=2.0)
        
        # Export final metrics
        try:
            self.export_metrics("final_metrics.json")
        except Exception as e:
            log_tp(None, f"Failed to export final metrics: {e}")
        
        log_tp(None, "P2P monitor closed")


# Global monitor instance
_global_monitor: Optional[P2PMonitor] = None


def get_global_monitor() -> Optional[P2PMonitor]:
    """Get the global monitor instance."""
    return _global_monitor


def initialize_global_monitor(
    active_devices: List[int],
    monitoring_level: str = "basic",
    max_history_size: int = 10000,
    enable_real_time: bool = True,
    output_dir: Optional[str] = None
) -> P2PMonitor:
    """
    Initialize the global monitor instance.
    
    Args:
        active_devices: List of active GPU device IDs
        monitoring_level: Level of monitoring
        max_history_size: Maximum history size
        enable_real_time: Enable real-time monitoring
        output_dir: Output directory
        
    Returns:
        Global monitor instance
    """
    global _global_monitor
    
    if _global_monitor:
        _global_monitor.close()
    
    _global_monitor = P2PMonitor(
        active_devices=active_devices,
        monitoring_level=monitoring_level,
        max_history_size=max_history_size,
        enable_real_time=enable_real_time,
        output_dir=output_dir
    )
    
    return _global_monitor


def close_global_monitor():
    """Close the global monitor instance."""
    global _global_monitor
    
    if _global_monitor:
        _global_monitor.close()
        _global_monitor = None