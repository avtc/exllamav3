"""
P2P Performance Profiling for ExLlamaV3

This module provides comprehensive profiling tools for P2P GPU communication operations,
including performance comparison, bottleneck analysis, and optimization recommendations.
"""

import time
import threading
import json
import os
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import numpy as np
import torch
import contextlib

from .debug import log_tp
from .p2p_monitor import P2PMonitor, get_global_monitor
from ..model.model_tp_p2p import P2PTopology


@dataclass
class P2PProfileResult:
    """Result of a profiling session."""
    session_id: str
    timestamp: float
    operation_type: str
    algorithm: Optional[str]
    tensor_shapes: List[Tuple[int, ...]]
    tensor_dtypes: List[str]
    device_pairs: List[Tuple[int, int]]
    durations_ms: List[float]
    bandwidths_gbps: List[float]
    latencies_us: List[float]
    memory_usage_mb: List[float]
    gpu_utilizations: List[float]
    success_rates: List[float]
    statistics: Dict[str, float]
    recommendations: List[str]


@dataclass
class P2PComparisonResult:
    """Result of comparing different algorithms or configurations."""
    comparison_id: str
    timestamp: float
    operation_type: str
    algorithms: List[str]
    performance_metrics: Dict[str, Dict[str, float]]
    winner: str
    improvement_percent: float
    analysis: Dict[str, Any]


class P2PProfiler:
    """
    Comprehensive P2P performance profiler.
    
    This class provides detailed profiling capabilities for P2P operations,
    including performance comparison, bottleneck analysis, and optimization recommendations.
    """
    
    def __init__(
        self,
        monitor: Optional[P2PMonitor] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize P2P profiler.
        
        Args:
            monitor: P2P monitor instance (uses global if None)
            output_dir: Directory to save profiling results
        """
        self.monitor = monitor or get_global_monitor()
        self.output_dir = output_dir or "./p2p_profiling"
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Profiling sessions
        self.sessions: Dict[str, P2PProfileResult] = {}
        self.comparisons: Dict[str, P2PComparisonResult] = {}
        
        # Profiling state
        self.profiling_active = False
        self.current_session: Optional[str] = None
        self.session_data: Dict[str, Any] = {}
        self.session_lock = threading.Lock()
    
    @contextlib.contextmanager
    def profile_session(
        self,
        session_id: str,
        operation_type: str,
        algorithm: Optional[str] = None
    ):
        """
        Context manager for profiling a session.
        
        Args:
            session_id: Unique session identifier
            operation_type: Type of operation being profiled
            algorithm: Algorithm being used (if applicable)
        """
        with self.session_lock:
            self.current_session = session_id
            self.session_data = {
                "session_id": session_id,
                "timestamp": time.time(),
                "operation_type": operation_type,
                "algorithm": algorithm,
                "tensor_shapes": [],
                "tensor_dtypes": [],
                "device_pairs": [],
                "durations_ms": [],
                "bandwidths_gbps": [],
                "latencies_us": [],
                "memory_usage_mb": [],
                "gpu_utilizations": [],
                "success_rates": [],
                "start_time": time.time()
            }
        
        try:
            yield self
        finally:
            with self.session_lock:
                if self.current_session:
                    self._finalize_session()
                    self.current_session = None
    
    def record_operation(
        self,
        tensor: torch.Tensor,
        device_id: int,
        peer_device: Optional[int] = None,
        duration_ms: Optional[float] = None,
        bandwidth_gbps: Optional[float] = None,
        success: bool = True
    ):
        """
        Record an operation within the current profiling session.
        
        Args:
            tensor: Tensor being transferred
            device_id: Device ID
            peer_device: Peer device ID (if applicable)
            duration_ms: Operation duration in milliseconds
            bandwidth_gbps: Bandwidth in GB/s
            success: Whether operation was successful
        """
        if not self.current_session:
            return
        
        with self.session_lock:
            if self.current_session not in self.session_data:
                return
            
            data = self.session_data[self.current_session] if isinstance(self.session_data, dict) else self.session_data
            
            # Record tensor information
            data["tensor_shapes"].append(tuple(tensor.shape))
            data["tensor_dtypes"].append(str(tensor.dtype))
            
            if peer_device is not None:
                data["device_pairs"].append((device_id, peer_device))
            else:
                data["device_pairs"].append((device_id, device_id))
            
            # Record performance metrics
            if duration_ms is not None:
                data["durations_ms"].append(duration_ms)
            
            if bandwidth_gbps is not None:
                data["bandwidths_gbps"].append(bandwidth_gbps)
            
            # Record memory usage
            if device_id >= 0:
                try:
                    memory_used = torch.cuda.memory_allocated(device_id) / (1024**2)  # MB
                    data["memory_usage_mb"].append(memory_used)
                except:
                    data["memory_usage_mb"].append(0.0)
            else:
                data["memory_usage_mb"].append(0.0)
            
            # Record GPU utilization (simplified)
            try:
                if device_id >= 0:
                    total_memory = torch.cuda.get_device_properties(device_id).total_memory
                    utilization = min(100.0, (torch.cuda.memory_allocated(device_id) / total_memory) * 100)
                    data["gpu_utilizations"].append(utilization)
                else:
                    data["gpu_utilizations"].append(0.0)
            except:
                data["gpu_utilizations"].append(0.0)
            
            # Record success rate
            data["success_rates"].append(1.0 if success else 0.0)
    
    def _finalize_session(self):
        """Finalize the current profiling session."""
        if not self.current_session or not isinstance(self.session_data, dict):
            return
        
        data = self.session_data
        end_time = time.time()
        
        # Calculate statistics
        statistics = {}
        
        if data["durations_ms"]:
            statistics.update({
                "avg_duration_ms": np.mean(data["durations_ms"]),
                "min_duration_ms": np.min(data["durations_ms"]),
                "max_duration_ms": np.max(data["durations_ms"]),
                "std_duration_ms": np.std(data["durations_ms"])
            })
        
        if data["bandwidths_gbps"]:
            statistics.update({
                "avg_bandwidth_gbps": np.mean(data["bandwidths_gbps"]),
                "min_bandwidth_gbps": np.min(data["bandwidths_gbps"]),
                "max_bandwidth_gbps": np.max(data["bandwidths_gbps"]),
                "std_bandwidth_gbps": np.std(data["bandwidths_gbps"])
            })
        
        if data["memory_usage_mb"]:
            statistics.update({
                "avg_memory_mb": np.mean(data["memory_usage_mb"]),
                "max_memory_mb": np.max(data["memory_usage_mb"])
            })
        
        if data["success_rates"]:
            statistics.update({
                "success_rate": np.mean(data["success_rates"]),
                "total_operations": len(data["success_rates"])
            })
        
        # Generate recommendations
        recommendations = self._generate_recommendations(data, statistics)
        
        # Create profile result
        result = P2PProfileResult(
            session_id=data["session_id"],
            timestamp=data["timestamp"],
            operation_type=data["operation_type"],
            algorithm=data["algorithm"],
            tensor_shapes=data["tensor_shapes"],
            tensor_dtypes=data["tensor_dtypes"],
            device_pairs=data["device_pairs"],
            durations_ms=data["durations_ms"],
            bandwidths_gbps=data["bandwidths_gbps"],
            latencies_us=data["latencies_us"],
            memory_usage_mb=data["memory_usage_mb"],
            gpu_utilizations=data["gpu_utilizations"],
            success_rates=data["success_rates"],
            statistics=statistics,
            recommendations=recommendations
        )
        
        self.sessions[data["session_id"]] = result
        
        # Export session results
        self._export_session_result(result)
    
    def _generate_recommendations(
        self,
        data: Dict[str, Any],
        statistics: Dict[str, float]
    ) -> List[str]:
        """Generate optimization recommendations based on profiling data."""
        recommendations = []
        
        # Performance recommendations
        if "avg_duration_ms" in statistics and statistics["avg_duration_ms"] > 100.0:
            recommendations.append("Consider optimizing for faster communication (average duration > 100ms)")
        
        if "avg_bandwidth_gbps" in statistics and statistics["avg_bandwidth_gbps"] < 10.0:
            recommendations.append("Low bandwidth detected - consider using different communication patterns")
        
        # Memory recommendations
        if "avg_memory_mb" in statistics and statistics["avg_memory_mb"] > 1000.0:
            recommendations.append("High memory usage - consider memory pooling or optimization")
        
        # Success rate recommendations
        if "success_rate" in statistics and statistics["success_rate"] < 0.95:
            recommendations.append("Low success rate - investigate reliability issues")
        
        # Tensor size recommendations
        if data["tensor_shapes"]:
            avg_elements = np.mean([np.prod(shape) for shape in data["tensor_shapes"]])
            if avg_elements > 1000000:  # Large tensors
                recommendations.append("Large tensors detected - consider chunking or compression")
        
        # Device pair recommendations
        if data["device_pairs"]:
            unique_pairs = set(data["device_pairs"])
            if len(unique_pairs) > len(data["device_pairs"]) * 0.5:
                recommendations.append("Many different device pairs - consider topology optimization")
        
        return recommendations
    
    def _export_session_result(self, result: P2PProfileResult):
        """Export session result to file."""
        filename = f"profile_{result.session_id}_{int(result.timestamp)}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(asdict(result), f, indent=2)
        
        log_tp(None, f"Profile result exported to {filepath}")
    
    def compare_algorithms(
        self,
        comparison_id: str,
        operation_type: str,
        algorithms: List[str],
        sessions: Dict[str, str]  # algorithm -> session_id
    ) -> P2PComparisonResult:
        """
        Compare performance of different algorithms.
        
        Args:
            comparison_id: Unique comparison identifier
            operation_type: Type of operation
            algorithms: List of algorithm names
            sessions: Mapping of algorithm to session ID
            
        Returns:
            Comparison result
        """
        performance_metrics = {}
        
        for algorithm, session_id in sessions.items():
            if session_id not in self.sessions:
                continue
            
            session = self.sessions[session_id]
            metrics = {
                "avg_duration_ms": session.statistics.get("avg_duration_ms", 0.0),
                "avg_bandwidth_gbps": session.statistics.get("avg_bandwidth_gbps", 0.0),
                "success_rate": session.statistics.get("success_rate", 0.0),
                "avg_memory_mb": session.statistics.get("avg_memory_mb", 0.0),
                "total_operations": session.statistics.get("total_operations", 0)
            }
            
            performance_metrics[algorithm] = metrics
        
        # Determine winner (based on duration and bandwidth)
        if performance_metrics:
            # Calculate score (lower duration and higher bandwidth is better)
            scores = {}
            for algorithm, metrics in performance_metrics.items():
                duration_score = 1.0 / (metrics["avg_duration_ms"] + 1e-6)
                bandwidth_score = metrics["avg_bandwidth_gbps"]
                scores[algorithm] = duration_score + bandwidth_score
            
            winner = max(scores, key=scores.get)
            
            # Calculate improvement
            if len(performance_metrics) > 1:
                winner_metrics = performance_metrics[winner]
                other_metrics = [m for a, m in performance_metrics.items() if a != winner]
                
                if other_metrics:
                    avg_other_duration = np.mean([m["avg_duration_ms"] for m in other_metrics])
                    improvement_percent = ((avg_other_duration - winner_metrics["avg_duration_ms"]) / 
                                        avg_other_duration * 100.0)
                else:
                    improvement_percent = 0.0
            else:
                improvement_percent = 0.0
        else:
            winner = ""
            improvement_percent = 0.0
        
        # Generate analysis
        analysis = {
            "performance_ranking": sorted(
                performance_metrics.items(),
                key=lambda x: x[1]["avg_duration_ms"]
            ),
            "bandwidth_ranking": sorted(
                performance_metrics.items(),
                key=lambda x: x[1]["avg_bandwidth_gbps"],
                reverse=True
            ),
            "reliability_ranking": sorted(
                performance_metrics.items(),
                key=lambda x: x[1]["success_rate"],
                reverse=True
            )
        }
        
        # Create comparison result
        result = P2PComparisonResult(
            comparison_id=comparison_id,
            timestamp=time.time(),
            operation_type=operation_type,
            algorithms=algorithms,
            performance_metrics=performance_metrics,
            winner=winner,
            improvement_percent=improvement_percent,
            analysis=analysis
        )
        
        self.comparisons[comparison_id] = result
        
        # Export comparison result
        self._export_comparison_result(result)
        
        return result
    
    def _export_comparison_result(self, result: P2PComparisonResult):
        """Export comparison result to file."""
        filename = f"comparison_{result.comparison_id}_{int(result.timestamp)}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(asdict(result), f, indent=2)
        
        log_tp(None, f"Comparison result exported to {filepath}")
    
    def profile_operation(
        self,
        operation_func: Callable,
        operation_type: str,
        algorithm: Optional[str] = None,
        num_iterations: int = 10,
        warmup_iterations: int = 3,
        tensor: Optional[torch.Tensor] = None,
        device_id: Optional[int] = None,
        peer_device: Optional[int] = None
    ) -> P2PProfileResult:
        """
        Profile a specific operation with multiple iterations.
        
        Args:
            operation_func: Function to execute (should accept tensor and return result)
            operation_type: Type of operation
            algorithm: Algorithm being used
            num_iterations: Number of profiling iterations
            warmup_iterations: Number of warmup iterations
            tensor: Tensor to use for operation
            device_id: Device ID
            peer_device: Peer device ID
            
        Returns:
            Profile result
        """
        session_id = f"profile_{operation_type}_{algorithm}_{int(time.time())}"
        
        with self.profile_session(session_id, operation_type, algorithm):
            # Warmup iterations
            for _ in range(warmup_iterations):
                try:
                    if tensor is not None:
                        operation_func(tensor)
                    else:
                        operation_func()
                except:
                    pass
            
            # Profiling iterations
            for i in range(num_iterations):
                start_time = time.time()
                success = True
                
                try:
                    if tensor is not None:
                        result = operation_func(tensor)
                    else:
                        result = operation_func()
                except Exception as e:
                    success = False
                    log_tp(device_id, f"Operation failed: {e}")
                
                end_time = time.time()
                duration_ms = (end_time - start_time) * 1000.0
                
                # Calculate bandwidth if tensor is provided
                bandwidth_gbps = 0.0
                if tensor is not None and success:
                    tensor_size_bytes = tensor.numel() * tensor.element_size()
                    if duration_ms > 0:
                        bandwidth_gbps = (tensor_size_bytes / (1024**3)) / (duration_ms / 1000.0)
                
                # Record operation
                if tensor is not None:
                    self.record_operation(
                        tensor=tensor,
                        device_id=device_id or 0,
                        peer_device=peer_device,
                        duration_ms=duration_ms,
                        bandwidth_gbps=bandwidth_gbps,
                        success=success
                    )
        
        return self.sessions[session_id]
    
    def analyze_communication_patterns(self) -> Dict[str, Any]:
        """
        Analyze communication patterns from profiling data.
        
        Returns:
            Dictionary with pattern analysis
        """
        patterns = {
            "device_pair_frequency": defaultdict(int),
            "tensor_size_distribution": [],
            "operation_frequency": defaultdict(int),
            "algorithm_frequency": defaultdict(int),
            "performance_by_size": defaultdict(list),
            "performance_by_algorithm": defaultdict(list)
        }
        
        for session in self.sessions.values():
            # Operation frequency
            patterns["operation_frequency"][session.operation_type] += 1
            
            # Algorithm frequency
            if session.algorithm:
                patterns["algorithm_frequency"][session.algorithm] += 1
            
            # Device pair frequency
            for pair in session.device_pairs:
                patterns["device_pair_frequency"][pair] += 1
            
            # Tensor size distribution
            for shape in session.tensor_shapes:
                size = np.prod(shape)
                patterns["tensor_size_distribution"].append(size)
                
                # Performance by size
                if session.durations_ms:
                    patterns["performance_by_size"][size].extend(session.durations_ms)
            
            # Performance by algorithm
            if session.algorithm and session.durations_ms:
                patterns["performance_by_algorithm"][session.algorithm].extend(session.durations_ms)
        
        # Calculate statistics
        analysis = {
            "most_common_device_pairs": sorted(
                patterns["device_pair_frequency"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
            "tensor_size_stats": {
                "mean": np.mean(patterns["tensor_size_distribution"]) if patterns["tensor_size_distribution"] else 0,
                "median": np.median(patterns["tensor_size_distribution"]) if patterns["tensor_size_distribution"] else 0,
                "std": np.std(patterns["tensor_size_distribution"]) if patterns["tensor_size_distribution"] else 0
            },
            "operation_distribution": dict(patterns["operation_frequency"]),
            "algorithm_distribution": dict(patterns["algorithm_frequency"])
        }
        
        # Performance analysis by size
        size_performance = {}
        for size, durations in patterns["performance_by_size"].items():
            if durations:
                size_performance[size] = {
                    "avg_duration_ms": np.mean(durations),
                    "std_duration_ms": np.std(durations),
                    "sample_count": len(durations)
                }
        
        analysis["performance_by_size"] = size_performance
        
        # Performance analysis by algorithm
        algorithm_performance = {}
        for algorithm, durations in patterns["performance_by_algorithm"].items():
            if durations:
                algorithm_performance[algorithm] = {
                    "avg_duration_ms": np.mean(durations),
                    "std_duration_ms": np.std(durations),
                    "sample_count": len(durations)
                }
        
        analysis["performance_by_algorithm"] = algorithm_performance
        
        return analysis
    
    def get_session(self, session_id: str) -> Optional[P2PProfileResult]:
        """Get a specific profiling session."""
        return self.sessions.get(session_id)
    
    def get_comparison(self, comparison_id: str) -> Optional[P2PComparisonResult]:
        """Get a specific comparison result."""
        return self.comparisons.get(comparison_id)
    
    def list_sessions(self) -> List[str]:
        """List all session IDs."""
        return list(self.sessions.keys())
    
    def list_comparisons(self) -> List[str]:
        """List all comparison IDs."""
        return list(self.comparisons.keys())
    
    def export_all_results(self, filename: Optional[str] = None) -> str:
        """
        Export all profiling results to a single file.
        
        Args:
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to exported file
        """
        if not filename:
            timestamp = int(time.time())
            filename = f"all_profiling_results_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Prepare export data
        export_data = {
            "export_timestamp": time.time(),
            "sessions": {sid: asdict(session) for sid, session in self.sessions.items()},
            "comparisons": {cid: asdict(comparison) for cid, comparison in self.comparisons.items()},
            "communication_patterns": self.analyze_communication_patterns()
        }
        
        # Write to file
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        log_tp(None, f"All profiling results exported to {filepath}")
        return filepath


# Global profiler instance
_global_profiler: Optional[P2PProfiler] = None


def get_global_profiler() -> Optional[P2PProfiler]:
    """Get the global profiler instance."""
    return _global_profiler


def initialize_global_profiler(
    monitor: Optional[P2PMonitor] = None,
    output_dir: Optional[str] = None
) -> P2PProfiler:
    """
    Initialize the global profiler instance.
    
    Args:
        monitor: P2P monitor instance
        output_dir: Output directory
        
    Returns:
        Global profiler instance
    """
    global _global_profiler
    
    _global_profiler = P2PProfiler(
        monitor=monitor,
        output_dir=output_dir
    )
    
    return _global_profiler


def profile_p2p_operation(
    operation_type: str,
    algorithm: Optional[str] = None,
    num_iterations: int = 10,
    warmup_iterations: int = 3
):
    """
    Decorator for profiling P2P operations.
    
    Args:
        operation_type: Type of operation
        algorithm: Algorithm being used
        num_iterations: Number of profiling iterations
        warmup_iterations: Number of warmup iterations
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            profiler = get_global_profiler()
            if not profiler:
                return func(*args, **kwargs)
            
            # Extract tensor from arguments if possible
            tensor = None
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    tensor = arg
                    break
            
            # Extract device ID from arguments or kwargs
            device_id = kwargs.get("device_id")
            if device_id is None and tensor is not None:
                device_id = tensor.device.index
            
            return profiler.profile_operation(
                operation_func=lambda: func(*args, **kwargs),
                operation_type=operation_type,
                algorithm=algorithm,
                num_iterations=num_iterations,
                warmup_iterations=warmup_iterations,
                tensor=tensor,
                device_id=device_id
            )
        
        return wrapper
    return decorator