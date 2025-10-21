"""
P2P Debugging Utilities for ExLlamaV3

This module provides comprehensive debugging and diagnostic tools for P2P GPU communication,
including error tracking, communication flow tracing, and diagnostic information.
"""

import time
import threading
import json
import os
import traceback
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import numpy as np
import torch

from .debug import log_tp
from .p2p_monitor import P2PMonitor, get_global_monitor
from ..model.model_tp_p2p import P2PTopology


@dataclass
class P2PDebugEvent:
    """Debug event for P2P operations."""
    timestamp: float
    event_type: str  # "operation_start", "operation_end", "error", "warning"
    operation_type: str  # "broadcast", "all_reduce", "gather", "direct_copy"
    device_id: int
    peer_device: Optional[int]
    tensor_info: Dict[str, Any]
    algorithm: Optional[str]
    message: str
    details: Optional[Dict[str, Any]] = None
    stack_trace: Optional[str] = None


@dataclass
class P2PErrorInfo:
    """Information about a P2P error."""
    timestamp: float
    error_type: str
    operation_type: str
    device_id: int
    peer_device: Optional[int]
    error_message: str
    stack_trace: str
    context: Dict[str, Any]
    recovery_attempted: bool
    recovery_successful: bool


@dataclass
class P2PCommunicationTrace:
    """Trace of P2P communication flow."""
    trace_id: str
    timestamp: float
    operation_type: str
    participants: List[int]
    communication_steps: List[Dict[str, Any]]
    total_duration_ms: float
    success: bool
    bottlenecks: List[str]


class P2PDebugger:
    """
    Comprehensive P2P debugging system.
    
    This class provides debugging and diagnostic capabilities for P2P operations,
    including error tracking, communication flow tracing, and diagnostic information.
    """
    
    def __init__(
        self,
        monitor: Optional[P2PMonitor] = None,
        debug_level: str = "basic",
        max_events: int = 10000,
        output_dir: Optional[str] = None
    ):
        """
        Initialize P2P debugger.
        
        Args:
            monitor: P2P monitor instance
            debug_level: Level of debugging ("basic", "detailed", "verbose")
            max_events: Maximum number of debug events to keep
            output_dir: Directory to save debug reports
        """
        self.monitor = monitor or get_global_monitor()
        self.debug_level = debug_level
        self.max_events = max_events
        self.output_dir = output_dir or "./p2p_debug"
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Debug events
        self.events: deque = deque(maxlen=max_events)
        self.events_lock = threading.Lock()
        
        # Error tracking
        self.errors: List[P2PErrorInfo] = []
        self.errors_lock = threading.Lock()
        
        # Communication traces
        self.traces: Dict[str, P2PCommunicationTrace] = {}
        self.traces_lock = threading.Lock()
        
        # Active traces
        self.active_traces: Dict[str, Dict[str, Any]] = {}
        self.active_traces_lock = threading.Lock()
        
        # Debug hooks
        self.debug_hooks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Statistics
        self.stats = {
            "total_events": 0,
            "total_errors": 0,
            "operation_counts": defaultdict(int),
            "error_counts": defaultdict(int),
            "device_errors": defaultdict(int)
        }
        self.stats_lock = threading.Lock()
    
    def log_event(
        self,
        event_type: str,
        operation_type: str,
        device_id: int,
        peer_device: Optional[int] = None,
        tensor: Optional[torch.Tensor] = None,
        algorithm: Optional[str] = None,
        message: str = "",
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Log a debug event.
        
        Args:
            event_type: Type of event
            operation_type: Type of operation
            device_id: Device ID
            peer_device: Peer device ID
            tensor: Tensor involved (if any)
            algorithm: Algorithm used (if any)
            message: Event message
            details: Additional details
        """
        # Skip events below debug level
        if self.debug_level == "basic" and event_type not in ["error", "warning"]:
            return
        
        # Prepare tensor info
        tensor_info = {}
        if tensor is not None:
            tensor_info = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "numel": tensor.numel(),
                "element_size": tensor.element_size(),
                "size_bytes": tensor.numel() * tensor.element_size()
            }
        
        # Create event
        event = P2PDebugEvent(
            timestamp=time.time(),
            event_type=event_type,
            operation_type=operation_type,
            device_id=device_id,
            peer_device=peer_device,
            tensor_info=tensor_info,
            algorithm=algorithm,
            message=message,
            details=details
        )
        
        # Add stack trace for errors
        if event_type == "error":
            event.stack_trace = traceback.format_exc()
        
        # Add to events
        with self.events_lock:
            self.events.append(event)
        
        # Update statistics
        with self.stats_lock:
            self.stats["total_events"] += 1
            self.stats["operation_counts"][operation_type] += 1
            if event_type == "error":
                self.stats["total_errors"] += 1
                self.stats["error_counts"][operation_type] += 1
                self.stats["device_errors"][device_id] += 1
        
        # Call debug hooks
        self._call_debug_hooks(event_type, event)
        
        # Log to standard debug
        log_tp(device_id, f"P2P Debug [{event_type}] {operation_type}: {message}")
    
    def log_error(
        self,
        error_type: str,
        operation_type: str,
        device_id: int,
        peer_device: Optional[int] = None,
        error_message: str = "",
        context: Optional[Dict[str, Any]] = None,
        recovery_attempted: bool = False,
        recovery_successful: bool = False
    ):
        """
        Log a P2P error.
        
        Args:
            error_type: Type of error
            operation_type: Type of operation
            device_id: Device ID
            peer_device: Peer device ID
            error_message: Error message
            context: Error context
            recovery_attempted: Whether recovery was attempted
            recovery_successful: Whether recovery was successful
        """
        error_info = P2PErrorInfo(
            timestamp=time.time(),
            error_type=error_type,
            operation_type=operation_type,
            device_id=device_id,
            peer_device=peer_device,
            error_message=error_message,
            stack_trace=traceback.format_exc(),
            context=context or {},
            recovery_attempted=recovery_attempted,
            recovery_successful=recovery_successful
        )
        
        with self.errors_lock:
            self.errors.append(error_info)
        
        # Log as event
        self.log_event(
            event_type="error",
            operation_type=operation_type,
            device_id=device_id,
            peer_device=peer_device,
            message=f"{error_type}: {error_message}",
            details={
                "error_type": error_type,
                "recovery_attempted": recovery_attempted,
                "recovery_successful": recovery_successful,
                "context": context
            }
        )
    
    def start_trace(
        self,
        trace_id: str,
        operation_type: str,
        participants: List[int]
    ) -> str:
        """
        Start a communication trace.
        
        Args:
            trace_id: Unique trace identifier
            operation_type: Type of operation
            participants: List of participating devices
            
        Returns:
            Trace ID
        """
        with self.active_traces_lock:
            self.active_traces[trace_id] = {
                "trace_id": trace_id,
                "operation_type": operation_type,
                "participants": participants,
                "start_time": time.time(),
                "steps": [],
                "active": True
            }
        
        self.log_event(
            event_type="trace_start",
            operation_type=operation_type,
            device_id=participants[0] if participants else 0,
            message=f"Started trace {trace_id} for {operation_type}",
            details={"participants": participants}
        )
        
        return trace_id
    
    def add_trace_step(
        self,
        trace_id: str,
        step_type: str,
        device_id: int,
        peer_device: Optional[int] = None,
        message: str = "",
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Add a step to an active trace.
        
        Args:
            trace_id: Trace identifier
            step_type: Type of step
            device_id: Device ID
            peer_device: Peer device ID
            message: Step message
            details: Step details
        """
        with self.active_traces_lock:
            if trace_id not in self.active_traces:
                return
            
            trace = self.active_traces[trace_id]
            if not trace["active"]:
                return
            
            step = {
                "timestamp": time.time(),
                "step_type": step_type,
                "device_id": device_id,
                "peer_device": peer_device,
                "message": message,
                "details": details or {}
            }
            
            trace["steps"].append(step)
        
        self.log_event(
            event_type="trace_step",
            operation_type=trace["operation_type"] if trace_id in self.active_traces else "unknown",
            device_id=device_id,
            peer_device=peer_device,
            message=f"Trace {trace_id} step: {step_type} - {message}",
            details={"trace_id": trace_id, "step_type": step_type}
        )
    
    def end_trace(
        self,
        trace_id: str,
        success: bool = True,
        bottlenecks: Optional[List[str]] = None
    ):
        """
        End a communication trace.
        
        Args:
            trace_id: Trace identifier
            success: Whether trace was successful
            bottlenecks: List of bottlenecks identified
        """
        with self.active_traces_lock:
            if trace_id not in self.active_traces:
                return
            
            trace = self.active_traces[trace_id]
            trace["active"] = False
            
            end_time = time.time()
            total_duration_ms = (end_time - trace["start_time"]) * 1000.0
            
            # Create communication trace
            comm_trace = P2PCommunicationTrace(
                trace_id=trace_id,
                timestamp=trace["start_time"],
                operation_type=trace["operation_type"],
                participants=trace["participants"],
                communication_steps=trace["steps"],
                total_duration_ms=total_duration_ms,
                success=success,
                bottlenecks=bottlenecks or []
            )
            
            with self.traces_lock:
                self.traces[trace_id] = comm_trace
            
            # Remove from active traces
            del self.active_traces[trace_id]
        
        self.log_event(
            event_type="trace_end",
            operation_type=trace["operation_type"],
            device_id=trace["participants"][0] if trace["participants"] else 0,
            message=f"Ended trace {trace_id}: {'success' if success else 'failed'}",
            details={
                "trace_id": trace_id,
                "success": success,
                "duration_ms": total_duration_ms,
                "bottlenecks": bottlenecks or []
            }
        )
    
    def add_debug_hook(
        self,
        event_type: str,
        hook_func: Callable[[P2PDebugEvent], None]
    ):
        """
        Add a debug hook for specific event types.
        
        Args:
            event_type: Event type to hook
            hook_func: Hook function
        """
        self.debug_hooks[event_type].append(hook_func)
    
    def _call_debug_hooks(self, event_type: str, event: P2PDebugEvent):
        """Call debug hooks for an event."""
        for hook_func in self.debug_hooks.get(event_type, []):
            try:
                hook_func(event)
            except Exception as e:
                log_tp(event.device_id, f"Debug hook failed: {e}")
    
    def get_events(
        self,
        event_type: Optional[str] = None,
        operation_type: Optional[str] = None,
        device_id: Optional[int] = None,
        limit: Optional[int] = None,
        since: Optional[float] = None
    ) -> List[P2PDebugEvent]:
        """
        Get debug events with optional filtering.
        
        Args:
            event_type: Filter by event type
            operation_type: Filter by operation type
            device_id: Filter by device ID
            limit: Maximum number of events
            since: Get events since timestamp
            
        Returns:
            List of debug events
        """
        with self.events_lock:
            events = list(self.events)
        
        # Apply filters
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if operation_type:
            events = [e for e in events if e.operation_type == operation_type]
        
        if device_id is not None:
            events = [e for e in events if e.device_id == device_id]
        
        if since:
            events = [e for e in events if e.timestamp >= since]
        
        # Sort by timestamp (newest first)
        events.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Apply limit
        if limit:
            events = events[:limit]
        
        return events
    
    def get_errors(
        self,
        error_type: Optional[str] = None,
        operation_type: Optional[str] = None,
        device_id: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List[P2PErrorInfo]:
        """
        Get error information with optional filtering.
        
        Args:
            error_type: Filter by error type
            operation_type: Filter by operation type
            device_id: Filter by device ID
            limit: Maximum number of errors
            
        Returns:
            List of error information
        """
        with self.errors_lock:
            errors = self.errors.copy()
        
        # Apply filters
        if error_type:
            errors = [e for e in errors if e.error_type == error_type]
        
        if operation_type:
            errors = [e for e in errors if e.operation_type == operation_type]
        
        if device_id is not None:
            errors = [e for e in errors if e.device_id == device_id]
        
        # Sort by timestamp (newest first)
        errors.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Apply limit
        if limit:
            errors = errors[:limit]
        
        return errors
    
    def get_trace(self, trace_id: str) -> Optional[P2PCommunicationTrace]:
        """Get a specific communication trace."""
        with self.traces_lock:
            return self.traces.get(trace_id)
    
    def get_all_traces(self) -> Dict[str, P2PCommunicationTrace]:
        """Get all communication traces."""
        with self.traces_lock:
            return self.traces.copy()
    
    def get_active_traces(self) -> Dict[str, Dict[str, Any]]:
        """Get all active traces."""
        with self.active_traces_lock:
            return self.active_traces.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get debugging statistics."""
        with self.stats_lock:
            stats = self.stats.copy()
        
        # Add recent error information
        recent_errors = self.get_errors(limit=10)
        stats["recent_errors"] = [
            {
                "timestamp": e.timestamp,
                "error_type": e.error_type,
                "operation_type": e.operation_type,
                "device_id": e.device_id,
                "error_message": e.error_message
            }
            for e in recent_errors
        ]
        
        # Add recent event information
        recent_events = self.get_events(limit=10)
        stats["recent_events"] = [
            {
                "timestamp": e.timestamp,
                "event_type": e.event_type,
                "operation_type": e.operation_type,
                "device_id": e.device_id,
                "message": e.message
            }
            for e in recent_events
        ]
        
        return stats
    
    def diagnose_issues(self) -> Dict[str, Any]:
        """
        Diagnose P2P issues based on debug information.
        
        Returns:
            Dictionary with diagnosis results
        """
        diagnosis = {
            "issues": [],
            "recommendations": [],
            "critical_errors": [],
            "performance_issues": [],
            "reliability_issues": []
        }
        
        # Analyze errors
        errors = self.get_errors()
        error_counts = defaultdict(int)
        device_error_counts = defaultdict(int)
        operation_error_counts = defaultdict(int)
        
        for error in errors:
            error_counts[error.error_type] += 1
            device_error_counts[error.device_id] += 1
            operation_error_counts[error.operation_type] += 1
            
            # Check for critical errors
            if error.error_type in ["memory_error", "device_error", "timeout"]:
                diagnosis["critical_errors"].append({
                    "timestamp": error.timestamp,
                    "error_type": error.error_type,
                    "device_id": error.device_id,
                    "operation_type": error.operation_type,
                    "error_message": error.error_message,
                    "recovery_successful": error.recovery_successful
                })
        
        # Check for high error rates
        total_operations = sum(self.stats["operation_counts"].values())
        if total_operations > 0:
            error_rate = self.stats["total_errors"] / total_operations
            if error_rate > 0.1:  # More than 10% error rate
                diagnosis["reliability_issues"].append({
                    "issue": "high_error_rate",
                    "error_rate": error_rate,
                    "description": f"High error rate detected: {error_rate:.2%}"
                })
        
        # Check for device-specific issues
        for device_id, error_count in device_error_counts.items():
            if error_count > 5:  # More than 5 errors on a device
                diagnosis["issues"].append({
                    "issue": "device_errors",
                    "device_id": device_id,
                    "error_count": error_count,
                    "description": f"Device {device_id} has {error_count} errors"
                })
        
        # Check for operation-specific issues
        for operation_type, error_count in operation_error_counts.items():
            if error_count > 3:  # More than 3 errors for an operation
                diagnosis["issues"].append({
                    "issue": "operation_errors",
                    "operation_type": operation_type,
                    "error_count": error_count,
                    "description": f"Operation {operation_type} has {error_count} errors"
                })
        
        # Analyze traces for performance issues
        traces = self.get_all_traces()
        slow_traces = [t for t in traces.values() if t.total_duration_ms > 1000.0]  # > 1 second
        
        if slow_traces:
            diagnosis["performance_issues"].append({
                "issue": "slow_operations",
                "count": len(slow_traces),
                "avg_duration_ms": np.mean([t.total_duration_ms for t in slow_traces]),
                "description": f"Found {len(slow_traces)} slow operations (> 1s)"
            })
        
        # Generate recommendations
        if diagnosis["critical_errors"]:
            diagnosis["recommendations"].append("Address critical errors immediately - check device status and memory")
        
        if diagnosis["reliability_issues"]:
            diagnosis["recommendations"].append("Improve reliability - investigate error patterns and implement better error handling")
        
        if diagnosis["performance_issues"]:
            diagnosis["recommendations"].append("Optimize performance - analyze slow operations and consider algorithm changes")
        
        if diagnosis["issues"]:
            diagnosis["recommendations"].append("Review device and operation-specific issues for targeted improvements")
        
        return diagnosis
    
    def export_debug_report(self, filename: Optional[str] = None) -> str:
        """
        Export comprehensive debug report.
        
        Args:
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to exported file
        """
        if not filename:
            timestamp = int(time.time())
            filename = f"p2p_debug_report_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Prepare report data
        report = {
            "export_timestamp": time.time(),
            "debug_level": self.debug_level,
            "statistics": self.get_statistics(),
            "diagnosis": self.diagnose_issues(),
            "recent_events": [asdict(e) for e in self.get_events(limit=100)],
            "recent_errors": [asdict(e) for e in self.get_errors(limit=50)],
            "traces": {tid: asdict(trace) for tid, trace in self.get_all_traces().items()},
            "active_traces": self.get_active_traces()
        }
        
        # Write to file
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        log_tp(None, f"Debug report exported to {filepath}")
        return filepath
    
    def clear_events(self, older_than: Optional[float] = None):
        """
        Clear debug events.
        
        Args:
            older_than: Clear events older than timestamp (None for all)
        """
        with self.events_lock:
            if older_than is None:
                self.events.clear()
            else:
                self.events = deque(
                    [e for e in self.events if e.timestamp >= older_than],
                    maxlen=self.max_events
                )
    
    def clear_errors(self, older_than: Optional[float] = None):
        """
        Clear error information.
        
        Args:
            older_than: Clear errors older than timestamp (None for all)
        """
        with self.errors_lock:
            if older_than is None:
                self.errors.clear()
            else:
                self.errors = [e for e in self.errors if e.timestamp >= older_than]
    
    def clear_traces(self):
        """Clear all traces."""
        with self.traces_lock:
            self.traces.clear()
        
        with self.active_traces_lock:
            self.active_traces.clear()
    
    def reset_statistics(self):
        """Reset debugging statistics."""
        with self.stats_lock:
            self.stats = {
                "total_events": 0,
                "total_errors": 0,
                "operation_counts": defaultdict(int),
                "error_counts": defaultdict(int),
                "device_errors": defaultdict(int)
            }


# Global debugger instance
_global_debugger: Optional[P2PDebugger] = None


def get_global_debugger() -> Optional[P2PDebugger]:
    """Get the global debugger instance."""
    return _global_debugger


def initialize_global_debugger(
    monitor: Optional[P2PMonitor] = None,
    debug_level: str = "basic",
    max_events: int = 10000,
    output_dir: Optional[str] = None
) -> P2PDebugger:
    """
    Initialize the global debugger instance.
    
    Args:
        monitor: P2P monitor instance
        debug_level: Level of debugging
        max_events: Maximum events to keep
        output_dir: Output directory
        
    Returns:
        Global debugger instance
    """
    global _global_debugger
    
    _global_debugger = P2PDebugger(
        monitor=monitor,
        debug_level=debug_level,
        max_events=max_events,
        output_dir=output_dir
    )
    
    return _global_debugger


def debug_p2p_operation(
    operation_type: str,
    trace_id: Optional[str] = None,
    participants: Optional[List[int]] = None
):
    """
    Decorator for debugging P2P operations.
    
    Args:
        operation_type: Type of operation
        trace_id: Trace ID (auto-generated if None)
        participants: List of participating devices
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            debugger = get_global_debugger()
            if not debugger:
                return func(*args, **kwargs)
            
            # Generate trace ID if not provided
            current_trace_id = trace_id
            if not current_trace_id:
                current_trace_id = f"trace_{operation_type}_{int(time.time())}"
            
            # Extract participants from arguments or kwargs
            current_participants = participants
            if not current_participants:
                # Try to extract from kwargs
                device_id = kwargs.get("device_id")
                peer_device = kwargs.get("peer_device")
                if device_id is not None:
                    current_participants = [device_id]
                    if peer_device is not None:
                        current_participants.append(peer_device)
            
            # Start trace
            if current_participants:
                debugger.start_trace(current_trace_id, operation_type, current_participants)
            
            try:
                # Log operation start
                debugger.log_event(
                    event_type="operation_start",
                    operation_type=operation_type,
                    device_id=current_participants[0] if current_participants else 0,
                    message=f"Starting {operation_type} operation"
                )
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Log operation end
                debugger.log_event(
                    event_type="operation_end",
                    operation_type=operation_type,
                    device_id=current_participants[0] if current_participants else 0,
                    message=f"Completed {operation_type} operation"
                )
                
                return result
            
            except Exception as e:
                # Log error
                debugger.log_error(
                    error_type="operation_error",
                    operation_type=operation_type,
                    device_id=current_participants[0] if current_participants else 0,
                    error_message=str(e),
                    context={"args": args, "kwargs": kwargs}
                )
                
                # End trace with failure
                if current_participants:
                    debugger.end_trace(current_trace_id, success=False)
                
                raise
            
            finally:
                # End trace if started
                if current_participants:
                    debugger.end_trace(current_trace_id, success=True)
        
        return wrapper
    return decorator