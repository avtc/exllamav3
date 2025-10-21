#!/usr/bin/env python3
"""
Compatibility tests for P2P GPU communication implementation.

This module provides comprehensive compatibility testing including:
- Different GPU architectures (Pascal, Volta, Turing, Ampere)
- Various interconnect types (NVLink, PCIe)
- Different model sizes and types
- Quantized and non-quantized models
- Existing backend compatibility
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
import json

# Add the exllamav3 directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from exllamav3.model.model_tp_p2p import P2PTopology
    from exllamav3.model.model_tp_backend import TPBackendP2P, TPBackendNative, TPBackendNCCL
    from exllamav3.util.p2p_monitor import P2PMonitor
    from exllamav3.ext import exllamav3_ext as ext
    P2P_AVAILABLE = True
except ImportError as e:
    print(f"P2P modules not available: {e}")
    P2P_AVAILABLE = False


class P2PCompatibilityTester:
    """Compatibility tester for P2P operations."""
    
    def __init__(self, active_devices: List[int], output_dir: str = "./p2p_compatibility"):
        """
        Initialize compatibility tester.
        
        Args:
            active_devices: List of active GPU device IDs
            output_dir: Directory to store compatibility test results
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
        
        # Compatibility test results
        self.compatibility_results = []
        
        if P2P_AVAILABLE:
            try:
                self.topology = P2PTopology(active_devices)
                self.monitor = P2PMonitor(active_devices=active_devices, output_dir=output_dir)
            except Exception as e:
                print(f"Failed to initialize P2P components: {e}")
    
    def initialize_backends(self, backend_type: str = "p2p"):
        """Initialize backends for testing."""
        if not P2P_AVAILABLE:
            return False
        
        try:
            for device in self.active_devices:
                if backend_type == "p2p":
                    backend = TPBackendP2P(
                        device=device,
                        active_devices=self.active_devices,
                        output_device=self.active_devices[0],
                        init_method="tcp://localhost:12345",
                        master=(device == self.active_devices[0]),
                        uuid="p2p_compatibility_test"
                    )
                elif backend_type == "native":
                    backend = TPBackendNative(
                        device=device,
                        active_devices=self.active_devices,
                        output_device=self.active_devices[0],
                        init_method="tcp://localhost:12345",
                        master=(device == self.active_devices[0]),
                        uuid="native_compatibility_test"
                    )
                elif backend_type == "nccl":
                    backend = TPBackendNCCL(
                        device=device,
                        active_devices=self.active_devices,
                        output_device=self.active_devices[0],
                        init_method="tcp://localhost:12345",
                        master=(device == self.active_devices[0]),
                        uuid="nccl_compatibility_test"
                    )
                else:
                    raise ValueError(f"Unknown backend type: {backend_type}")
                
                self.backends[device] = backend
            return True
        except Exception as e:
            print(f"Failed to initialize {backend_type} backends: {e}")
            return False
    
    def cleanup_backends(self):
        """Clean up backends."""
        for device, backend in self.backends.items():
            try:
                backend.close()
            except Exception:
                pass
        self.backends.clear()
    
    def get_gpu_architecture(self, device_id: int) -> Dict:
        """Get GPU architecture information."""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        try:
            props = torch.cuda.get_device_properties(device_id)
            return {
                "name": props.name,
                "major": props.major,
                "minor": props.minor,
                "total_memory": props.total_memory,
                "multiprocessor_count": props.multiprocessor_count,
                "architecture": self._get_architecture_name(props.major, props.minor)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _get_architecture_name(self, major: int, minor: int) -> str:
        """Get architecture name from version numbers."""
        if major == 6:
            return "Pascal"
        elif major == 7:
            return "Volta"
        elif major == 8:
            return "Ampere"
        elif major == 9:
            return "Hopper"
        else:
            return f"Unknown ({major}.{minor})"
    
    def test_gpu_architecture_compatibility(self) -> Dict:
        """Test compatibility across different GPU architectures."""
        results = {
            "test_name": "gpu_architecture_compatibility",
            "devices": [],
            "compatibility_matrix": {},
            "supported_features": {},
            "performance_impact": {}
        }
        
        # Get architecture info for all devices
        for device in self.active_devices:
            arch_info = self.get_gpu_architecture(device)
            results["devices"].append({
                "device_id": device,
                "architecture": arch_info
            })
        
        # Test P2P compatibility between architectures
        for i, device_i in enumerate(self.active_devices):
            for j, device_j in enumerate(self.active_devices):
                if i != j:
                    try:
                        # Test P2P access
                        if self.topology:
                            can_access = self.topology.can_access_peer(device_i, device_j)
                            results["compatibility_matrix"][f"{device_i}->{device_j}"] = can_access
                        
                        # Test performance impact
                        if device_i in self.backends and device_j in self.backends:
                            backend = self.backends[device_i]
                            if hasattr(backend, 'measure_p2p_bandwidth'):
                                bandwidth = backend.measure_p2p_bandwidth(device_i, device_j, 1, 3)
                                results["performance_impact"][f"{device_i}->{device_j}"] = bandwidth
                    
                    except Exception as e:
                        results["compatibility_matrix"][f"{device_i}->{device_j}"] = f"Error: {e}"
        
        # Check supported features
        for device in self.active_devices:
            arch_info = self.get_gpu_architecture(device)
            if "architecture" in arch_info:
                arch = arch_info["architecture"]
                results["supported_features"][device] = self._get_supported_features(arch)
        
        return results
    
    def _get_supported_features(self, architecture: str) -> Dict:
        """Get supported features for a given architecture."""
        features = {
            "p2p": True,
            "nvlink": False,
            "unified_memory": False,
            "tensor_cores": False,
            "cuda_cooperative_groups": False
        }
        
        if architecture in ["Volta", "Turing", "Ampere", "Hopper"]:
            features["nvlink"] = True
            features["unified_memory"] = True
            features["tensor_cores"] = True
            features["cuda_cooperative_groups"] = True
        elif architecture == "Pascal":
            features["unified_memory"] = True
        
        return features
    
    def test_interconnect_compatibility(self) -> Dict:
        """Test compatibility across different interconnect types."""
        results = {
            "test_name": "interconnect_compatibility",
            "interconnect_types": {},
            "bandwidth_measurements": {},
            "latency_measurements": {},
            "optimization_suggestions": []
        }
        
        # Detect interconnect types
        for device in self.active_devices:
            interconnect_type = self._detect_interconnect_type(device)
            results["interconnect_types"][device] = interconnect_type
        
        # Measure bandwidth and latency
        for i, device_i in enumerate(self.active_devices):
            for j, device_j in enumerate(self.active_devices):
                if i != j and device_i in self.backends:
                    backend = self.backends[device_i]
                    
                    # Measure bandwidth
                    if hasattr(backend, 'measure_p2p_bandwidth'):
                        try:
                            bandwidth = backend.measure_p2p_bandwidth(device_i, device_j, 4, 5)
                            results["bandwidth_measurements"][f"{device_i}->{device_j}"] = bandwidth
                        except Exception as e:
                            results["bandwidth_measurements"][f"{device_i}->{device_j}"] = f"Error: {e}"
                    
                    # Measure latency
                    if hasattr(backend, 'measure_p2p_latency'):
                        try:
                            latency = backend.measure_p2p_latency(device_i, device_j, 4, 10)
                            results["latency_measurements"][f"{device_i}->{device_j}"] = latency
                        except Exception as e:
                            results["latency_measurements"][f"{device_i}->{device_j}"] = f"Error: {e}"
        
        # Generate optimization suggestions
        results["optimization_suggestions"] = self._generate_interconnect_suggestions(results)
        
        return results
    
    def _detect_interconnect_type(self, device_id: int) -> str:
        """Detect interconnect type for a device."""
        # This is a simplified detection - in practice, you'd query the hardware
        arch_info = self.get_gpu_architecture(device_id)
        
        if "architecture" in arch_info:
            arch = arch_info["architecture"]
            if arch in ["Volta", "Turing", "Ampere", "Hopper"]:
                return "NVLink/PCIe"
            else:
                return "PCIe"
        
        return "Unknown"
    
    def _generate_interconnect_suggestions(self, results: Dict) -> List[str]:
        """Generate optimization suggestions based on interconnect type."""
        suggestions = []
        
        interconnect_types = results["interconnect_types"]
        if all(t == "NVLink/PCIe" for t in interconnect_types.values()):
            suggestions.append("All devices support NVLink - use direct P2P for optimal performance")
            suggestions.append("Consider using tree-based reductions for better scalability")
        elif all(t == "PCIe" for t in interconnect_types.values()):
            suggestions.append("PCIe-only interconnect - use ring-based algorithms for better performance")
            suggestions.append("Consider reducing tensor sizes to minimize PCIe overhead")
        else:
            suggestions.append("Mixed interconnect types - use adaptive algorithm selection")
            suggestions.append("Monitor performance and adjust algorithms based on bottlenecks")
        
        return suggestions
    
    def test_model_size_compatibility(self) -> Dict:
        """Test compatibility with different model sizes."""
        results = {
            "test_name": "model_size_compatibility",
            "model_sizes": [],
            "performance_metrics": {},
            "memory_usage": {},
            "scalability_limits": {}
        }
        
        # Test different model sizes (simulated)
        model_sizes = [
            {"name": "small", "params": 1e6, "tensor_size": 1024},
            {"name": "medium", "params": 1e7, "tensor_size": 1024*1024},
            {"name": "large", "params": 1e8, "tensor_size": 4*1024*1024},
            {"name": "xlarge", "params": 1e9, "tensor_size": 16*1024*1024}
        ]
        
        device = self.active_devices[0] if self.active_devices else 0
        
        for model_size in model_sizes:
            size_results = {
                "name": model_size["name"],
                "params": model_size["params"],
                "tensor_size": model_size["tensor_size"],
                "compatible": False,
                "performance": {},
                "memory_usage_mb": 0
            }
            
            try:
                # Create test tensor
                tensor = torch.randn(model_size["tensor_size"], dtype=torch.float32, device=device)
                
                # Test operations
                if device in self.backends:
                    backend = self.backends[device]
                    
                    # Test broadcast
                    start_time = time.time()
                    backend.broadcast(tensor, device)
                    broadcast_time = time.time() - start_time
                    
                    # Test all_reduce
                    test_tensor = tensor.clone()
                    start_time = time.time()
                    backend.all_reduce(test_tensor)
                    all_reduce_time = time.time() - start_time
                    
                    size_results["performance"] = {
                        "broadcast_time": broadcast_time,
                        "all_reduce_time": all_reduce_time,
                        "bandwidth_gbps": (model_size["tensor_size"] * 4) / broadcast_time / (1024**3)
                    }
                    
                    # Check memory usage
                    if torch.cuda.is_available():
                        memory_usage = torch.cuda.memory_allocated(device) / (1024**2)
                        size_results["memory_usage_mb"] = memory_usage
                    
                    size_results["compatible"] = True
                
                results["model_sizes"].append(size_results)
                
            except Exception as e:
                size_results["error"] = str(e)
                results["model_sizes"].append(size_results)
        
        # Analyze scalability limits
        results["scalability_limits"] = self._analyze_scalability_limits(results["model_sizes"])
        
        return results
    
    def _analyze_scalability_limits(self, model_sizes: List[Dict]) -> Dict:
        """Analyze scalability limits based on model size tests."""
        limits = {
            "max_compatible_size": None,
            "memory_bottleneck": False,
            "performance_degradation": False,
            "recommendations": []
        }
        
        compatible_sizes = [s for s in model_sizes if s.get("compatible", False)]
        
        if compatible_sizes:
            max_size = max(compatible_sizes, key=lambda x: x["tensor_size"])
            limits["max_compatible_size"] = {
                "name": max_size["name"],
                "tensor_size": max_size["tensor_size"],
                "params": max_size["params"]
            }
        
        # Check for memory bottlenecks
        memory_usages = [s.get("memory_usage_mb", 0) for s in model_sizes]
        if memory_usages:
            max_memory = max(memory_usages)
            if max_memory > 1024:  # More than 1GB
                limits["memory_bottleneck"] = True
                limits["recommendations"].append("Consider memory optimization for large models")
        
        # Check for performance degradation
        if len(compatible_sizes) > 1:
            performances = [s.get("performance", {}).get("bandwidth_gbps", 0) for s in compatible_sizes]
            if performances and max(performances) > 0:
                min_performance = min(p for p in performances if p > 0)
                max_performance = max(performances)
                if max_performance / min_performance > 2.0:
                    limits["performance_degradation"] = True
                    limits["recommendations"].append("Performance degrades with larger models - consider algorithm optimization")
        
        return limits
    
    def test_quantization_compatibility(self) -> Dict:
        """Test compatibility with quantized models."""
        results = {
            "test_name": "quantization_compatibility",
            "dtypes": [],
            "performance_comparison": {},
            "accuracy_impact": {},
            "memory_savings": {}
        }
        
        device = self.active_devices[0] if self.active_devices else 0
        tensor_size = 1024*1024
        
        # Test different data types
        dtypes = [
            {"name": "float32", "dtype": torch.float32, "bits": 32},
            {"name": "float16", "dtype": torch.float16, "bits": 16},
            {"name": "bfloat16", "dtype": torch.bfloat16, "bits": 16},
            {"name": "int8", "dtype": torch.int8, "bits": 8},
            {"name": "int4", "dtype": torch.int8, "bits": 4}  # Simulated int4
        ]
        
        base_tensor = torch.randn(tensor_size, dtype=torch.float32, device=device)
        
        for dtype_info in dtypes:
            dtype_results = {
                "name": dtype_info["name"],
                "dtype": str(dtype_info["dtype"]),
                "bits": dtype_info["bits"],
                "compatible": False,
                "performance": {},
                "memory_usage_mb": 0,
                "accuracy_loss": 0.0
            }
            
            try:
                # Convert tensor to target dtype
                if dtype_info["name"] == "int4":
                    # Simulate int4 by using int8 and packing
                    quantized_tensor = torch.randint(-8, 8, (tensor_size,), dtype=torch.int8, device=device)
                else:
                    quantized_tensor = base_tensor.to(dtype_info["dtype"])
                
                # Test operations
                if device in self.backends:
                    backend = self.backends[device]
                    
                    # Test broadcast
                    start_time = time.time()
                    backend.broadcast(quantized_tensor, device)
                    broadcast_time = time.time() - start_time
                    
                    # Test all_reduce
                    test_tensor = quantized_tensor.clone()
                    start_time = time.time()
                    backend.all_reduce(test_tensor)
                    all_reduce_time = time.time() - start_time
                    
                    dtype_results["performance"] = {
                        "broadcast_time": broadcast_time,
                        "all_reduce_time": all_reduce_time,
                        "bandwidth_gbps": (tensor_size * dtype_info["bits"] // 8) / broadcast_time / (1024**3)
                    }
                    
                    # Check memory usage
                    if torch.cuda.is_available():
                        memory_usage = torch.cuda.memory_allocated(device) / (1024**2)
                        dtype_results["memory_usage_mb"] = memory_usage
                    
                    # Calculate accuracy loss (simplified)
                    if dtype_info["name"] != "float32":
                        # Convert back to float32 for comparison
                        if dtype_info["name"] == "int4":
                            dequantized = quantized_tensor.float() / 8.0
                        else:
                            dequantized = quantized_tensor.float()
                        
                        mse = torch.mean((base_tensor - dequantized) ** 2).item()
                        dtype_results["accuracy_loss"] = mse
                    
                    dtype_results["compatible"] = True
                
                results["dtypes"].append(dtype_results)
                
            except Exception as e:
                dtype_results["error"] = str(e)
                results["dtypes"].append(dtype_results)
        
        # Generate performance comparison
        results["performance_comparison"] = self._compare_quantization_performance(results["dtypes"])
        
        # Generate memory savings analysis
        results["memory_savings"] = self._analyze_memory_savings(results["dtypes"])
        
        return results
    
    def _compare_quantization_performance(self, dtypes: List[Dict]) -> Dict:
        """Compare performance across different quantization types."""
        comparison = {
            "fastest_dtype": None,
            "slowest_dtype": None,
            "performance_ratio": 0.0,
            "recommendations": []
        }
        
        compatible_dtypes = [d for d in dtypes if d.get("compatible", False)]
        
        if len(compatible_dtypes) > 1:
            performances = [(d["name"], d.get("performance", {}).get("bandwidth_gbps", 0)) for d in compatible_dtypes]
            performances.sort(key=lambda x: x[1], reverse=True)
            
            comparison["fastest_dtype"] = performances[0][0]
            comparison["slowest_dtype"] = performances[-1][0]
            
            if performances[-1][1] > 0:
                comparison["performance_ratio"] = performances[0][1] / performances[-1][1]
            
            # Generate recommendations
            if comparison["performance_ratio"] > 1.5:
                comparison["recommendations"].append(f"Use {comparison['fastest_dtype']} for best performance")
            
            if any(d["bits"] <= 8 for d in compatible_dtypes):
                comparison["recommendations"].append("Consider quantization for memory efficiency")
        
        return comparison
    
    def _analyze_memory_savings(self, dtypes: List[Dict]) -> Dict:
        """Analyze memory savings from quantization."""
        savings = {
            "baseline_mb": 0,
            "max_savings_mb": 0,
            "max_savings_percent": 0.0,
            "recommendations": []
        }
        
        compatible_dtypes = [d for d in dtypes if d.get("compatible", False)]
        
        if compatible_dtypes:
            # Find float32 baseline
            float32_dtype = next((d for d in compatible_dtypes if d["name"] == "float32"), None)
            if float32_dtype:
                savings["baseline_mb"] = float32_dtype.get("memory_usage_mb", 0)
                
                # Find maximum savings
                max_savings = 0
                for dtype in compatible_dtypes:
                    if dtype["name"] != "float32":
                        memory_diff = float32_dtype.get("memory_usage_mb", 0) - dtype.get("memory_usage_mb", 0)
                        if memory_diff > max_savings:
                            max_savings = memory_diff
                
                savings["max_savings_mb"] = max_savings
                
                if savings["baseline_mb"] > 0:
                    savings["max_savings_percent"] = (max_savings / savings["baseline_mb"]) * 100
                
                # Generate recommendations
                if savings["max_savings_percent"] > 25:
                    savings["recommendations"].append("Quantization provides significant memory savings")
        
        return savings
    
    def test_backend_compatibility(self) -> Dict:
        """Test compatibility across different backends."""
        results = {
            "test_name": "backend_compatibility",
            "backends": {},
            "performance_comparison": {},
            "feature_support": {},
            "recommendations": []
        }
        
        backend_types = ["p2p", "native", "nccl"]
        device = self.active_devices[0] if self.active_devices else 0
        tensor_size = 1024*1024
        
        for backend_type in backend_types:
            try:
                # Initialize backend
                self.cleanup_backends()
                if self.initialize_backends(backend_type):
                    backend_results = {
                        "type": backend_type,
                        "compatible": False,
                        "performance": {},
                        "features": []
                    }
                    
                    # Test operations
                    if device in self.backends:
                        backend = self.backends[device]
                        tensor = torch.randn(tensor_size, dtype=torch.float32, device=device)
                        
                        # Test broadcast
                        start_time = time.time()
                        backend.broadcast(tensor, device)
                        broadcast_time = time.time() - start_time
                        
                        # Test all_reduce
                        test_tensor = tensor.clone()
                        start_time = time.time()
                        backend.all_reduce(test_tensor)
                        all_reduce_time = time.time() - start_time
                        
                        backend_results["performance"] = {
                            "broadcast_time": broadcast_time,
                            "all_reduce_time": all_reduce_time,
                            "bandwidth_gbps": (tensor_size * 4) / broadcast_time / (1024**3)
                        }
                        
                        # Check features
                        if hasattr(backend, 'use_p2p'):
                            if backend.use_p2p:
                                backend_results["features"].append("p2p")
                        
                        if hasattr(backend, 'copy_tensor_direct'):
                            backend_results["features"].append("direct_copy")
                        
                        backend_results["compatible"] = True
                    
                    results["backends"][backend_type] = backend_results
                
            except Exception as e:
                results["backends"][backend_type] = {
                    "type": backend_type,
                    "compatible": False,
                    "error": str(e)
                }
        
        # Generate performance comparison
        results["performance_comparison"] = self._compare_backend_performance(results["backends"])
        
        # Generate feature support analysis
        results["feature_support"] = self._analyze_backend_features(results["backends"])
        
        # Generate recommendations
        results["recommendations"] = self._generate_backend_recommendations(results)
        
        return results
    
    def _compare_backend_performance(self, backends: Dict) -> Dict:
        """Compare performance across different backends."""
        comparison = {
            "fastest_backend": None,
            "slowest_backend": None,
            "performance_ratio": 0.0,
            "detailed_comparison": {}
        }
        
        compatible_backends = {k: v for k, v in backends.items() if v.get("compatible", False)}
        
        if len(compatible_backends) > 1:
            performances = {}
            for backend_type, backend_info in compatible_backends.items():
                bandwidth = backend_info.get("performance", {}).get("bandwidth_gbps", 0)
                performances[backend_type] = bandwidth
            
            if performances:
                sorted_backends = sorted(performances.items(), key=lambda x: x[1], reverse=True)
                comparison["fastest_backend"] = sorted_backends[0][0]
                comparison["slowest_backend"] = sorted_backends[-1][0]
                
                if sorted_backends[-1][1] > 0:
                    comparison["performance_ratio"] = sorted_backends[0][1] / sorted_backends[-1][1]
                
                comparison["detailed_comparison"] = performances
        
        return comparison
    
    def _analyze_backend_features(self, backends: Dict) -> Dict:
        """Analyze feature support across backends."""
        feature_analysis = {
            "all_features": set(),
            "common_features": set(),
            "unique_features": {}
        }
        
        compatible_backends = {k: v for k, v in backends.items() if v.get("compatible", False)}
        
        # Collect all features
        for backend_type, backend_info in compatible_backends.items():
            features = backend_info.get("features", [])
            feature_analysis["all_features"].update(features)
            
            if not feature_analysis["common_features"]:
                feature_analysis["common_features"] = set(features)
            else:
                feature_analysis["common_features"] &= set(features)
        
        # Find unique features
        for backend_type, backend_info in compatible_backends.items():
            features = set(backend_info.get("features", []))
            unique = features - feature_analysis["common_features"]
            if unique:
                feature_analysis["unique_features"][backend_type] = list(unique)
        
        # Convert sets to lists for JSON serialization
        feature_analysis["all_features"] = list(feature_analysis["all_features"])
        feature_analysis["common_features"] = list(feature_analysis["common_features"])
        
        return feature_analysis
    
    def _generate_backend_recommendations(self, results: Dict) -> List[str]:
        """Generate backend recommendations."""
        recommendations = []
        
        backends = results["backends"]
        performance_comparison = results["performance_comparison"]
        feature_support = results["feature_support"]
        
        # Performance-based recommendations
        if performance_comparison.get("fastest_backend"):
            fastest = performance_comparison["fastest_backend"]
            recommendations.append(f"Use {fastest} backend for best performance")
        
        if performance_comparison.get("performance_ratio", 0) > 1.5:
            recommendations.append("Significant performance difference between backends - choose based on workload")
        
        # Feature-based recommendations
        common_features = feature_support.get("common_features", [])
        if "p2p" in common_features:
            recommendations.append("All backends support P2P - can use direct GPU communication")
        
        unique_features = feature_support.get("unique_features", {})
        if unique_features:
            for backend, features in unique_features.items():
                recommendations.append(f"{backend} backend offers unique features: {', '.join(features)}")
        
        # Compatibility-based recommendations
        compatible_count = sum(1 for b in backends.values() if b.get("compatible", False))
        if compatible_count == 1:
            recommendations.append("Only one backend compatible - use fallback mechanisms")
        elif compatible_count == len(backends):
            recommendations.append("All backends compatible - can choose based on performance requirements")
        
        return recommendations
    
    def generate_compatibility_report(self) -> str:
        """Generate a comprehensive compatibility report."""
        report_path = os.path.join(self.output_dir, "compatibility_report.json")
        
        report_data = {
            "timestamp": time.time(),
            "num_devices": self.num_devices,
            "active_devices": self.active_devices,
            "compatibility_results": self.compatibility_results,
            "summary": {
                "total_tests": len(self.compatibility_results),
                "passed_tests": sum(1 for r in self.compatibility_results if r.get("compatible", True)),
                "failed_tests": sum(1 for r in self.compatibility_results if not r.get("compatible", True))
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        return report_path


class TestP2PCompatibility(unittest.TestCase):
    """Test cases for P2P compatibility testing."""
    
    def setUp(self):
        """Set up test environment."""
        if not P2P_AVAILABLE:
            self.skipTest("P2P modules not available")
        
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        if torch.cuda.device_count() < 2:
            self.skipTest("Need at least 2 GPUs for P2P compatibility testing")
        
        self.active_devices = list(range(min(4, torch.cuda.device_count())))
        self.temp_dir = tempfile.mkdtemp()
        
        self.compatibility_tester = P2PCompatibilityTester(
            active_devices=self.active_devices,
            output_dir=self.temp_dir
        )
        
        # Initialize backends
        if not self.compatibility_tester.initialize_backends("p2p"):
            self.skipTest("Failed to initialize P2P backends")
    
    def tearDown(self):
        """Clean up test environment."""
        self.compatibility_tester.cleanup_backends()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_gpu_architecture_compatibility(self):
        """Test GPU architecture compatibility."""
        result = self.compatibility_tester.test_gpu_architecture_compatibility()
        
        self.assertIsInstance(result, dict)
        self.assertIn("test_name", result)
        self.assertEqual(result["test_name"], "gpu_architecture_compatibility")
        self.assertIn("devices", result)
        self.assertIn("compatibility_matrix", result)
        self.assertIn("supported_features", result)
        self.assertIn("performance_impact", result)
        
        # Check device info
        self.assertEqual(len(result["devices"]), len(self.active_devices))
        for device_info in result["devices"]:
            self.assertIn("device_id", device_info)
            self.assertIn("architecture", device_info)
        
        # Store result for report
        self.compatibility_tester.compatibility_results.append(result)
    
    def test_interconnect_compatibility(self):
        """Test interconnect compatibility."""
        result = self.compatibility_tester.test_interconnect_compatibility()
        
        self.assertIsInstance(result, dict)
        self.assertIn("test_name", result)
        self.assertEqual(result["test_name"], "interconnect_compatibility")
        self.assertIn("interconnect_types", result)
        self.assertIn("bandwidth_measurements", result)
        self.assertIn("latency_measurements", result)
        self.assertIn("optimization_suggestions", result)
        
        # Check interconnect types
        self.assertEqual(len(result["interconnect_types"]), len(self.active_devices))
        
        # Check optimization suggestions
        self.assertIsInstance(result["optimization_suggestions"], list)
        
        # Store result for report
        self.compatibility_tester.compatibility_results.append(result)
    
    def test_model_size_compatibility(self):
        """Test model size compatibility."""
        result = self.compatibility_tester.test_model_size_compatibility()
        
        self.assertIsInstance(result, dict)
        self.assertIn("test_name", result)
        self.assertEqual(result["test_name"], "model_size_compatibility")
        self.assertIn("model_sizes", result)
        self.assertIn("performance_metrics", result)
        self.assertIn("memory_usage", result)
        self.assertIn("scalability_limits", result)
        
        # Check model sizes
        self.assertGreater(len(result["model_sizes"]), 0)
        
        # Check scalability limits
        self.assertIsInstance(result["scalability_limits"], dict)
        self.assertIn("recommendations", result["scalability_limits"])
        
        # Store result for report
        self.compatibility_tester.compatibility_results.append(result)
    
    def test_quantization_compatibility(self):
        """Test quantization compatibility."""
        result = self.compatibility_tester.test_quantization_compatibility()
        
        self.assertIsInstance(result, dict)
        self.assertIn("test_name", result)
        self.assertEqual(result["test_name"], "quantization_compatibility")
        self.assertIn("dtypes", result)
        self.assertIn("performance_comparison", result)
        self.assertIn("accuracy_impact", result)
        self.assertIn("memory_savings", result)
        
        # Check data types
        self.assertGreater(len(result["dtypes"]), 0)
        
        # Check performance comparison
        self.assertIsInstance(result["performance_comparison"], dict)
        
        # Check memory savings
        self.assertIsInstance(result["memory_savings"], dict)
        
        # Store result for report
        self.compatibility_tester.compatibility_results.append(result)
    
    def test_backend_compatibility(self):
        """Test backend compatibility."""
        result = self.compatibility_tester.test_backend_compatibility()
        
        self.assertIsInstance(result, dict)
        self.assertIn("test_name", result)
        self.assertEqual(result["test_name"], "backend_compatibility")
        self.assertIn("backends", result)
        self.assertIn("performance_comparison", result)
        self.assertIn("feature_support", result)
        self.assertIn("recommendations", result)
        
        # Check backends
        self.assertIn("p2p", result["backends"])
        self.assertIn("native", result["backends"])
        
        # Check recommendations
        self.assertIsInstance(result["recommendations"], list)
        
        # Store result for report
        self.compatibility_tester.compatibility_results.append(result)
    
    def test_compatibility_report_generation(self):
        """Test compatibility report generation."""
        # Run some compatibility tests
        self.compatibility_tester.test_gpu_architecture_compatibility()
        self.compatibility_tester.test_interconnect_compatibility()
        self.compatibility_tester.test_model_size_compatibility()
        
        # Generate report
        report_path = self.compatibility_tester.generate_compatibility_report()
        
        self.assertTrue(os.path.exists(report_path))
        
        # Check report content
        with open(report_path, 'r') as f:
            report_data = json.load(f)
        
        self.assertIn("timestamp", report_data)
        self.assertIn("num_devices", report_data)
        self.assertIn("active_devices", report_data)
        self.assertIn("compatibility_results", report_data)
        self.assertIn("summary", report_data)
        
        # Check that we have results
        self.assertGreater(len(report_data["compatibility_results"]), 0)
        
        # Check summary
        summary = report_data["summary"]
        self.assertIn("total_tests", summary)
        self.assertIn("passed_tests", summary)
        self.assertIn("failed_tests", summary)
        
        self.assertGreater(summary["total_tests"], 0)


def run_compatibility_tests():
    """Run all compatibility tests and return results."""
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestP2PCompatibility)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    print("Running P2P Compatibility Tests")
    print("=" * 50)
    
    result = run_compatibility_tests()
    
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All compatibility tests passed!")
        exit(0)
    else:
        print(f"❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        exit(1)