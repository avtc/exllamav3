#!/usr/bin/env python3
"""
P2P Implementation Validation Tool

This tool provides comprehensive validation of the P2P GPU communication implementation,
including correctness checks, performance validation, and integration testing.
"""

import os
import sys
import time
import json
import argparse
import tempfile
import shutil
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

# Add the exllamav3 directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import torch
    from exllamav3.model.model_tp_p2p import P2PTopology
    from exllamav3.model.model_tp_backend import TPBackendP2P, TPBackendNative
    from exllamav3.util.p2p_monitor import P2PMonitor
    from exllamav3.util.p2p_profiler import P2PProfiler
    from exllamav3.util.p2p_debug import P2PDebugger
    from exllamav3.util.p2p_config import P2PConfigManager
    from exllamav3.ext import exllamav3_ext as ext
    P2P_AVAILABLE = True
except ImportError as e:
    print(f"P2P modules not available: {e}")
    P2P_AVAILABLE = False


class P2PImplementationValidator:
    """Comprehensive validator for P2P implementation."""
    
    def __init__(self, active_devices: List[int] = None, output_dir: str = "./p2p_validation"):
        """
        Initialize the validator.
        
        Args:
            active_devices: List of active GPU device IDs
            output_dir: Directory to store validation results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine active devices
        if active_devices is None:
            if torch.cuda.is_available():
                self.active_devices = list(range(torch.cuda.device_count()))
            else:
                self.active_devices = []
        else:
            self.active_devices = active_devices
        
        self.num_devices = len(self.active_devices)
        
        # Validation results
        self.validation_results = {
            "timestamp": time.time(),
            "num_devices": self.num_devices,
            "active_devices": self.active_devices,
            "validation_categories": {},
            "overall_status": "unknown",
            "summary": {}
        }
        
        # Initialize components
        self.topology = None
        self.backends = {}
        self.monitor = None
        self.profiler = None
        self.debugger = None
        
        if P2P_AVAILABLE and self.num_devices > 0:
            try:
                self.topology = P2PTopology(self.active_devices)
                self.monitor = P2PMonitor(active_devices=self.active_devices, output_dir=output_dir)
                self.profiler = P2PProfiler(output_dir=output_dir)
                self.debugger = P2PDebugger(output_dir=output_dir)
            except Exception as e:
                print(f"Failed to initialize P2P components: {e}")
    
    def validate_all(self) -> Dict:
        """
        Run all validation checks.
        
        Returns:
            Dictionary with validation results
        """
        print("Starting P2P Implementation Validation...")
        print(f"Active devices: {self.active_devices}")
        print(f"Output directory: {self.output_dir}")
        print()
        
        # Run validation categories
        self._validate_topology()
        self._validate_backend_initialization()
        self._validate_p2p_operations()
        self._validate_performance()
        self._validate_memory_management()
        self._validate_error_handling()
        self._validate_monitoring_integration()
        self._validate_configuration()
        
        # Generate summary
        self._generate_summary()
        
        # Save results
        self._save_validation_results()
        
        return self.validation_results
    
    def _validate_topology(self):
        """Validate P2P topology detection and analysis."""
        print("Validating P2P topology...")
        
        category_results = {
            "status": "unknown",
            "tests": {},
            "issues": [],
            "recommendations": []
        }
        
        try:
            if not self.topology:
                category_results["status"] = "failed"
                category_results["issues"].append("P2P topology not initialized")
                self.validation_results["validation_categories"]["topology"] = category_results
                return
            
            # Test topology initialization
            test_result = self._test_topology_initialization()
            category_results["tests"]["initialization"] = test_result
            
            # Test P2P matrix
            test_result = self._test_p2p_matrix()
            category_results["tests"]["p2p_matrix"] = test_result
            
            # Test connectivity analysis
            test_result = self._test_connectivity_analysis()
            category_results["tests"]["connectivity"] = test_result
            
            # Test tree building
            test_result = self._test_tree_building()
            category_results["tests"]["tree_building"] = test_result
            
            # Test algorithm selection
            test_result = self._test_algorithm_selection()
            category_results["tests"]["algorithm_selection"] = test_result
            
            # Determine overall status
            if all(test.get("passed", False) for test in category_results["tests"].values()):
                category_results["status"] = "passed"
            elif any(test.get("passed", False) for test in category_results["tests"].values()):
                category_results["status"] = "partial"
            else:
                category_results["status"] = "failed"
            
            # Generate recommendations
            category_results["recommendations"] = self._generate_topology_recommendations(category_results["tests"])
            
        except Exception as e:
            category_results["status"] = "error"
            category_results["issues"].append(f"Topology validation error: {e}")
        
        self.validation_results["validation_categories"]["topology"] = category_results
        print(f"Topology validation: {category_results['status']}")
    
    def _test_topology_initialization(self) -> Dict:
        """Test topology initialization."""
        result = {"passed": False, "details": {}, "issues": []}
        
        try:
            # Check basic properties
            result["details"]["num_devices"] = self.topology.num_devices
            result["details"]["active_devices"] = self.topology.active_devices
            result["details"]["device_to_index"] = self.topology.device_to_index
            result["details"]["index_to_device"] = self.topology.index_to_device
            
            # Validate device mappings
            if len(self.topology.device_to_index) == self.num_devices:
                result["details"]["device_mapping_valid"] = True
            else:
                result["details"]["device_mapping_valid"] = False
                result["issues"].append("Device mapping incomplete")
            
            # Check P2P matrix
            if self.topology.p2p_matrix is not None:
                result["details"]["p2p_matrix_initialized"] = True
                result["details"]["p2p_matrix_shape"] = self.topology.p2p_matrix.shape
            else:
                result["details"]["p2p_matrix_initialized"] = False
                result["issues"].append("P2P matrix not initialized")
            
            result["passed"] = len(result["issues"]) == 0
            
        except Exception as e:
            result["issues"].append(f"Topology initialization test failed: {e}")
        
        return result
    
    def _test_p2p_matrix(self) -> Dict:
        """Test P2P matrix properties."""
        result = {"passed": False, "details": {}, "issues": []}
        
        try:
            if not self.topology.p2p_matrix is not None:
                result["issues"].append("P2P matrix not available")
                return result
            
            matrix = self.topology.p2p_matrix
            
            # Check matrix shape
            expected_shape = (self.num_devices, self.num_devices)
            if matrix.shape == expected_shape:
                result["details"]["matrix_shape_correct"] = True
            else:
                result["details"]["matrix_shape_correct"] = False
                result["issues"].append(f"Matrix shape {matrix.shape} != expected {expected_shape}")
            
            # Check diagonal elements (should be True)
            diagonal_correct = True
            for i in range(self.num_devices):
                if not matrix[i][i]:
                    diagonal_correct = False
                    break
            result["details"]["diagonal_correct"] = diagonal_correct
            if not diagonal_correct:
                result["issues"].append("Diagonal elements should be True")
            
            # Check symmetry (P2P should be symmetric)
            symmetry_correct = True
            for i in range(self.num_devices):
                for j in range(self.num_devices):
                    if matrix[i][j] != matrix[j][i]:
                        symmetry_correct = False
                        break
                if not symmetry_correct:
                    break
            result["details"]["symmetry_correct"] = symmetry_correct
            if not symmetry_correct:
                result["issues"].append("P2P matrix should be symmetric")
            
            # Calculate connectivity ratio
            connectivity_ratio = self.topology.get_connectivity_ratio()
            result["details"]["connectivity_ratio"] = connectivity_ratio
            result["details"]["is_fully_connected"] = self.topology.is_fully_connected()
            
            result["passed"] = len(result["issues"]) == 0
            
        except Exception as e:
            result["issues"].append(f"P2P matrix test failed: {e}")
        
        return result
    
    def _test_connectivity_analysis(self) -> Dict:
        """Test connectivity analysis."""
        result = {"passed": False, "details": {}, "issues": []}
        
        try:
            # Test connectivity ratio
            connectivity_ratio = self.topology.get_connectivity_ratio()
            result["details"]["connectivity_ratio"] = connectivity_ratio
            
            if 0.0 <= connectivity_ratio <= 1.0:
                result["details"]["connectivity_ratio_valid"] = True
            else:
                result["details"]["connectivity_ratio_valid"] = False
                result["issues"].append(f"Invalid connectivity ratio: {connectivity_ratio}")
            
            # Test fully connected detection
            is_fully_connected = self.topology.is_fully_connected()
            result["details"]["is_fully_connected"] = is_fully_connected
            
            # Test topology summary
            summary = self.topology.get_topology_summary()
            result["details"]["topology_summary"] = summary
            
            required_keys = ["status", "num_devices", "topology_type", "connected_pairs", 
                           "total_pairs", "connectivity_ratio", "is_fully_connected"]
            
            summary_valid = all(key in summary for key in required_keys)
            result["details"]["summary_valid"] = summary_valid
            if not summary_valid:
                result["issues"].append("Topology summary missing required keys")
            
            result["passed"] = len(result["issues"]) == 0
            
        except Exception as e:
            result["issues"].append(f"Connectivity analysis test failed: {e}")
        
        return result
    
    def _test_tree_building(self) -> Dict:
        """Test tree building algorithms."""
        result = {"passed": False, "details": {}, "issues": []}
        
        try:
            # Test binary tree
            binary_tree = self.topology.build_binary_tree()
            result["details"]["binary_tree_built"] = binary_tree is not None
            
            if binary_tree:
                # Check tree structure
                required_keys = ["root", "children", "parent", "depth", "tree_type"]
                tree_valid = all(key in binary_tree for key in required_keys)
                result["details"]["binary_tree_structure_valid"] = tree_valid
                
                if tree_valid:
                    # Check that all devices are included
                    all_devices = set(binary_tree["children"].keys()) | set(binary_tree["parent"].keys())
                    if binary_tree["root"] not in binary_tree["parent"]:
                        all_devices.add(binary_tree["root"])
                    
                    if len(all_devices) == self.num_devices:
                        result["details"]["binary_tree_complete"] = True
                    else:
                        result["details"]["binary_tree_complete"] = False
                        result["issues"].append(f"Binary tree missing devices: {self.num_devices - len(all_devices)}")
            
            # Test k-ary tree
            kary_tree = self.topology.build_kary_tree(4)
            result["details"]["kary_tree_built"] = kary_tree is not None
            
            # Test balanced tree
            balanced_tree = self.topology.build_balanced_tree()
            result["details"]["balanced_tree_built"] = balanced_tree is not None
            
            # Test tree stats
            if binary_tree:
                stats = self.topology.get_tree_stats(binary_tree)
                result["details"]["tree_stats"] = stats
                result["details"]["tree_stats_valid"] = isinstance(stats, dict)
            
            result["passed"] = len(result["issues"]) == 0
            
        except Exception as e:
            result["issues"].append(f"Tree building test failed: {e}")
        
        return result
    
    def _test_algorithm_selection(self) -> Dict:
        """Test algorithm selection."""
        result = {"passed": False, "details": {}, "issues": []}
        
        try:
            # Test with different tensor sizes
            test_sizes = [1024, 1024*1024, 10*1024*1024]
            algorithms = []
            
            for size in test_sizes:
                algorithm = self.topology.select_reduce_algorithm(size)
                algorithms.append(algorithm)
                result["details"][f"algorithm_for_{size//1024}KB"] = algorithm
                
                # Validate algorithm
                valid_algorithms = ["ring", "binary_tree", "kary_tree", "balanced_tree"]
                if algorithm not in valid_algorithms:
                    result["issues"].append(f"Invalid algorithm for size {size}: {algorithm}")
            
            # Test optimal topology building
            for operation in ["reduce", "broadcast", "gather"]:
                topology_result = self.topology.build_optimal_topology(operation, 1024*1024)
                result["details"][f"optimal_topology_{operation}"] = topology_result
                
                if not isinstance(topology_result, dict):
                    result["issues"].append(f"Invalid topology result for {operation}")
                elif "type" not in topology_result or "reason" not in topology_result:
                    result["issues"].append(f"Incomplete topology result for {operation}")
            
            result["passed"] = len(result["issues"]) == 0
            
        except Exception as e:
            result["issues"].append(f"Algorithm selection test failed: {e}")
        
        return result
    
    def _generate_topology_recommendations(self, test_results: Dict) -> List[str]:
        """Generate topology validation recommendations."""
        recommendations = []
        
        # Check connectivity
        connectivity_test = test_results.get("connectivity", {})
        if connectivity_test.get("details", {}).get("connectivity_ratio", 0) < 0.5:
            recommendations.append("Low P2P connectivity detected - consider using alternative communication patterns")
        
        # Check tree building
        tree_test = test_results.get("tree_building", {})
        if not tree_test.get("details", {}).get("binary_tree_complete", False):
            recommendations.append("Tree building incomplete - check P2P connectivity between devices")
        
        # Check algorithm selection
        algo_test = test_results.get("algorithm_selection", {})
        if algo_test.get("issues"):
            recommendations.append("Algorithm selection issues detected - review topology analysis logic")
        
        return recommendations
    
    def _validate_backend_initialization(self):
        """Validate P2P backend initialization."""
        print("Validating backend initialization...")
        
        category_results = {
            "status": "unknown",
            "tests": {},
            "issues": [],
            "recommendations": []
        }
        
        try:
            # Initialize backends
            if not self._initialize_backends():
                category_results["status"] = "failed"
                category_results["issues"].append("Failed to initialize backends")
                self.validation_results["validation_categories"]["backend_initialization"] = category_results
                return
            
            # Test backend properties
            test_result = self._test_backend_properties()
            category_results["tests"]["properties"] = test_result
            
            # Test P2P availability
            test_result = self._test_p2p_availability()
            category_results["tests"]["p2p_availability"] = test_result
            
            # Test fallback mechanism
            test_result = self._test_fallback_mechanism()
            category_results["tests"]["fallback"] = test_result
            
            # Determine overall status
            if all(test.get("passed", False) for test in category_results["tests"].values()):
                category_results["status"] = "passed"
            elif any(test.get("passed", False) for test in category_results["tests"].values()):
                category_results["status"] = "partial"
            else:
                category_results["status"] = "failed"
            
            # Generate recommendations
            category_results["recommendations"] = self._generate_backend_recommendations(category_results["tests"])
            
        except Exception as e:
            category_results["status"] = "error"
            category_results["issues"].append(f"Backend initialization validation error: {e}")
        
        self.validation_results["validation_categories"]["backend_initialization"] = category_results
        print(f"Backend initialization validation: {category_results['status']}")
    
    def _initialize_backends(self) -> bool:
        """Initialize P2P backends."""
        try:
            for device in self.active_devices:
                backend = TPBackendP2P(
                    device=device,
                    active_devices=self.active_devices,
                    output_device=self.active_devices[0],
                    init_method="tcp://localhost:12345",
                    master=(device == self.active_devices[0]),
                    uuid="p2p_validation"
                )
                self.backends[device] = backend
            return True
        except Exception as e:
            print(f"Failed to initialize backends: {e}")
            return False
    
    def _test_backend_properties(self) -> Dict:
        """Test backend properties."""
        result = {"passed": False, "details": {}, "issues": []}
        
        try:
            for device, backend in self.backends.items():
                device_results = {}
                
                # Check basic properties
                device_results["device"] = backend.device
                device_results["active_devices"] = backend.active_devices
                device_results["world_size"] = backend.world_size
                device_results["rank"] = backend.rank
                device_results["output_device"] = backend.output_device
                
                # Validate properties
                if backend.device == device:
                    device_results["device_correct"] = True
                else:
                    device_results["device_correct"] = False
                    result["issues"].append(f"Device mismatch for backend {device}")
                
                if backend.world_size == self.num_devices:
                    device_results["world_size_correct"] = True
                else:
                    device_results["world_size_correct"] = False
                    result["issues"].append(f"World size mismatch for backend {device}")
                
                if backend.rank == self.active_devices.index(device):
                    device_results["rank_correct"] = True
                else:
                    device_results["rank_correct"] = False
                    result["issues"].append(f"Rank mismatch for backend {device}")
                
                # Check P2P availability
                if hasattr(backend, 'use_p2p'):
                    device_results["p2p_available"] = backend.use_p2p
                else:
                    device_results["p2p_available"] = False
                    result["issues"].append(f"P2P availability not set for backend {device}")
                
                result["details"][f"device_{device}"] = device_results
            
            result["passed"] = len(result["issues"]) == 0
            
        except Exception as e:
            result["issues"].append(f"Backend properties test failed: {e}")
        
        return result
    
    def _test_p2p_availability(self) -> Dict:
        """Test P2P availability."""
        result = {"passed": False, "details": {}, "issues": []}
        
        try:
            p2p_available_count = 0
            
            for device, backend in self.backends.items():
                if hasattr(backend, 'use_p2p') and backend.use_p2p:
                    p2p_available_count += 1
                    
                    # Test P2P operations
                    try:
                        # Test direct copy
                        test_tensor = torch.randn(100, 100, device=device)
                        if hasattr(backend, 'copy_tensor_direct'):
                            copied = backend.copy_tensor_direct(device, device, test_tensor)
                            result["details"][f"direct_copy_{device}"] = copied is not None
                        
                        # Test bandwidth measurement
                        if hasattr(backend, 'measure_p2p_bandwidth'):
                            bandwidth = backend.measure_p2p_bandwidth(device, device, 1, 3)
                            result["details"][f"bandwidth_{device}"] = bandwidth
                        
                    except Exception as e:
                        result["issues"].append(f"P2P operation test failed for device {device}: {e}")
            
            result["details"]["p2p_available_count"] = p2p_available_count
            result["details"]["total_backends"] = len(self.backends)
            
            if p2p_available_count == len(self.backends):
                result["passed"] = True
            elif p2p_available_count > 0:
                result["passed"] = False  # Partial success is still a failure for availability
                result["issues"].append(f"P2P only available on {p2p_available_count}/{len(self.backends)} backends")
            else:
                result["issues"].append("P2P not available on any backend")
            
        except Exception as e:
            result["issues"].append(f"P2P availability test failed: {e}")
        
        return result
    
    def _test_fallback_mechanism(self) -> Dict:
        """Test fallback mechanism."""
        result = {"passed": False, "details": {}, "issues": []}
        
        try:
            for device, backend in self.backends.items():
                # Check if fallback backend exists
                if hasattr(backend, 'fallback'):
                    result["details"][f"fallback_exists_{device}"] = True
                    
                    # Test fallback operations
                    try:
                        test_tensor = torch.randn(100, 100, device=device)
                        
                        # Test broadcast through fallback
                        backend.fallback.broadcast(test_tensor, device)
                        result["details"][f"fallback_broadcast_{device}"] = True
                        
                        # Test all_reduce through fallback
                        test_tensor_copy = test_tensor.clone()
                        backend.fallback.all_reduce(test_tensor_copy)
                        result["details"][f"fallback_all_reduce_{device}"] = True
                        
                    except Exception as e:
                        result["issues"].append(f"Fallback operation test failed for device {device}: {e}")
                else:
                    result["details"][f"fallback_exists_{device}"] = False
                    result["issues"].append(f"No fallback backend available for device {device}")
            
            result["passed"] = len(result["issues"]) == 0
            
        except Exception as e:
            result["issues"].append(f"Fallback mechanism test failed: {e}")
        
        return result
    
    def _generate_backend_recommendations(self, test_results: Dict) -> List[str]:
        """Generate backend validation recommendations."""
        recommendations = []
        
        # Check P2P availability
        p2p_test = test_results.get("p2p_availability", {})
        if p2p_test.get("details", {}).get("p2p_available_count", 0) < len(self.backends):
            recommendations.append("P2P not available on all backends - check hardware compatibility")
        
        # Check fallback mechanism
        fallback_test = test_results.get("fallback", {})
        if fallback_test.get("issues"):
            recommendations.append("Fallback mechanism issues detected - ensure robust error handling")
        
        return recommendations
    
    def _validate_p2p_operations(self):
        """Validate P2P operations."""
        print("Validating P2P operations...")
        
        category_results = {
            "status": "unknown",
            "tests": {},
            "issues": [],
            "recommendations": []
        }
        
        try:
            if not self.backends:
                category_results["status"] = "failed"
                category_results["issues"].append("No backends available for operation testing")
                self.validation_results["validation_categories"]["p2p_operations"] = category_results
                return
            
            # Test broadcast operation
            test_result = self._test_broadcast_operation()
            category_results["tests"]["broadcast"] = test_result
            
            # Test all_reduce operation
            test_result = self._test_all_reduce_operation()
            category_results["tests"]["all_reduce"] = test_result
            
            # Test gather operation
            test_result = self._test_gather_operation()
            category_results["tests"]["gather"] = test_result
            
            # Test direct memory operations
            test_result = self._test_direct_memory_operations()
            category_results["tests"]["direct_memory"] = test_result
            
            # Determine overall status
            if all(test.get("passed", False) for test in category_results["tests"].values()):
                category_results["status"] = "passed"
            elif any(test.get("passed", False) for test in category_results["tests"].values()):
                category_results["status"] = "partial"
            else:
                category_results["status"] = "failed"
            
            # Generate recommendations
            category_results["recommendations"] = self._generate_operation_recommendations(category_results["tests"])
            
        except Exception as e:
            category_results["status"] = "error"
            category_results["issues"].append(f"P2P operations validation error: {e}")
        
        self.validation_results["validation_categories"]["p2p_operations"] = category_results
        print(f"P2P operations validation: {category_results['status']}")
    
    def _test_broadcast_operation(self) -> Dict:
        """Test broadcast operation."""
        result = {"passed": False, "details": {}, "issues": []}
        
        try:
            device = self.active_devices[0]
            backend = self.backends[device]
            
            # Test different tensor sizes
            tensor_sizes = [100, 1000, 10000]
            
            for size in tensor_sizes:
                test_tensor = torch.randn(size, dtype=torch.float32, device=device)
                
                try:
                    start_time = time.time()
                    backend.broadcast(test_tensor, device)
                    end_time = time.time()
                    
                    result["details"][f"broadcast_size_{size}"] = {
                        "success": True,
                        "time": end_time - start_time
                    }
                    
                except Exception as e:
                    result["details"][f"broadcast_size_{size}"] = {
                        "success": False,
                        "error": str(e)
                    }
                    result["issues"].append(f"Broadcast failed for size {size}: {e}")
            
            # Test correctness
            try:
                original_tensor = torch.randn(1000, dtype=torch.float32, device=device)
                test_tensor = original_tensor.clone()
                
                backend.broadcast(test_tensor, device)
                
                if torch.allclose(original_tensor, test_tensor):
                    result["details"]["broadcast_correctness"] = True
                else:
                    result["details"]["broadcast_correctness"] = False
                    result["issues"].append("Broadcast correctness test failed")
                
            except Exception as e:
                result["issues"].append(f"Broadcast correctness test error: {e}")
            
            result["passed"] = len(result["issues"]) == 0
            
        except Exception as e:
            result["issues"].append(f"Broadcast operation test failed: {e}")
        
        return result
    
    def _test_all_reduce_operation(self) -> Dict:
        """Test all_reduce operation."""
        result = {"passed": False, "details": {}, "issues": []}
        
        try:
            device = self.active_devices[0]
            backend = self.backends[device]
            
            # Test different tensor sizes
            tensor_sizes = [100, 1000, 10000]
            
            for size in tensor_sizes:
                test_tensor = torch.randn(size, dtype=torch.float32, device=device)
                original_sum = test_tensor.sum().item()
                
                try:
                    start_time = time.time()
                    backend.all_reduce(test_tensor)
                    end_time = time.time()
                    
                    # Check if tensor was reduced (should be multiplied by num_devices)
                    expected_sum = original_sum * self.num_devices
                    actual_sum = test_tensor.sum().item()
                    
                    reduction_correct = np.isclose(actual_sum, expected_sum, rtol=1e-3)
                    
                    result["details"][f"all_reduce_size_{size}"] = {
                        "success": True,
                        "time": end_time - start_time,
                        "reduction_correct": reduction_correct,
                        "expected_sum": expected_sum,
                        "actual_sum": actual_sum
                    }
                    
                    if not reduction_correct:
                        result["issues"].append(f"All_reduce correctness failed for size {size}")
                
                except Exception as e:
                    result["details"][f"all_reduce_size_{size}"] = {
                        "success": False,
                        "error": str(e)
                    }
                    result["issues"].append(f"All_reduce failed for size {size}: {e}")
            
            result["passed"] = len(result["issues"]) == 0
            
        except Exception as e:
            result["issues"].append(f"All_reduce operation test failed: {e}")
        
        return result
    
    def _test_gather_operation(self) -> Dict:
        """Test gather operation."""
        result = {"passed": False, "details": {}, "issues": []}
        
        try:
            device = self.active_devices[0]
            backend = self.backends[device]
            
            # Test different tensor sizes
            tensor_sizes = [100, 1000, 10000]
            
            for size in tensor_sizes:
                test_tensor = torch.randn(size, dtype=torch.float32, device=device)
                out_tensor = torch.zeros(size * self.num_devices, dtype=torch.float32, device=device)
                gather_devices = torch.tensor(self.active_devices, dtype=torch.int)
                ldims = [size] * self.num_devices
                
                try:
                    start_time = time.time()
                    backend.gather(test_tensor, out_tensor, gather_devices, device, ldims)
                    end_time = time.time()
                    
                    # Check if gather was successful
                    # The first part of out_tensor should match test_tensor
                    gathered_correct = torch.allclose(out_tensor[:size], test_tensor)
                    
                    result["details"][f"gather_size_{size}"] = {
                        "success": True,
                        "time": end_time - start_time,
                        "gather_correct": gathered_correct
                    }
                    
                    if not gathered_correct:
                        result["issues"].append(f"Gather correctness failed for size {size}")
                
                except Exception as e:
                    result["details"][f"gather_size_{size}"] = {
                        "success": False,
                        "error": str(e)
                    }
                    result["issues"].append(f"Gather failed for size {size}: {e}")
            
            result["passed"] = len(result["issues"]) == 0
            
        except Exception as e:
            result["issues"].append(f"Gather operation test failed: {e}")
        
        return result
    
    def _test_direct_memory_operations(self) -> Dict:
        """Test direct memory operations."""
        result = {"passed": False, "details": {}, "issues": []}
        
        try:
            if len(self.active_devices) < 2:
                result["issues"].append("Need at least 2 devices for direct memory testing")
                return result
            
            src_device = self.active_devices[0]
            dst_device = self.active_devices[1]
            backend = self.backends[src_device]
            
            # Test direct tensor copy
            test_tensor = torch.randn(1000, dtype=torch.float32, device=src_device)
            
            try:
                if hasattr(backend, 'copy_tensor_direct'):
                    copied_tensor = backend.copy_tensor_direct(src_device, dst_device, test_tensor)
                    
                    if copied_tensor is not None and copied_tensor.device.index == dst_device:
                        if torch.allclose(test_tensor.cpu(), copied_tensor.cpu()):
                            result["details"]["direct_copy_correctness"] = True
                        else:
                            result["details"]["direct_copy_correctness"] = False
                            result["issues"].append("Direct copy correctness failed")
                        
                        result["details"]["direct_copy_success"] = True
                    else:
                        result["details"]["direct_copy_success"] = False
                        result["issues"].append("Direct copy returned None or wrong device")
                else:
                    result["issues"].append("Direct copy not available")
                
            except Exception as e:
                result["issues"].append(f"Direct copy test failed: {e}")
            
            # Test memory validation
            try:
                if hasattr(backend, 'validate_p2p_memory_access'):
                    is_valid = backend.validate_p2p_memory_access(src_device, dst_device, 1024)
                    result["details"]["memory_validation"] = is_valid
                    
                    if not is_valid:
                        result["issues"].append("Memory validation failed")
                else:
                    result["issues"].append("Memory validation not available")
                
            except Exception as e:
                result["issues"].append(f"Memory validation test failed: {e}")
            
            result["passed"] = len(result["issues"]) == 0
            
        except Exception as e:
            result["issues"].append(f"Direct memory operations test failed: {e}")
        
        return result
    
    def _generate_operation_recommendations(self, test_results: Dict) -> List[str]:
        """Generate operation validation recommendations."""
        recommendations = []
        
        # Check broadcast
        broadcast_test = test_results.get("broadcast", {})
        if broadcast_test.get("issues"):
            recommendations.append("Broadcast operation issues detected - check P2P connectivity")
        
        # Check all_reduce
        all_reduce_test = test_results.get("all_reduce", {})
        if all_reduce_test.get("issues"):
            recommendations.append("All_reduce operation issues detected - check reduction logic")
        
        # Check gather
        gather_test = test_results.get("gather", {})
        if gather_test.get("issues"):
            recommendations.append("Gather operation issues detected - check gather implementation")
        
        # Check direct memory
        direct_test = test_results.get("direct_memory", {})
        if direct_test.get("issues"):
            recommendations.append("Direct memory operation issues detected - check P2P memory access")
        
        return recommendations
    
    def _validate_performance(self):
        """Validate P2P performance."""
        print("Validating P2P performance...")
        
        category_results = {
            "status": "unknown",
            "tests": {},
            "issues": [],
            "recommendations": []
        }
        
        try:
            if not self.backends:
                category_results["status"] = "failed"
                category_results["issues"].append("No backends available for performance testing")
                self.validation_results["validation_categories"]["performance"] = category_results
                return
            
            # Test bandwidth performance
            test_result = self._test_bandwidth_performance()
            category_results["tests"]["bandwidth"] = test_result
            
            # Test latency performance
            test_result = self._test_latency_performance()
            category_results["tests"]["latency"] = test_result
            
            # Test throughput performance
            test_result = self._test_throughput_performance()
            category_results["tests"]["throughput"] = test_result
            
            # Determine overall status
            if all(test.get("passed", False) for test in category_results["tests"].values()):
                category_results["status"] = "passed"
            elif any(test.get("passed", False) for test in category_results["tests"].values()):
                category_results["status"] = "partial"
            else:
                category_results["status"] = "failed"
            
            # Generate recommendations
            category_results["recommendations"] = self._generate_performance_recommendations(category_results["tests"])
            
        except Exception as e:
            category_results["status"] = "error"
            category_results["issues"].append(f"Performance validation error: {e}")
        
        self.validation_results["validation_categories"]["performance"] = category_results
        print(f"Performance validation: {category_results['status']}")
    
    def _test_bandwidth_performance(self) -> Dict:
        """Test bandwidth performance."""
        result = {"passed": False, "details": {}, "issues": []}
        
        try:
            if len(self.active_devices) < 2:
                result["issues"].append("Need at least 2 devices for bandwidth testing")
                return result
            
            src_device = self.active_devices[0]
            dst_device = self.active_devices[1]
            backend = self.backends[src_device]
            
            # Test different transfer sizes
            sizes_mb = [1, 4, 16]
            
            for size_mb in sizes_mb:
                try:
                    if hasattr(backend, 'measure_p2p_bandwidth'):
                        bandwidth = backend.measure_p2p_bandwidth(src_device, dst_device, size_mb, 3)
                        result["details"][f"bandwidth_{size_mb}MB"] = bandwidth
                        
                        # Check if bandwidth is reasonable (at least 0.1 GB/s)
                        if bandwidth > 0.1:
                            result["details"][f"bandwidth_{size_mb}MB_reasonable"] = True
                        else:
                            result["details"][f"bandwidth_{size_mb}MB_reasonable"] = False
                            result["issues"].append(f"Low bandwidth for {size_mb}MB transfer: {bandwidth} GB/s")
                    else:
                        result["issues"].append(f"Bandwidth measurement not available for {size_mb}MB")
                
                except Exception as e:
                    result["issues"].append(f"Bandwidth test failed for {size_mb}MB: {e}")
            
            result["passed"] = len(result["issues"]) == 0
            
        except Exception as e:
            result["issues"].append(f"Bandwidth performance test failed: {e}")
        
        return result
    
    def _test_latency_performance(self) -> Dict:
        """Test latency performance."""
        result = {"passed": False, "details": {}, "issues": []}
        
        try:
            if len(self.active_devices) < 2:
                result["issues"].append("Need at least 2 devices for latency testing")
                return result
            
            src_device = self.active_devices[0]
            dst_device = self.active_devices[1]
            backend = self.backends[src_device]
            
            # Test different message sizes
            sizes_kb = [1, 4, 16]
            
            for size_kb in sizes_kb:
                try:
                    if hasattr(backend, 'measure_p2p_latency'):
                        latency = backend.measure_p2p_latency(src_device, dst_device, size_kb, 10)
                        result["details"][f"latency_{size_kb}KB"] = latency
                        
                        # Check if latency is reasonable (less than 100ms)
                        if latency < 100000:  # 100ms in microseconds
                            result["details"][f"latency_{size_kb}KB_reasonable"] = True
                        else:
                            result["details"][f"latency_{size_kb}KB_reasonable"] = False
                            result["issues"].append(f"High latency for {size_kb}KB message: {latency} μs")
                    else:
                        result["issues"].append(f"Latency measurement not available for {size_kb}KB")
                
                except Exception as e:
                    result["issues"].append(f"Latency test failed for {size_kb}KB: {e}")
            
            result["passed"] = len(result["issues"]) == 0
            
        except Exception as e:
            result["issues"].append(f"Latency performance test failed: {e}")
        
        return result
    
    def _test_throughput_performance(self) -> Dict:
        """Test throughput performance."""
        result = {"passed": False, "details": {}, "issues": []}
        
        try:
            device = self.active_devices[0]
            backend = self.backends[device]
            
            # Test operation throughput
            tensor_size = 1024*1024  # 1M elements
            test_tensor = torch.randn(tensor_size, dtype=torch.float32, device=device)
            
            operations = ["broadcast", "all_reduce"]
            
            for op in operations:
                try:
                    # Measure throughput (operations per second)
                    num_iterations = 10
                    start_time = time.time()
                    
                    for _ in range(num_iterations):
                        if op == "broadcast":
                            backend.broadcast(test_tensor, device)
                        elif op == "all_reduce":
                            test_tensor_copy = test_tensor.clone()
                            backend.all_reduce(test_tensor_copy)
                    
                    end_time = time.time()
                    total_time = end_time - start_time
                    throughput = num_iterations / total_time
                    
                    result["details"][f"throughput_{op}"] = throughput
                    
                    # Check if throughput is reasonable (at least 1 op/sec)
                    if throughput > 1.0:
                        result["details"][f"throughput_{op}_reasonable"] = True
                    else:
                        result["details"][f"throughput_{op}_reasonable"] = False
                        result["issues"].append(f"Low throughput for {op}: {throughput} ops/sec")
                
                except Exception as e:
                    result["issues"].append(f"Throughput test failed for {op}: {e}")
            
            result["passed"] = len(result["issues"]) == 0
            
        except Exception as e:
            result["issues"].append(f"Throughput performance test failed: {e}")
        
        return result
    
    def _generate_performance_recommendations(self, test_results: Dict) -> List[str]:
        """Generate performance validation recommendations."""
        recommendations = []
        
        # Check bandwidth
        bandwidth_test = test_results.get("bandwidth", {})
        if bandwidth_test.get("issues"):
            recommendations.append("Low bandwidth detected - check P2P connectivity and hardware")
        
        # Check latency
        latency_test = test_results.get("latency", {})
        if latency_test.get("issues"):
            recommendations.append("High latency detected - optimize communication patterns")
        
        # Check throughput
        throughput_test = test_results.get("throughput", {})
        if throughput_test.get("issues"):
            recommendations.append("Low throughput detected - consider algorithm optimization")
        
        return recommendations
    
    def _validate_memory_management(self):
        """Validate memory management."""
        print("Validating memory management...")
        
        category_results = {
            "status": "unknown",
            "tests": {},
            "issues": [],
            "recommendations": []
        }
        
        try:
            if not self.backends:
                category_results["status"] = "failed"
                category_results["issues"].append("No backends available for memory testing")
                self.validation_results["validation_categories"]["memory_management"] = category_results
                return
            
            # Test memory pool usage
            test_result = self._test_memory_pool_usage()
            category_results["tests"]["memory_pool"] = test_result
            
            # Test memory efficiency
            test_result = self._test_memory_efficiency()
            category_results["tests"]["memory_efficiency"] = test_result
            
            # Test memory cleanup
            test_result = self._test_memory_cleanup()
            category_results["tests"]["memory_cleanup"] = test_result
            
            # Determine overall status
            if all(test.get("passed", False) for test in category_results["tests"].values()):
                category_results["status"] = "passed"
            elif any(test.get("passed", False) for test in category_results["tests"].values()):
                category_results["status"] = "partial"
            else:
                category_results["status"] = "failed"
            
            # Generate recommendations
            category_results["recommendations"] = self._generate_memory_recommendations(category_results["tests"])
            
        except Exception as e:
            category_results["status"] = "error"
            category_results["issues"].append(f"Memory management validation error: {e}")
        
        self.validation_results["validation_categories"]["memory_management"] = category_results
        print(f"Memory management validation: {category_results['status']}")
    
    def _test_memory_pool_usage(self) -> Dict:
        """Test memory pool usage."""
        result = {"passed": False, "details": {}, "issues": []}
        
        try:
            for device, backend in self.backends.items():
                device_results = {}
                
                try:
                    if hasattr(backend, 'get_memory_pool_stats'):
                        stats = backend.get_memory_pool_stats()
                        device_results["memory_stats"] = stats
                        
                        # Check if stats are reasonable
                        if isinstance(stats, dict) and "pool_usage_bytes" in stats:
                            device_results["stats_valid"] = True
                            
                            # Check pool usage
                            if stats["pool_usage_bytes"] >= 0:
                                device_results["pool_usage_valid"] = True
                            else:
                                device_results["pool_usage_valid"] = False
                                result["issues"].append(f"Invalid pool usage for device {device}")
                        else:
                            device_results["stats_valid"] = False
                            result["issues"].append(f"Invalid memory stats for device {device}")
                    else:
                        device_results["memory_stats_available"] = False
                        result["issues"].append(f"Memory stats not available for device {device}")
                
                except Exception as e:
                    result["issues"].append(f"Memory pool test failed for device {device}: {e}")
                
                result["details"][f"device_{device}"] = device_results
            
            result["passed"] = len(result["issues"]) == 0
            
        except Exception as e:
            result["issues"].append(f"Memory pool usage test failed: {e}")
        
        return result
    
    def _test_memory_efficiency(self) -> Dict:
        """Test memory efficiency."""
        result = {"passed": False, "details": {}, "issues": []}
        
        try:
            device = self.active_devices[0]
            backend = self.backends[device]
            
            # Test memory usage with different tensor sizes
            tensor_sizes = [1024, 1024*1024, 4*1024*1024]
            
            for size in tensor_sizes:
                try:
                    # Get initial memory usage
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        initial_memory = torch.cuda.memory_allocated(device)
                    
                    # Create tensor and perform operation
                    test_tensor = torch.randn(size, dtype=torch.float32, device=device)
                    backend.broadcast(test_tensor, device)
                    
                    # Get final memory usage
                    if torch.cuda.is_available():
                        final_memory = torch.cuda.memory_allocated(device)
                        memory_increase = final_memory - initial_memory
                        
                        result["details"][f"memory_increase_{size}"] = memory_increase
                        
                        # Check if memory increase is reasonable
                        expected_increase = size * 4  # 4 bytes per float32
                        if memory_increase <= expected_increase * 2:  # Allow 2x overhead
                            result["details"][f"memory_efficient_{size}"] = True
                        else:
                            result["details"][f"memory_efficient_{size}"] = False
                            result["issues"].append(f"High memory usage for size {size}: {memory_increase} bytes")
                
                except Exception as e:
                    result["issues"].append(f"Memory efficiency test failed for size {size}: {e}")
            
            result["passed"] = len(result["issues"]) == 0
            
        except Exception as e:
            result["issues"].append(f"Memory efficiency test failed: {e}")
        
        return result
    
    def _test_memory_cleanup(self) -> Dict:
        """Test memory cleanup."""
        result = {"passed": False, "details": {}, "issues": []}
        
        try:
            device = self.active_devices[0]
            
            # Allocate memory
            tensors = []
            for i in range(10):
                tensor = torch.randn(1024*1024, dtype=torch.float32, device=device)
                tensors.append(tensor)
            
            # Get memory usage
            if torch.cuda.is_available():
                peak_memory = torch.cuda.memory_allocated(device)
                
                # Clean up tensors
                for tensor in tensors:
                    del tensor
                
                torch.cuda.empty_cache()
                
                # Check memory after cleanup
                final_memory = torch.cuda.memory_allocated(device)
                memory_freed = peak_memory - final_memory
                
                result["details"]["peak_memory"] = peak_memory
                result["details"]["final_memory"] = final_memory
                result["details"]["memory_freed"] = memory_freed
                
                # Check if memory was freed
                if memory_freed > peak_memory * 0.8:  # At least 80% freed
                    result["details"]["cleanup_successful"] = True
                else:
                    result["details"]["cleanup_successful"] = False
                    result["issues"].append(f"Incomplete memory cleanup: {memory_freed}/{peak_memory} bytes freed")
            
            result["passed"] = len(result["issues"]) == 0
            
        except Exception as e:
            result["issues"].append(f"Memory cleanup test failed: {e}")
        
        return result
    
    def _generate_memory_recommendations(self, test_results: Dict) -> List[str]:
        """Generate memory management recommendations."""
        recommendations = []
        
        # Check memory pool
        pool_test = test_results.get("memory_pool", {})
        if pool_test.get("issues"):
            recommendations.append("Memory pool issues detected - check pool initialization and management")
        
        # Check memory efficiency
        efficiency_test = test_results.get("memory_efficiency", {})
        if efficiency_test.get("issues"):
            recommendations.append("Memory efficiency issues detected - optimize memory usage patterns")
        
        # Check memory cleanup
        cleanup_test = test_results.get("memory_cleanup", {})
        if cleanup_test.get("issues"):
            recommendations.append("Memory cleanup issues detected - ensure proper memory deallocation")
        
        return recommendations
    
    def _validate_error_handling(self):
        """Validate error handling."""
        print("Validating error handling...")
        
        category_results = {
            "status": "unknown",
            "tests": {},
            "issues": [],
            "recommendations": []
        }
        
        try:
            # Test invalid parameters
            test_result = self._test_invalid_parameters()
            category_results["tests"]["invalid_parameters"] = test_result
            
            # Test error recovery
            test_result = self._test_error_recovery()
            category_results["tests"]["error_recovery"] = test_result
            
            # Test timeout handling
            test_result = self._test_timeout_handling()
            category_results["tests"]["timeout_handling"] = test_result
            
            # Determine overall status
            if all(test.get("passed", False) for test in category_results["tests"].values()):
                category_results["status"] = "passed"
            elif any(test.get("passed", False) for test in category_results["tests"].values()):
                category_results["status"] = "partial"
            else:
                category_results["status"] = "failed"
            
            # Generate recommendations
            category_results["recommendations"] = self._generate_error_handling_recommendations(category_results["tests"])
            
        except Exception as e:
            category_results["status"] = "error"
            category_results["issues"].append(f"Error handling validation error: {e}")
        
        self.validation_results["validation_categories"]["error_handling"] = category_results
        print(f"Error handling validation: {category_results['status']}")
    
    def _test_invalid_parameters(self) -> Dict:
        """Test handling of invalid parameters."""
        result = {"passed": False, "details": {}, "issues": []}
        
        try:
            if not self.backends:
                result["issues"].append("No backends available for parameter testing")
                return result
            
            device = self.active_devices[0]
            backend = self.backends[device]
            
            # Test invalid device ID
            try:
                test_tensor = torch.randn(100, dtype=torch.float32, device=device)
                backend.broadcast(test_tensor, 999)  # Invalid device
                
                # If we get here, the backend didn't handle the invalid device properly
                result["issues"].append("Invalid device ID not handled properly")
            except Exception as e:
                result["details"]["invalid_device_handled"] = True
            
            # Test None tensor
            try:
                backend.broadcast(None, device)
                
                # If we get here, the backend didn't handle None tensor properly
                result["issues"].append("None tensor not handled properly")
            except Exception as e:
                result["details"]["none_tensor_handled"] = True
            
            # Test mismatched tensor sizes
            try:
                if len(self.active_devices) > 1:
                    large_tensor = torch.randn(200, dtype=torch.float32, device=device)
                    small_tensor = torch.zeros(100, dtype=torch.float32, device=device)
                    
                    # This should fail gracefully
                    backend.gather(large_tensor, small_tensor, torch.tensor([0]), device, [200])
                    
                    result["issues"].append("Mismatched tensor sizes not handled properly")
            except Exception as e:
                result["details"]["mismatched_sizes_handled"] = True
            
            result["passed"] = len(result["issues"]) == 0
            
        except Exception as e:
            result["issues"].append(f"Invalid parameters test failed: {e}")
        
        return result
    
    def _test_error_recovery(self) -> Dict:
        """Test error recovery mechanisms."""
        result = {"passed": False, "details": {}, "issues": []}
        
        try:
            if not self.backends:
                result["issues"].append("No backends available for recovery testing")
                return result
            
            device = self.active_devices[0]
            backend = self.backends[device]
            
            # Test recovery after failed operation
            try:
                # Force an error
                test_tensor = torch.randn(100, dtype=torch.float32, device=device)
                
                # Try operation with invalid parameters to force an error
                try:
                    backend.broadcast(test_tensor, 999)  # Invalid device
                except:
                    pass  # Expected to fail
                
                # Try a valid operation to test recovery
                backend.broadcast(test_tensor, device)
                
                result["details"]["recovery_successful"] = True
                
            except Exception as e:
                result["details"]["recovery_successful"] = False
                result["issues"].append(f"Error recovery failed: {e}")
            
            # Test fallback recovery
            if hasattr(backend, 'fallback'):
                try:
                    test_tensor = torch.randn(100, dtype=torch.float32, device=device)
                    
                    # Force P2P to fail (if possible)
                    # Then test that fallback works
                    backend.fallback.broadcast(test_tensor, device)
                    
                    result["details"]["fallback_recovery"] = True
                    
                except Exception as e:
                    result["details"]["fallback_recovery"] = False
                    result["issues"].append(f"Fallback recovery failed: {e}")
            
            result["passed"] = len(result["issues"]) == 0
            
        except Exception as e:
            result["issues"].append(f"Error recovery test failed: {e}")
        
        return result
    
    def _test_timeout_handling(self) -> Dict:
        """Test timeout handling."""
        result = {"passed": False, "details": {}, "issues": []}
        
        try:
            # This is a simplified timeout test
            # In practice, you'd implement actual timeout mechanisms
            
            device = self.active_devices[0] if self.active_devices else 0
            
            # Simulate a long operation
            start_time = time.time()
            
            # Create a large tensor that might take time to process
            large_tensor = torch.randn(10*1024*1024, dtype=torch.float32, device=device)
            
            # Check if operation completes within reasonable time
            processing_time = time.time() - start_time
            
            result["details"]["processing_time"] = processing_time
            
            # Allow up to 10 seconds for large tensor creation
            if processing_time < 10.0:
                result["details"]["timeout_not_triggered"] = True
            else:
                result["details"]["timeout_not_triggered"] = False
                result["issues"].append(f"Operation took too long: {processing_time} seconds")
            
            result["passed"] = len(result["issues"]) == 0
            
        except Exception as e:
            result["issues"].append(f"Timeout handling test failed: {e}")
        
        return result
    
    def _generate_error_handling_recommendations(self, test_results: Dict) -> List[str]:
        """Generate error handling recommendations."""
        recommendations = []
        
        # Check invalid parameters
        param_test = test_results.get("invalid_parameters", {})
        if param_test.get("issues"):
            recommendations.append("Invalid parameter handling needs improvement")
        
        # Check error recovery
        recovery_test = test_results.get("error_recovery", {})
        if recovery_test.get("issues"):
            recommendations.append("Error recovery mechanisms need improvement")
        
        # Check timeout handling
        timeout_test = test_results.get("timeout_handling", {})
        if timeout_test.get("issues"):
            recommendations.append("Timeout handling needs improvement")
        
        return recommendations
    
    def _validate_monitoring_integration(self):
        """Validate monitoring integration."""
        print("Validating monitoring integration...")
        
        category_results = {
            "status": "unknown",
            "tests": {},
            "issues": [],
            "recommendations": []
        }
        
        try:
            # Test monitor initialization
            test_result = self._test_monitor_initialization()
            category_results["tests"]["monitor_initialization"] = test_result
            
            # Test operation recording
            test_result = self._test_operation_recording()
            category_results["tests"]["operation_recording"] = test_result
            
            # Test performance tracking
            test_result = self._test_performance_tracking()
            category_results["tests"]["performance_tracking"] = test_result
            
            # Determine overall status
            if all(test.get("passed", False) for test in category_results["tests"].values()):
                category_results["status"] = "passed"
            elif any(test.get("passed", False) for test in category_results["tests"].values()):
                category_results["status"] = "partial"
            else:
                category_results["status"] = "failed"
            
            # Generate recommendations
            category_results["recommendations"] = self._generate_monitoring_recommendations(category_results["tests"])
            
        except Exception as e:
            category_results["status"] = "error"
            category_results["issues"].append(f"Monitoring integration validation error: {e}")
        
        self.validation_results["validation_categories"]["monitoring_integration"] = category_results
        print(f"Monitoring integration validation: {category_results['status']}")
    
    def _test_monitor_initialization(self) -> Dict:
        """Test monitor initialization."""
        result = {"passed": False, "details": {}, "issues": []}
        
        try:
            if not self.monitor:
                result["issues"].append("Monitor not initialized")
                return result
            
            # Check monitor properties
            result["details"]["active_devices"] = self.monitor.active_devices
            result["details"]["num_devices"] = self.monitor.num_devices
            result["details"]["monitoring_level"] = getattr(self.monitor, 'monitoring_level', 'unknown')
            
            # Validate properties
            if self.monitor.active_devices == self.active_devices:
                result["details"]["devices_match"] = True
            else:
                result["details"]["devices_match"] = False
                result["issues"].append("Monitor devices don't match expected devices")
            
            if self.monitor.num_devices == self.num_devices:
                result["details"]["device_count_match"] = True
            else:
                result["details"]["device_count_match"] = False
                result["issues"].append("Monitor device count doesn't match expected")
            
            result["passed"] = len(result["issues"]) == 0
            
        except Exception as e:
            result["issues"].append(f"Monitor initialization test failed: {e}")
        
        return result
    
    def _test_operation_recording(self) -> Dict:
        """Test operation recording."""
        result = {"passed": False, "details": {}, "issues": []}
        
        try:
            if not self.monitor:
                result["issues"].append("Monitor not available for recording test")
                return result
            
            # Record a test operation
            device = self.active_devices[0] if self.active_devices else 0
            tensor = torch.randn(100, dtype=torch.float32, device=device)
            
            start_time = time.time()
            time.sleep(0.01)  # Small delay
            end_time = time.time()
            
            self.monitor.record_operation(
                operation_type="test",
                device_id=device,
                peer_device=None,
                tensor=tensor,
                start_time=start_time,
                end_time=end_time,
                algorithm="test_algorithm",
                success=True
            )
            
            # Check if operation was recorded
            operations = self.monitor.get_operation_history()
            result["details"]["recorded_operations"] = len(operations)
            
            if len(operations) > 0:
                result["details"]["recording_successful"] = True
                
                # Check operation details
                op = operations[0]
                if op.operation_type == "test" and op.device_id == device:
                    result["details"]["operation_details_correct"] = True
                else:
                    result["details"]["operation_details_correct"] = False
                    result["issues"].append("Recorded operation details incorrect")
            else:
                result["details"]["recording_successful"] = False
                result["issues"].append("No operations recorded")
            
            result["passed"] = len(result["issues"]) == 0
            
        except Exception as e:
            result["issues"].append(f"Operation recording test failed: {e}")
        
        return result
    
    def _test_performance_tracking(self) -> Dict:
        """Test performance tracking."""
        result = {"passed": False, "details": {}, "issues": []}
        
        try:
            if not self.monitor:
                result["issues"].append("Monitor not available for performance tracking test")
                return result
            
            # Record multiple operations with different performance
            device = self.active_devices[0] if self.active_devices else 0
            
            for i in range(5):
                tensor = torch.randn(100, dtype=torch.float32, device=device)
                
                start_time = time.time()
                time.sleep(0.01 * (i + 1))  # Varying delay
                end_time = time.time()
                
                self.monitor.record_operation(
                    operation_type="performance_test",
                    device_id=device,
                    peer_device=None,
                    tensor=tensor,
                    start_time=start_time,
                    end_time=end_time,
                    algorithm="test_algorithm",
                    success=True
                )
            
            # Get performance summary
            summary = self.monitor.get_performance_summary()
            result["details"]["performance_summary"] = summary
            
            if isinstance(summary, dict):
                result["details"]["summary_valid"] = True
                
                # Check if summary contains expected keys
                expected_keys = ["monitoring_info", "device_metrics", "operation_stats", "performance_analysis"]
                missing_keys = [key for key in expected_keys if key not in summary]
                
                if not missing_keys:
                    result["details"]["summary_complete"] = True
                else:
                    result["details"]["summary_complete"] = False
                    result["issues"].append(f"Performance summary missing keys: {missing_keys}")
            else:
                result["details"]["summary_valid"] = False
                result["issues"].append("Performance summary not a dictionary")
            
            result["passed"] = len(result["issues"]) == 0
            
        except Exception as e:
            result["issues"].append(f"Performance tracking test failed: {e}")
        
        return result
    
    def _generate_monitoring_recommendations(self, test_results: Dict) -> List[str]:
        """Generate monitoring recommendations."""
        recommendations = []
        
        # Check monitor initialization
        init_test = test_results.get("monitor_initialization", {})
        if init_test.get("issues"):
            recommendations.append("Monitor initialization issues detected - check monitor setup")
        
        # Check operation recording
        recording_test = test_results.get("operation_recording", {})
        if recording_test.get("issues"):
            recommendations.append("Operation recording issues detected - check recording logic")
        
        # Check performance tracking
        tracking_test = test_results.get("performance_tracking", {})
        if tracking_test.get("issues"):
            recommendations.append("Performance tracking issues detected - check tracking implementation")
        
        return recommendations
    
    def _validate_configuration(self):
        """Validate configuration."""
        print("Validating configuration...")
        
        category_results = {
            "status": "unknown",
            "tests": {},
            "issues": [],
            "recommendations": []
        }
        
        try:
            # Test configuration loading
            test_result = self._test_configuration_loading()
            category_results["tests"]["config_loading"] = test_result
            
            # Test configuration updates
            test_result = self._test_configuration_updates()
            category_results["tests"]["config_updates"] = test_result
            
            # Test environment configuration
            test_result = self._test_environment_configuration()
            category_results["tests"]["env_config"] = test_result
            
            # Determine overall status
            if all(test.get("passed", False) for test in category_results["tests"].values()):
                category_results["status"] = "passed"
            elif any(test.get("passed", False) for test in category_results["tests"].values()):
                category_results["status"] = "partial"
            else:
                category_results["status"] = "failed"
            
            # Generate recommendations
            category_results["recommendations"] = self._generate_configuration_recommendations(category_results["tests"])
            
        except Exception as e:
            category_results["status"] = "error"
            category_results["issues"].append(f"Configuration validation error: {e}")
        
        self.validation_results["validation_categories"]["configuration"] = category_results
        print(f"Configuration validation: {category_results['status']}")
    
    def _test_configuration_loading(self) -> Dict:
        """Test configuration loading."""
        result = {"passed": False, "details": {}, "issues": []}
        
        try:
            # Test configuration manager initialization
            config_file = os.path.join(self.output_dir, "test_config.json")
            config_manager = P2PConfigManager(config_file)
            
            result["details"]["config_manager_initialized"] = True
            result["details"]["config_file"] = config_file
            
            # Test getting configuration
            config = config_manager.get_config()
            result["details"]["config_retrieved"] = config is not None
            
            if config:
                result["details"]["config_type"] = type(config).__name__
                
                # Check configuration structure
                if hasattr(config, 'monitoring') and hasattr(config, 'profiling'):
                    result["details"]["config_structure_valid"] = True
                else:
                    result["details"]["config_structure_valid"] = False
                    result["issues"].append("Configuration structure invalid")
            
            result["passed"] = len(result["issues"]) == 0
            
        except Exception as e:
            result["issues"].append(f"Configuration loading test failed: {e}")
        
        return result
    
    def _test_configuration_updates(self) -> Dict:
        """Test configuration updates."""
        result = {"passed": False, "details": {}, "issues": []}
        
        try:
            config_file = os.path.join(self.output_dir, "test_config.json")
            config_manager = P2PConfigManager(config_file)
            
            # Test monitoring config update
            original_monitoring = config_manager.get_config().monitoring.monitoring_level
            config_manager.update_monitoring_config(monitoring_level="detailed")
            
            updated_monitoring = config_manager.get_config().monitoring.monitoring_level
            
            if updated_monitoring == "detailed":
                result["details"]["monitoring_update_successful"] = True
            else:
                result["details"]["monitoring_update_successful"] = False
                result["issues"].append("Monitoring config update failed")
            
            # Test profiling config update
            original_profiling = config_manager.get_config().profiling.profiling_level
            config_manager.update_profiling_config(profiling_level="comprehensive")
            
            updated_profiling = config_manager.get_config().profiling.profiling_level
            
            if updated_profiling == "comprehensive":
                result["details"]["profiling_update_successful"] = True
            else:
                result["details"]["profiling_update_successful"] = False
                result["issues"].append("Profiling config update failed")
            
            # Test config save/load
            config_manager.save_config()
            new_config_manager = P2PConfigManager(config_file)
            
            if new_config_manager.get_config().monitoring.monitoring_level == "detailed":
                result["details"]["save_load_successful"] = True
            else:
                result["details"]["save_load_successful"] = False
                result["issues"].append("Configuration save/load failed")
            
            result["passed"] = len(result["issues"]) == 0
            
        except Exception as e:
            result["issues"].append(f"Configuration updates test failed: {e}")
        
        return result
    
    def _test_environment_configuration(self) -> Dict:
        """Test environment configuration."""
        result = {"passed": False, "details": {}, "issues": []}
        
        try:
            # Set environment variables
            os.environ["P2P_MONITORING_LEVEL"] = "comprehensive"
            os.environ["P2P_ENABLE_MONITORING"] = "true"
            
            # Load configuration from environment
            from exllamav3.util.p2p_config import load_config_from_env
            config_manager = load_config_from_env()
            
            config = config_manager.get_config()
            
            # Check if environment variables were loaded
            if config.monitoring.monitoring_level == "comprehensive":
                result["details"]["env_monitoring_loaded"] = True
            else:
                result["details"]["env_monitoring_loaded"] = False
                result["issues"].append("Environment monitoring level not loaded")
            
            if config.monitoring.enable_monitoring == True:
                result["details"]["env_monitoring_enabled"] = True
            else:
                result["details"]["env_monitoring_enabled"] = False
                result["issues"].append("Environment monitoring enable not loaded")
            
            # Clean up environment variables
            os.environ.pop("P2P_MONITORING_LEVEL", None)
            os.environ.pop("P2P_ENABLE_MONITORING", None)
            
            result["passed"] = len(result["issues"]) == 0
            
        except Exception as e:
            result["issues"].append(f"Environment configuration test failed: {e}")
        
        return result
    
    def _generate_configuration_recommendations(self, test_results: Dict) -> List[str]:
        """Generate configuration recommendations."""
        recommendations = []
        
        # Check config loading
        loading_test = test_results.get("config_loading", {})
        if loading_test.get("issues"):
            recommendations.append("Configuration loading issues detected - check config file format")
        
        # Check config updates
        updates_test = test_results.get("config_updates", {})
        if updates_test.get("issues"):
            recommendations.append("Configuration update issues detected - check update logic")
        
        # Check environment config
        env_test = test_results.get("env_config", {})
        if env_test.get("issues"):
            recommendations.append("Environment configuration issues detected - check env var handling")
        
        return recommendations
    
    def _generate_summary(self):
        """Generate validation summary."""
        categories = self.validation_results["validation_categories"]
        
        # Count passed/failed/partial categories
        passed_count = sum(1 for cat in categories.values() if cat.get("status") == "passed")
        failed_count = sum(1 for cat in categories.values() if cat.get("status") == "failed")
        partial_count = sum(1 for cat in categories.values() if cat.get("status") == "partial")
        error_count = sum(1 for cat in categories.values() if cat.get("status") == "error")
        total_count = len(categories)
        
        # Determine overall status
        if passed_count == total_count:
            overall_status = "passed"
        elif failed_count == 0 and partial_count > 0:
            overall_status = "partial"
        elif error_count > 0:
            overall_status = "error"
        else:
            overall_status = "failed"
        
        # Collect all issues and recommendations
        all_issues = []
        all_recommendations = []
        
        for cat_name, cat_results in categories.items():
            all_issues.extend([f"{cat_name}: {issue}" for issue in cat_results.get("issues", [])])
            all_recommendations.extend([f"{cat_name}: {rec}" for rec in cat_results.get("recommendations", [])])
        
        self.validation_results["overall_status"] = overall_status
        self.validation_results["summary"] = {
            "total_categories": total_count,
            "passed_categories": passed_count,
            "failed_categories": failed_count,
            "partial_categories": partial_count,
            "error_categories": error_count,
            "overall_status": overall_status,
            "total_issues": len(all_issues),
            "total_recommendations": len(all_recommendations),
            "issues": all_issues[:10],  # First 10 issues
            "recommendations": all_recommendations[:10]  # First 10 recommendations
        }
    
    def _save_validation_results(self):
        """Save validation results to file."""
        results_file = os.path.join(self.output_dir, "validation_results.json")
        
        with open(results_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        print(f"Validation results saved to: {results_file}")
    
    def cleanup(self):
        """Clean up resources."""
        try:
            # Close backends
            for device, backend in self.backends.items():
                try:
                    backend.close()
                except Exception:
                    pass
            
            # Close monitor
            if self.monitor:
                try:
                    self.monitor.close()
                except Exception:
                    pass
            
            # Close profiler
            if self.profiler:
                try:
                    # Profiler might not have close method
                    pass
                except Exception:
                    pass
            
            # Close debugger
            if self.debugger:
                try:
                    # Debugger might not have close method
                    pass
                except Exception:
                    pass
        
        except Exception as e:
            print(f"Cleanup error: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="P2P Implementation Validation Tool")
    parser.add_argument("--devices", type=str, help="Comma-separated list of GPU devices to use")
    parser.add_argument("--output-dir", type=str, default="./p2p_validation", 
                       help="Output directory for validation results")
    parser.add_argument("--category", type=str, 
                       choices=["topology", "backend", "operations", "performance", 
                                "memory", "error_handling", "monitoring", "configuration"],
                       help="Run only specific validation category")
    
    args = parser.parse_args()
    
    # Parse devices
    active_devices = None
    if args.devices:
        try:
            active_devices = [int(d.strip()) for d in args.devices.split(",")]
        except ValueError:
            print("Invalid device list")
            return 1
    
    # Create validator
    validator = P2PImplementationValidator(
        active_devices=active_devices,
        output_dir=args.output_dir
    )
    
    try:
        if args.category:
            # Run specific category
            if args.category == "topology":
                validator._validate_topology()
            elif args.category == "backend":
                validator._validate_backend_initialization()
            elif args.category == "operations":
                validator._validate_p2p_operations()
            elif args.category == "performance":
                validator._validate_performance()
            elif args.category == "memory":
                validator._validate_memory_management()
            elif args.category == "error_handling":
                validator._validate_error_handling()
            elif args.category == "monitoring":
                validator._validate_monitoring_integration()
            elif args.category == "configuration":
                validator._validate_configuration()
        else:
            # Run all validation
            validator.validate_all()
        
        # Print summary
        summary = validator.validation_results["summary"]
        print("\n" + "=" * 50)
        print("VALIDATION SUMMARY")
        print("=" * 50)
        print(f"Overall Status: {summary['overall_status'].upper()}")
        print(f"Categories: {summary['total_categories']}")
        print(f"  Passed: {summary['passed_categories']}")
        print(f"  Failed: {summary['failed_categories']}")
        print(f"  Partial: {summary['partial_categories']}")
        print(f"  Error: {summary['error_categories']}")
        print(f"Issues: {summary['total_issues']}")
        print(f"Recommendations: {summary['total_recommendations']}")
        
        if summary["issues"]:
            print("\nTop Issues:")
            for issue in summary["issues"]:
                print(f"  - {issue}")
        
        if summary["recommendations"]:
            print("\nTop Recommendations:")
            for rec in summary["recommendations"]:
                print(f"  - {rec}")
        
        # Return appropriate exit code
        if summary["overall_status"] == "passed":
            return 0
        elif summary["overall_status"] == "partial":
            return 1
        else:
            return 2
    
    finally:
        validator.cleanup()


if __name__ == "__main__":
    sys.exit(main())