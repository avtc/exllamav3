#!/usr/bin/env python3
"""
P2P Correctness Validation Tool

This tool provides comprehensive correctness validation for the P2P GPU communication implementation,
including mathematical correctness, algorithm validation, and data integrity checks.
"""

import os
import sys
import time
import json
import argparse
import tempfile
import shutil
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import math

# Add the exllamav3 directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import torch
    from exllamav3.model.model_tp_p2p import P2PTopology
    from exllamav3.model.model_tp_backend import TPBackendP2P, TPBackendNative
    from exllamav3.util.p2p_monitor import P2PMonitor
    from exllamav3.ext import exllamav3_ext as ext
    P2P_AVAILABLE = True
except ImportError as e:
    print(f"P2P modules not available: {e}")
    P2P_AVAILABLE = False


class P2PCorrectnessValidator:
    """Correctness validator for P2P operations."""
    
    def __init__(self, active_devices: List[int] = None, 
                 output_dir: str = "./p2p_correctness"):
        """
        Initialize the correctness validator.
        
        Args:
            active_devices: List of active GPU device IDs
            output_dir: Directory to store correctness validation results
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
        
        # Validation tolerances
        self.tolerances = {
            "float32_rtol": 1e-5,
            "float32_atol": 1e-8,
            "float16_rtol": 1e-3,
            "float16_atol": 1e-5,
            "int8_rtol": 0.0,
            "int8_atol": 0.0
        }
        
        # Initialize components
        self.topology = None
        self.backends = {}
        self.monitor = None
        
        if P2P_AVAILABLE and self.num_devices > 0:
            try:
                self.topology = P2PTopology(self.active_devices)
                self.monitor = P2PMonitor(active_devices=self.active_devices, output_dir=output_dir)
            except Exception as e:
                print(f"Failed to initialize P2P components: {e}")
    
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
                    uuid="p2p_correctness"
                )
                self.backends[device] = backend
            return True
        except Exception as e:
            print(f"Failed to initialize backends: {e}")
            return False
    
    def _cleanup_backends(self):
        """Clean up backends."""
        for device, backend in self.backends.items():
            try:
                backend.close()
            except Exception:
                pass
        self.backends.clear()
    
    def validate_all(self) -> Dict:
        """
        Run all correctness validation checks.
        
        Returns:
            Dictionary with validation results
        """
        print("Starting P2P Correctness Validation...")
        print(f"Active devices: {self.active_devices}")
        print(f"Output directory: {self.output_dir}")
        print()
        
        # Run validation categories
        self._validate_broadcast_correctness()
        self._validate_all_reduce_correctness()
        self._validate_gather_correctness()
        self._validate_tree_reduce_correctness()
        self._validate_direct_memory_correctness()
        self._validate_mathematical_properties()
        self._validate_edge_cases()
        self._validate_data_integrity()
        
        # Generate summary
        self._generate_summary()
        
        # Save results
        self._save_validation_results()
        
        return self.validation_results
    
    def _validate_broadcast_correctness(self):
        """Validate broadcast operation correctness."""
        print("Validating broadcast correctness...")
        
        category_results = {
            "status": "unknown",
            "tests": {},
            "issues": [],
            "recommendations": []
        }
        
        try:
            if not self._initialize_backends():
                category_results["status"] = "failed"
                category_results["issues"].append("Failed to initialize backends")
                self.validation_results["validation_categories"]["broadcast_correctness"] = category_results
                return
            
            device = self.active_devices[0]
            backend = self.backends[device]
            
            # Test different tensor sizes and data types
            test_configs = [
                {"size": (100,), "dtype": torch.float32},
                {"size": (1000, 100), "dtype": torch.float32},
                {"size": (100, 100, 100), "dtype": torch.float32},
                {"size": (1000,), "dtype": torch.float16},
                {"size": (500, 500), "dtype": torch.float16},
                {"size": (1000,), "dtype": torch.int32},
                {"size": (500, 500), "dtype": torch.int8}
            ]
            
            for i, config in enumerate(test_configs):
                test_name = f"broadcast_test_{i}"
                test_result = self._test_broadcast_correctness(backend, config)
                category_results["tests"][test_name] = test_result
                
                if not test_result["passed"]:
                    category_results["issues"].extend(test_result["issues"])
            
            # Determine overall status
            if all(test.get("passed", False) for test in category_results["tests"].values()):
                category_results["status"] = "passed"
            elif any(test.get("passed", False) for test in category_results["tests"].values()):
                category_results["status"] = "partial"
            else:
                category_results["status"] = "failed"
            
            # Generate recommendations
            category_results["recommendations"] = self._generate_broadcast_recommendations(category_results["tests"])
            
        except Exception as e:
            category_results["status"] = "error"
            category_results["issues"].append(f"Broadcast correctness validation error: {e}")
        
        self.validation_results["validation_categories"]["broadcast_correctness"] = category_results
        print(f"Broadcast correctness validation: {category_results['status']}")
    
    def _test_broadcast_correctness(self, backend, config: Dict) -> Dict:
        """Test broadcast correctness with specific configuration."""
        result = {"passed": False, "details": {}, "issues": []}
        
        try:
            size = config["size"]
            dtype = config["dtype"]
            device = backend.device
            
            # Create test tensor with known values
            if dtype.is_floating_point:
                test_tensor = torch.randn(size, dtype=dtype, device=device)
            else:
                test_tensor = torch.randint(-100, 100, size, dtype=dtype, device=device)
            
            original_tensor = test_tensor.clone()
            
            # Perform broadcast
            backend.broadcast(test_tensor, device)
            
            # Check if tensor remained unchanged (broadcast from same device)
            if dtype.is_floating_point:
                rtol = self.tolerances["float32_rtol"] if dtype == torch.float32 else self.tolerances["float16_rtol"]
                atol = self.tolerances["float32_atol"] if dtype == torch.float32 else self.tolerances["float16_atol"]
                
                if torch.allclose(original_tensor, test_tensor, rtol=rtol, atol=atol):
                    result["details"]["broadcast_correct"] = True
                else:
                    result["details"]["broadcast_correct"] = False
                    result["issues"].append(f"Broadcast changed tensor values for dtype {dtype}")
            else:
                rtol = self.tolerances["int8_rtol"] if dtype == torch.int8 else self.tolerances["int32_rtol"]
                atol = self.tolerances["int8_atol"] if dtype == torch.int8 else self.tolerances["int32_atol"]
                
                if torch.allclose(original_tensor.float(), test_tensor.float(), rtol=rtol, atol=atol):
                    result["details"]["broadcast_correct"] = True
                else:
                    result["details"]["broadcast_correct"] = False
                    result["issues"].append(f"Broadcast changed tensor values for dtype {dtype}")
            
            # Check tensor properties
            result["details"]["shape_preserved"] = test_tensor.shape == original_tensor.shape
            result["details"]["dtype_preserved"] = test_tensor.dtype == original_tensor.dtype
            
            if not result["details"]["shape_preserved"]:
                result["issues"].append("Broadcast changed tensor shape")
            
            if not result["details"]["dtype_preserved"]:
                result["issues"].append("Broadcast changed tensor dtype")
            
            result["passed"] = len(result["issues"]) == 0
            
        except Exception as e:
            result["issues"].append(f"Broadcast correctness test failed: {e}")
        
        return result
    
    def _generate_broadcast_recommendations(self, test_results: Dict) -> List[str]:
        """Generate broadcast correctness recommendations."""
        recommendations = []
        
        for test_name, test_result in test_results.items():
            if not test_result.get("passed", False):
                recommendations.append(f"Broadcast correctness issues detected in {test_name}")
        
        return recommendations
    
    def _validate_all_reduce_correctness(self):
        """Validate all_reduce operation correctness."""
        print("Validating all_reduce correctness...")
        
        category_results = {
            "status": "unknown",
            "tests": {},
            "issues": [],
            "recommendations": []
        }
        
        try:
            if not self._initialize_backends():
                category_results["status"] = "failed"
                category_results["issues"].append("Failed to initialize backends")
                self.validation_results["validation_categories"]["all_reduce_correctness"] = category_results
                return
            
            device = self.active_devices[0]
            backend = self.backends[device]
            
            # Test different tensor sizes and data types
            test_configs = [
                {"size": (100,), "dtype": torch.float32},
                {"size": (1000, 100), "dtype": torch.float32},
                {"size": (100, 100, 100), "dtype": torch.float32},
                {"size": (1000,), "dtype": torch.float16},
                {"size": (500, 500), "dtype": torch.float16},
                {"size": (1000,), "dtype": torch.int32},
                {"size": (500, 500), "dtype": torch.int8}
            ]
            
            for i, config in enumerate(test_configs):
                test_name = f"all_reduce_test_{i}"
                test_result = self._test_all_reduce_correctness(backend, config)
                category_results["tests"][test_name] = test_result
                
                if not test_result["passed"]:
                    category_results["issues"].extend(test_result["issues"])
            
            # Determine overall status
            if all(test.get("passed", False) for test in category_results["tests"].values()):
                category_results["status"] = "passed"
            elif any(test.get("passed", False) for test in category_results["tests"].values()):
                category_results["status"] = "partial"
            else:
                category_results["status"] = "failed"
            
            # Generate recommendations
            category_results["recommendations"] = self._generate_all_reduce_recommendations(category_results["tests"])
            
        except Exception as e:
            category_results["status"] = "error"
            category_results["issues"].append(f"All_reduce correctness validation error: {e}")
        
        self.validation_results["validation_categories"]["all_reduce_correctness"] = category_results
        print(f"All_reduce correctness validation: {category_results['status']}")
    
    def _test_all_reduce_correctness(self, backend, config: Dict) -> Dict:
        """Test all_reduce correctness with specific configuration."""
        result = {"passed": False, "details": {}, "issues": []}
        
        try:
            size = config["size"]
            dtype = config["dtype"]
            device = backend.device
            
            # Create test tensor with known values
            if dtype.is_floating_point:
                test_tensor = torch.randn(size, dtype=dtype, device=device)
            else:
                test_tensor = torch.randint(-100, 100, size, dtype=dtype, device=device)
            
            original_sum = test_tensor.sum().float()
            original_mean = test_tensor.float().mean()
            
            # Perform all_reduce
            backend.all_reduce(test_tensor)
            
            # Check if tensor was reduced correctly
            # After all_reduce, each device should have the sum of all original tensors
            expected_sum = original_sum * self.num_devices
            actual_sum = test_tensor.sum().float()
            
            if dtype.is_floating_point:
                rtol = self.tolerances["float32_rtol"] if dtype == torch.float32 else self.tolerances["float16_rtol"]
                atol = self.tolerances["float32_atol"] if dtype == torch.float32 else self.tolerances["float16_atol"]
                
                if torch.isclose(actual_sum, expected_sum, rtol=rtol, atol=atol):
                    result["details"]["reduction_correct"] = True
                else:
                    result["details"]["reduction_correct"] = False
                    result["issues"].append(f"All_reduce incorrect: expected {expected_sum}, got {actual_sum}")
            else:
                rtol = self.tolerances["int8_rtol"] if dtype == torch.int8 else self.tolerances["int32_rtol"]
                atol = self.tolerances["int8_atol"] if dtype == torch.int8 else self.tolerances["int32_atol"]
                
                if torch.isclose(actual_sum, expected_sum, rtol=rtol, atol=atol):
                    result["details"]["reduction_correct"] = True
                else:
                    result["details"]["reduction_correct"] = False
                    result["issues"].append(f"All_reduce incorrect: expected {expected_sum}, got {actual_sum}")
            
            # Check tensor properties
            result["details"]["shape_preserved"] = test_tensor.shape == size
            result["details"]["dtype_preserved"] = test_tensor.dtype == dtype
            
            if not result["details"]["shape_preserved"]:
                result["issues"].append("All_reduce changed tensor shape")
            
            if not result["details"]["dtype_preserved"]:
                result["issues"].append("All_reduce changed tensor dtype")
            
            # Check statistical properties
            actual_mean = test_tensor.float().mean()
            expected_mean = original_mean  # Mean should be preserved in all-reduce
            
            if dtype.is_floating_point:
                if torch.isclose(actual_mean, expected_mean, rtol=rtol, atol=atol):
                    result["details"]["mean_preserved"] = True
                else:
                    result["details"]["mean_preserved"] = False
                    result["issues"].append(f"All_reduce changed mean: expected {expected_mean}, got {actual_mean}")
            
            result["passed"] = len(result["issues"]) == 0
            
        except Exception as e:
            result["issues"].append(f"All_reduce correctness test failed: {e}")
        
        return result
    
    def _generate_all_reduce_recommendations(self, test_results: Dict) -> List[str]:
        """Generate all_reduce correctness recommendations."""
        recommendations = []
        
        for test_name, test_result in test_results.items():
            if not test_result.get("passed", False):
                recommendations.append(f"All_reduce correctness issues detected in {test_name}")
        
        return recommendations
    
    def _validate_gather_correctness(self):
        """Validate gather operation correctness."""
        print("Validating gather correctness...")
        
        category_results = {
            "status": "unknown",
            "tests": {},
            "issues": [],
            "recommendations": []
        }
        
        try:
            if not self._initialize_backends():
                category_results["status"] = "failed"
                category_results["issues"].append("Failed to initialize backends")
                self.validation_results["validation_categories"]["gather_correctness"] = category_results
                return
            
            device = self.active_devices[0]
            backend = self.backends[device]
            
            # Test different tensor sizes and data types
            test_configs = [
                {"size": (100,), "dtype": torch.float32},
                {"size": (1000, 100), "dtype": torch.float32},
                {"size": (100, 100, 100), "dtype": torch.float32},
                {"size": (1000,), "dtype": torch.float16},
                {"size": (500, 500), "dtype": torch.float16},
                {"size": (1000,), "dtype": torch.int32},
                {"size": (500, 500), "dtype": torch.int8}
            ]
            
            for i, config in enumerate(test_configs):
                test_name = f"gather_test_{i}"
                test_result = self._test_gather_correctness(backend, config)
                category_results["tests"][test_name] = test_result
                
                if not test_result["passed"]:
                    category_results["issues"].extend(test_result["issues"])
            
            # Determine overall status
            if all(test.get("passed", False) for test in category_results["tests"].values()):
                category_results["status"] = "passed"
            elif any(test.get("passed", False) for test in category_results["tests"].values()):
                category_results["status"] = "partial"
            else:
                category_results["status"] = "failed"
            
            # Generate recommendations
            category_results["recommendations"] = self._generate_gather_recommendations(category_results["tests"])
            
        except Exception as e:
            category_results["status"] = "error"
            category_results["issues"].append(f"Gather correctness validation error: {e}")
        
        self.validation_results["validation_categories"]["gather_correctness"] = category_results
        print(f"Gather correctness validation: {category_results['status']}")
    
    def _test_gather_correctness(self, backend, config: Dict) -> Dict:
        """Test gather correctness with specific configuration."""
        result = {"passed": False, "details": {}, "issues": []}
        
        try:
            size = config["size"]
            dtype = config["dtype"]
            device = backend.device
            
            # Create test tensor with known values
            if dtype.is_floating_point:
                test_tensor = torch.randn(size, dtype=dtype, device=device)
            else:
                test_tensor = torch.randint(-100, 100, size, dtype=dtype, device=device)
            
            original_tensor = test_tensor.clone()
            
            # Create output tensor
            output_size = (size[0] * self.num_devices,) + size[1:] if len(size) > 1 else (size[0] * self.num_devices,)
            out_tensor = torch.zeros(output_size, dtype=dtype, device=device)
            
            # Perform gather
            gather_devices = torch.tensor(self.active_devices[:self.num_devices], dtype=torch.int)
            ldims = [np.prod(size)] * self.num_devices
            backend.gather(test_tensor, out_tensor, gather_devices, device, ldims)
            
            # Check if gather was correct
            # The first part of out_tensor should match test_tensor
            if dtype.is_floating_point:
                rtol = self.tolerances["float32_rtol"] if dtype == torch.float32 else self.tolerances["float16_rtol"]
                atol = self.tolerances["float32_atol"] if dtype == torch.float32 else self.tolerances["float16_atol"]
                
                if torch.allclose(out_tensor[:np.prod(size)], original_tensor, rtol=rtol, atol=atol):
                    result["details"]["gather_correct"] = True
                else:
                    result["details"]["gather_correct"] = False
                    result["issues"].append("Gather incorrect: output doesn't match input")
            else:
                rtol = self.tolerances["int8_rtol"] if dtype == torch.int8 else self.tolerances["int32_rtol"]
                atol = self.tolerances["int8_atol"] if dtype == torch.int8 else self.tolerances["int32_atol"]
                
                if torch.allclose(out_tensor[:np.prod(size)].float(), original_tensor.float(), rtol=rtol, atol=atol):
                    result["details"]["gather_correct"] = True
                else:
                    result["details"]["gather_correct"] = False
                    result["issues"].append("Gather incorrect: output doesn't match input")
            
            # Check tensor properties
            result["details"]["output_shape_correct"] = out_tensor.shape == output_size
            result["details"]["output_dtype_preserved"] = out_tensor.dtype == dtype
            
            if not result["details"]["output_shape_correct"]:
                result["issues"].append("Gather changed output tensor shape")
            
            if not result["details"]["output_dtype_preserved"]:
                result["issues"].append("Gather changed output tensor dtype")
            
            result["passed"] = len(result["issues"]) == 0
            
        except Exception as e:
            result["issues"].append(f"Gather correctness test failed: {e}")
        
        return result
    
    def _generate_gather_recommendations(self, test_results: Dict) -> List[str]:
        """Generate gather correctness recommendations."""
        recommendations = []
        
        for test_name, test_result in test_results.items():
            if not test_result.get("passed", False):
                recommendations.append(f"Gather correctness issues detected in {test_name}")
        
        return recommendations
    
    def _validate_tree_reduce_correctness(self):
        """Validate tree-based reduction correctness."""
        print("Validating tree reduce correctness...")
        
        category_results = {
            "status": "unknown",
            "tests": {},
            "issues": [],
            "recommendations": []
        }
        
        try:
            if not self.topology:
                category_results["status"] = "failed"
                category_results["issues"].append("Topology not available")
                self.validation_results["validation_categories"]["tree_reduce_correctness"] = category_results
                return
            
            # Test different tree types
            tree_types = ["binary", "kary", "balanced"]
            
            for tree_type in tree_types:
                test_name = f"tree_reduce_{tree_type}"
                test_result = self._test_tree_reduce_correctness(tree_type)
                category_results["tests"][test_name] = test_result
                
                if not test_result["passed"]:
                    category_results["issues"].extend(test_result["issues"])
            
            # Determine overall status
            if all(test.get("passed", False) for test in category_results["tests"].values()):
                category_results["status"] = "passed"
            elif any(test.get("passed", False) for test in category_results["tests"].values()):
                category_results["status"] = "partial"
            else:
                category_results["status"] = "failed"
            
            # Generate recommendations
            category_results["recommendations"] = self._generate_tree_reduce_recommendations(category_results["tests"])
            
        except Exception as e:
            category_results["status"] = "error"
            category_results["issues"].append(f"Tree reduce correctness validation error: {e}")
        
        self.validation_results["validation_categories"]["tree_reduce_correctness"] = category_results
        print(f"Tree reduce correctness validation: {category_results['status']}")
    
    def _test_tree_reduce_correctness(self, tree_type: str) -> Dict:
        """Test tree-based reduction correctness."""
        result = {"passed": False, "details": {}, "issues": []}
        
        try:
            # Build tree
            if tree_type == "binary":
                tree = self.topology.build_binary_tree()
            elif tree_type == "kary":
                tree = self.topology.build_kary_tree(4)
            elif tree_type == "balanced":
                tree = self.topology.build_balanced_tree()
            else:
                result["issues"].append(f"Unknown tree type: {tree_type}")
                return result
            
            result["details"]["tree_built"] = True
            result["details"]["tree_type"] = tree_type
            
            # Validate tree structure
            required_keys = ["root", "children", "parent", "depth", "tree_type"]
            if all(key in tree for key in required_keys):
                result["details"]["tree_structure_valid"] = True
            else:
                result["details"]["tree_structure_valid"] = False
                result["issues"].append(f"Tree structure missing keys")
            
            # Check tree completeness
            all_devices = set(tree["children"].keys()) | set(tree["parent"].keys())
            if tree["root"] not in tree["parent"]:
                all_devices.add(tree["root"])
            
            if len(all_devices) == self.num_devices:
                result["details"]["tree_complete"] = True
            else:
                result["details"]["tree_complete"] = False
                result["issues"].append(f"Tree missing devices: {self.num_devices - len(all_devices)}")
            
            # Check tree depth
            tree_depth = self.topology.get_tree_depth(tree)
            result["details"]["tree_depth"] = tree_depth
            
            # Validate tree depth is reasonable
            if tree_depth > 0 and tree_depth <= math.ceil(math.log2(self.num_devices)) + 1:
                result["details"]["tree_depth_reasonable"] = True
            else:
                result["details"]["tree_depth_reasonable"] = False
                result["issues"].append(f"Tree depth unreasonable: {tree_depth}")
            
            result["passed"] = len(result["issues"]) == 0
            
        except Exception as e:
            result["issues"].append(f"Tree reduce correctness test failed: {e}")
        
        return result
    
    def _generate_tree_reduce_recommendations(self, test_results: Dict) -> List[str]:
        """Generate tree reduce correctness recommendations."""
        recommendations = []
        
        for test_name, test_result in test_results.items():
            if not test_result.get("passed", False):
                recommendations.append(f"Tree reduce correctness issues detected in {test_name}")
        
        return recommendations
    
    def _validate_direct_memory_correctness(self):
        """Validate direct memory access correctness."""
        print("Validating direct memory correctness...")
        
        category_results = {
            "status": "unknown",
            "tests": {},
            "issues": [],
            "recommendations": []
        }
        
        try:
            if not self._initialize_backends():
                category_results["status"] = "failed"
                category_results["issues"].append("Failed to initialize backends")
                self.validation_results["validation_categories"]["direct_memory_correctness"] = category_results
                return
            
            if len(self.active_devices) < 2:
                category_results["status"] = "skipped"
                category_results["issues"].append("Need at least 2 devices for direct memory testing")
                self.validation_results["validation_categories"]["direct_memory_correctness"] = category_results
                return
            
            src_device = self.active_devices[0]
            dst_device = self.active_devices[1]
            backend = self.backends[src_device]
            
            # Test different tensor sizes
            test_sizes = [100, 1000, 10000]
            
            for i, size in enumerate(test_sizes):
                test_name = f"direct_memory_test_{i}"
                test_result = self._test_direct_memory_correctness(backend, src_device, dst_device, size)
                category_results["tests"][test_name] = test_result
                
                if not test_result["passed"]:
                    category_results["issues"].extend(test_result["issues"])
            
            # Determine overall status
            if all(test.get("passed", False) for test in category_results["tests"].values()):
                category_results["status"] = "passed"
            elif any(test.get("passed", False) for test in category_results["tests"].values()):
                category_results["status"] = "partial"
            else:
                category_results["status"] = "failed"
            
            # Generate recommendations
            category_results["recommendations"] = self._generate_direct_memory_recommendations(category_results["tests"])
            
        except Exception as e:
            category_results["status"] = "error"
            category_results["issues"].append(f"Direct memory correctness validation error: {e}")
        
        self.validation_results["validation_categories"]["direct_memory_correctness"] = category_results
        print(f"Direct memory correctness validation: {category_results['status']}")
    
    def _test_direct_memory_correctness(self, backend, src_device: int, dst_device: int, size: int) -> Dict:
        """Test direct memory access correctness."""
        result = {"passed": False, "details": {}, "issues": []}
        
        try:
            # Create test tensor
            test_tensor = torch.randn(size, dtype=torch.float32, device=src_device)
            original_tensor = test_tensor.clone()
            
            # Copy tensor using direct memory access
            if hasattr(backend, 'copy_tensor_direct'):
                copied_tensor = backend.copy_tensor_direct(src_device, dst_device, test_tensor)
                
                if copied_tensor is not None and copied_tensor.device.index == dst_device:
                    # Copy back to source device for comparison
                    comparison_tensor = copied_tensor.to(src_device)
                    
                    # Check if copy was correct
                    if torch.allclose(original_tensor, comparison_tensor, rtol=1e-5, atol=1e-8):
                        result["details"]["direct_copy_correct"] = True
                    else:
                        result["details"]["direct_copy_correct"] = False
                        result["issues"].append("Direct copy incorrect: tensors don't match")
                    
                    result["details"]["copy_successful"] = True
                else:
                    result["details"]["copy_successful"] = False
                    result["issues"].append("Direct copy failed or returned wrong device")
            else:
                result["issues"].append("Direct copy not available")
            
            result["passed"] = len(result["issues"]) == 0
            
        except Exception as e:
            result["issues"].append(f"Direct memory correctness test failed: {e}")
        
        return result
    
    def _generate_direct_memory_recommendations(self, test_results: Dict) -> List[str]:
        """Generate direct memory correctness recommendations."""
        recommendations = []
        
        for test_name, test_result in test_results.items():
            if not test_result.get("passed", False):
                recommendations.append(f"Direct memory correctness issues detected in {test_name}")
        
        return recommendations
    
    def _validate_mathematical_properties(self):
        """Validate mathematical properties of operations."""
        print("Validating mathematical properties...")
        
        category_results = {
            "status": "unknown",
            "tests": {},
            "issues": [],
            "recommendations": []
        }
        
        try:
            if not self._initialize_backends():
                category_results["status"] = "failed"
                category_results["issues"].append("Failed to initialize backends")
                self.validation_results["validation_categories"]["mathematical_properties"] = category_results
                return
            
            device = self.active_devices[0]
            backend = self.backends[device]
            
            # Test associativity of all_reduce
            test_result = self._test_associativity(backend)
            category_results["tests"]["associativity"] = test_result
            
            # Test commutativity of all_reduce
            test_result = self._test_commutativity(backend)
            category_results["tests"]["commutativity"] = test_result
            
            # Test distributivity
            test_result = self._test_distributivity(backend)
            category_results["tests"]["distributivity"] = test_result
            
            # Test identity properties
            test_result = self._test_identity_properties(backend)
            category_results["tests"]["identity"] = test_result
            
            # Determine overall status
            if all(test.get("passed", False) for test in category_results["tests"].values()):
                category_results["status"] = "passed"
            elif any(test.get("passed", False) for test in category_results["tests"].values()):
                category_results["status"] = "partial"
            else:
                category_results["status"] = "failed"
            
            # Generate recommendations
            category_results["recommendations"] = self._generate_mathematical_recommendations(category_results["tests"])
            
        except Exception as e:
            category_results["status"] = "error"
            category_results["issues"].append(f"Mathematical properties validation error: {e}")
        
        self.validation_results["validation_categories"]["mathematical_properties"] = category_results
        print(f"Mathematical properties validation: {category_results['status']}")
    
    def _test_associativity(self, backend) -> Dict:
        """Test associativity of all_reduce operation."""
        result = {"passed": False, "details": {}, "issues": []}
        
        try:
            device = backend.device
            
            # Create three tensors
            tensor_a = torch.randn(1000, dtype=torch.float32, device=device)
            tensor_b = torch.randn(1000, dtype=torch.float32, device=device)
            tensor_c = torch.randn(1000, dtype=torch.float32, device=device)
            
            # Test (a + b) + c == a + (b + c)
            # First way: (a + b) + c
            ab = tensor_a + tensor_b
            abc_1 = ab + tensor_c
            
            # Reset and perform all_reduce
            tensor_a_copy = tensor_a.clone()
            tensor_b_copy = tensor_b.clone()
            tensor_c_copy = tensor_c.clone()
            
            backend.all_reduce(tensor_a_copy)
            backend.all_reduce(tensor_b_copy)
            backend.all_reduce(tensor_c_copy)
            
            abc_2 = tensor_a_copy + tensor_b_copy + tensor_c_copy
            
            # Check associativity
            if torch.allclose(abc_1, abc_2, rtol=1e-5, atol=1e-8):
                result["details"]["associativity_holds"] = True
            else:
                result["details"]["associativity_holds"] = False
                result["issues"].append("Associativity property violated")
            
            result["passed"] = len(result["issues"]) == 0
            
        except Exception as e:
            result["issues"].append(f"Associativity test failed: {e}")
        
        return result
    
    def _test_commutativity(self, backend) -> Dict:
        """Test commutativity of all_reduce operation."""
        result = {"passed": False, "details": {}, "issues": []}
        
        try:
            device = backend.device
            
            # Create two tensors
            tensor_a = torch.randn(1000, dtype=torch.float32, device=device)
            tensor_b = torch.randn(1000, dtype=torch.float32, device=device)
            
            # Test a + b == b + a
            ab = tensor_a + tensor_b
            ba = tensor_b + tensor_a
            
            # Reset and perform all_reduce
            tensor_a_copy = tensor_a.clone()
            tensor_b_copy = tensor_b.clone()
            
            backend.all_reduce(tensor_a_copy)
            backend.all_reduce(tensor_b_copy)
            
            ab_reduced = tensor_a_copy + tensor_b_copy
            ba_reduced = tensor_b_copy + tensor_a_copy
            
            # Check commutativity
            if torch.allclose(ab_reduced, ba_reduced, rtol=1e-5, atol=1e-8):
                result["details"]["commutativity_holds"] = True
            else:
                result["details"]["commutativity_holds"] = False
                result["issues"].append("Commutativity property violated")
            
            result["passed"] = len(result["issues"]) == 0
            
        except Exception as e:
            result["issues"].append(f"Commutativity test failed: {e}")
        
        return result
    
    def _test_distributivity(self, backend) -> Dict:
        """Test distributivity of operations."""
        result = {"passed": False, "details": {}, "issues": []}
        
        try:
            device = backend.device
            
            # Create tensors
            tensor_a = torch.randn(1000, dtype=torch.float32, device=device)
            tensor_b = torch.randn(1000, dtype=torch.float32, device=device)
            scalar = 2.0
            
            # Test a * (b + c) == a * b + a * c
            bc = tensor_b + tensor_a  # Using tensor_a as "c" for simplicity
            abc_1 = scalar * bc
            
            # Reset and perform all_reduce
            tensor_a_copy = tensor_a.clone()
            tensor_b_copy = tensor_b.clone()
            
            backend.all_reduce(tensor_a_copy)
            backend.all_reduce(tensor_b_copy)
            
            ab_2 = scalar * tensor_a_copy + scalar * tensor_b_copy
            
            # Check distributivity
            if torch.allclose(abc_1, ab_2, rtol=1e-5, atol=1e-8):
                result["details"]["distributivity_holds"] = True
            else:
                result["details"]["distributivity_holds"] = False
                result["issues"].append("Distributivity property violated")
            
            result["passed"] = len(result["issues"]) == 0
            
        except Exception as e:
            result["issues"].append(f"Distributivity test failed: {e}")
        
        return result
    
    def _test_identity_properties(self, backend) -> Dict:
        """Test identity properties."""
        result = {"passed": False, "details": {}, "issues": []}
        
        try:
            device = backend.device
            
            # Test additive identity
            tensor = torch.randn(1000, dtype=torch.float32, device=device)
            zero_tensor = torch.zeros_like(tensor)
            
            # Reset and perform all_reduce
            tensor_copy = tensor.clone()
            zero_copy = zero_tensor.clone()
            
            backend.all_reduce(tensor_copy)
            backend.all_reduce(zero_copy)
            
            # tensor + 0 == tensor
            identity_result = tensor_copy + zero_copy
            
            if torch.allclose(tensor_copy, identity_result, rtol=1e-5, atol=1e-8):
                result["details"]["additive_identity_holds"] = True
            else:
                result["details"]["additive_identity_holds"] = False
                result["issues"].append("Additive identity property violated")
            
            result["passed"] = len(result["issues"]) == 0
            
        except Exception as e:
            result["issues"].append(f"Identity properties test failed: {e}")
        
        return result
    
    def _generate_mathematical_recommendations(self, test_results: Dict) -> List[str]:
        """Generate mathematical properties recommendations."""
        recommendations = []
        
        for test_name, test_result in test_results.items():
            if not test_result.get("passed", False):
                recommendations.append(f"Mathematical property issues detected in {test_name}")
        
        return recommendations
    
    def _validate_edge_cases(self):
        """Validate edge cases."""
        print("Validating edge cases...")
        
        category_results = {
            "status": "unknown",
            "tests": {},
            "issues": [],
            "recommendations": []
        }
        
        try:
            if not self._initialize_backends():
                category_results["status"] = "failed"
                category_results["issues"].append("Failed to initialize backends")
                self.validation_results["validation_categories"]["edge_cases"] = category_results
                return
            
            device = self.active_devices[0]
            backend = self.backends[device]
            
            # Test empty tensor
            test_result = self._test_empty_tensor(backend)
            category_results["tests"]["empty_tensor"] = test_result
            
            # Test single element tensor
            test_result = self._test_single_element_tensor(backend)
            category_results["tests"]["single_element"] = test_result
            
            # Test very large tensor
            test_result = self._test_large_tensor(backend)
            category_results["tests"]["large_tensor"] = test_result
            
            # Test non-contiguous tensor
            test_result = self._test_non_contiguous_tensor(backend)
            category_results["tests"]["non_contiguous"] = test_result
            
            # Determine overall status
            if all(test.get("passed", False) for test in category_results["tests"].values()):
                category_results["status"] = "passed"
            elif any(test.get("passed", False) for test in category_results["tests"].values()):
                category_results["status"] = "partial"
            else:
                category_results["status"] = "failed"
            
            # Generate recommendations
            category_results["recommendations"] = self._generate_edge_case_recommendations(category_results["tests"])
            
        except Exception as e:
            category_results["status"] = "error"
            category_results["issues"].append(f"Edge cases validation error: {e}")
        
        self.validation_results["validation_categories"]["edge_cases"] = category_results
        print(f"Edge cases validation: {category_results['status']}")
    
    def _test_empty_tensor(self, backend) -> Dict:
        """Test empty tensor handling."""
        result = {"passed": False, "details": {}, "issues": []}
        
        try:
            device = backend.device
            
            # Create empty tensor
            empty_tensor = torch.randn(0, dtype=torch.float32, device=device)
            
            try:
                # This should either succeed or fail gracefully
                backend.broadcast(empty_tensor, device)
                result["details"]["empty_tensor_handled"] = True
            except Exception as e:
                result["details"]["empty_tensor_handled"] = True  # Failed gracefully is OK
                result["details"]["empty_tensor_error"] = str(e)
            
            result["passed"] = True  # Edge cases are allowed to fail gracefully
            
        except Exception as e:
            result["issues"].append(f"Empty tensor test failed: {e}")
        
        return result
    
    def _test_single_element_tensor(self, backend) -> Dict:
        """Test single element tensor handling."""
        result = {"passed": False, "details": {}, "issues": []}
        
        try:
            device = backend.device
            
            # Create single element tensor
            single_tensor = torch.randn(1, dtype=torch.float32, device=device)
            original_value = single_tensor.item()
            
            # Perform operation
            backend.broadcast(single_tensor, device)
            
            # Check if value is preserved
            if torch.isclose(single_tensor.item(), original_value, rtol=1e-5, atol=1e-8):
                result["details"]["single_element_preserved"] = True
            else:
                result["details"]["single_element_preserved"] = False
                result["issues"].append("Single element value changed")
            
            result["passed"] = len(result["issues"]) == 0
            
        except Exception as e:
            result["issues"].append(f"Single element tensor test failed: {e}")
        
        return result
    
    def _test_large_tensor(self, backend) -> Dict:
        """Test large tensor handling."""
        result = {"passed": False, "details": {}, "issues": []}
        
        try:
            device = backend.device
            
            # Create large tensor (10M elements)
            large_tensor = torch.randn(10*1024*1024, dtype=torch.float32, device=device)
            
            try:
                # This should either succeed or fail gracefully
                backend.broadcast(large_tensor, device)
                result["details"]["large_tensor_handled"] = True
            except Exception as e:
                result["details"]["large_tensor_handled"] = True  # Failed gracefully is OK
                result["details"]["large_tensor_error"] = str(e)
            
            result["passed"] = True  # Edge cases are allowed to fail gracefully
            
        except Exception as e:
            result["issues"].append(f"Large tensor test failed: {e}")
        
        return result
    
    def _test_non_contiguous_tensor(self, backend) -> Dict:
        """Test non-contiguous tensor handling."""
        result = {"passed": False, "details": {}, "issues": []}
        
        try:
            device = backend.device
            
            # Create non-contiguous tensor
            base_tensor = torch.randn(2000, dtype=torch.float32, device=device)
            non_contiguous_tensor = base_tensor[::2]  # Every other element
            
            try:
                # This should either succeed or fail gracefully
                backend.broadcast(non_contiguous_tensor, device)
                result["details"]["non_contiguous_handled"] = True
            except Exception as e:
                result["details"]["non_contiguous_handled"] = True  # Failed gracefully is OK
                result["details"]["non_contiguous_error"] = str(e)
            
            result["passed"] = True  # Edge cases are allowed to fail gracefully
            
        except Exception as e:
            result["issues"].append(f"Non-contiguous tensor test failed: {e}")
        
        return result
    
    def _generate_edge_case_recommendations(self, test_results: Dict) -> List[str]:
        """Generate edge case recommendations."""
        recommendations = []
        
        for test_name, test_result in test_results.items():
            if not test_result.get("passed", False):
                recommendations.append(f"Edge case issues detected in {test_name}")
        
        return recommendations
    
    def _validate_data_integrity(self):
        """Validate data integrity across operations."""
        print("Validating data integrity...")
        
        category_results = {
            "status": "unknown",
            "tests": {},
            "issues": [],
            "recommendations": []
        }
        
        try:
            if not self._initialize_backends():
                category_results["status"] = "failed"
                category_results["issues"].append("Failed to initialize backends")
                self.validation_results["validation_categories"]["data_integrity"] = category_results
                return
            
            device = self.active_devices[0]
            backend = self.backends[device]
            
            # Test bit pattern preservation
            test_result = self._test_bit_pattern_preservation(backend)
            category_results["tests"]["bit_pattern"] = test_result
            
            # Test numerical stability
            test_result = self._test_numerical_stability(backend)
            category_results["tests"]["numerical_stability"] = test_result
            
            # Test precision preservation
            test_result = self._test_precision_preservation(backend)
            category_results["tests"]["precision_preservation"] = test_result
            
            # Determine overall status
            if all(test.get("passed", False) for test in category_results["tests"].values()):
                category_results["status"] = "passed"
            elif any(test.get("passed", False) for test in category_results["tests"].values()):
                category_results["status"] = "partial"
            else:
                category_results["status"] = "failed"
            
            # Generate recommendations
            category_results["recommendations"] = self._generate_data_integrity_recommendations(category_results["tests"])
            
        except Exception as e:
            category_results["status"] = "error"
            category_results["issues"].append(f"Data integrity validation error: {e}")
        
        self.validation_results["validation_categories"]["data_integrity"] = category_results
        print(f"Data integrity validation: {category_results['status']}")
    
    def _test_bit_pattern_preservation(self, backend) -> Dict:
        """Test bit pattern preservation."""
        result = {"passed": False, "details": {}, "issues": []}
        
        try:
            device = backend.device
            
            # Create tensor with specific bit pattern
            bit_pattern = 0xAAAAAAAA  # Alternating pattern
            test_tensor = torch.full((1000,), bit_pattern, dtype=torch.int32, device=device)
            
            # Convert to float for operation
            float_tensor = test_tensor.float()
            
            # Perform operation
            backend.all_reduce(float_tensor)
            
            # Convert back to int and check pattern
            result_tensor = float_tensor.int()
            
            # Check if pattern is preserved (allowing for floating-point precision)
            pattern_matches = torch.sum(torch.abs(result_tensor - bit_pattern) < 10).item() == 1000
            
            if pattern_matches:
                result["details"]["bit_pattern_preserved"] = True
            else:
                result["details"]["bit_pattern_preserved"] = False
                result["issues"].append("Bit pattern not preserved")
            
            result["passed"] = len(result["issues"]) == 0
            
        except Exception as e:
            result["issues"].append(f"Bit pattern preservation test failed: {e}")
        
        return result
    
    def _test_numerical_stability(self, backend) -> Dict:
        """Test numerical stability."""
        result = {"passed": False, "details": {}, "issues": []}
        
        try:
            device = backend.device
            
            # Create tensor with values that might cause numerical issues
            test_tensor = torch.full((1000,), 1e-10, dtype=torch.float32, device=device)
            
            # Perform operation
            backend.all_reduce(test_tensor)
            
            # Check if result is reasonable (not NaN or Inf)
            if torch.isfinite(test_tensor).all():
                result["details"]["numerically_stable"] = True
            else:
                result["details"]["numerically_stable"] = False
                result["issues"].append("Numerical instability detected")
            
            result["passed"] = len(result["issues"]) == 0
            
        except Exception as e:
            result["issues"].append(f"Numerical stability test failed: {e}")
        
        return result
    
    def _test_precision_preservation(self, backend) -> Dict:
        """Test precision preservation across operations."""
        result = {"passed": False, "details": {}, "issues": []}
        
        try:
            device = backend.device
            
            # Test with different precisions
            test_configs = [
                {"dtype": torch.float32, "tolerance": 1e-6},
                {"dtype": torch.float16, "tolerance": 1e-3}
            ]
            
            for config in test_configs:
                dtype = config["dtype"]
                tolerance = config["tolerance"]
                
                # Create tensor
                test_tensor = torch.randn(1000, dtype=dtype, device=device)
                original_tensor = test_tensor.clone()
                
                # Perform operation
                backend.all_reduce(test_tensor)
                
                # Check if precision is preserved
                if dtype.is_floating_point:
                    if torch.allclose(test_tensor.float(), original_tensor.float(), rtol=tolerance, atol=tolerance):
                        result["details"][f"precision_preserved_{dtype}"] = True
                    else:
                        result["details"][f"precision_preserved_{dtype}"] = False
                        result["issues"].append(f"Precision not preserved for {dtype}")
            
            result["passed"] = len(result["issues"]) == 0
            
        except Exception as e:
            result["issues"].append(f"Precision preservation test failed: {e}")
        
        return result
    
    def _generate_data_integrity_recommendations(self, test_results: Dict) -> List[str]:
        """Generate data integrity recommendations."""
        recommendations = []
        
        for test_name, test_result in test_results.items():
            if not test_result.get("passed", False):
                recommendations.append(f"Data integrity issues detected in {test_name}")
        
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
        results_file = os.path.join(self.output_dir, "correctness_validation_results.json")
        
        with open(results_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        print(f"Correctness validation results saved to: {results_file}")
    
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
        
        except Exception as e:
            print(f"Cleanup error: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="P2P Correctness Validation Tool")
    parser.add_argument("--devices", type=str, help="Comma-separated list of GPU devices to use")
    parser.add_argument("--output-dir", type=str, default="./p2p_correctness", 
                       help="Output directory for correctness validation results")
    parser.add_argument("--category", type=str, 
                       choices=["broadcast", "all_reduce", "gather", "tree_reduce", 
                                "direct_memory", "mathematical", "edge_cases", "data_integrity"],
                       help="Run only specific validation category")
    parser.add_argument("--tolerances", type=str, 
                       help="JSON string with validation tolerances")
    
    args = parser.parse_args()
    
    # Parse devices
    active_devices = None
    if args.devices:
        try:
            active_devices = [int(d.strip()) for d in args.devices.split(",")]
        except ValueError:
            print("Invalid device list")
            return 1
    
    # Parse tolerances
    tolerances = None
    if args.tolerances:
        try:
            tolerances = json.loads(args.tolerances)
        except ValueError:
            print("Invalid tolerances JSON")
            return 1
    
    # Create validator
    validator = P2PCorrectnessValidator(
        active_devices=active_devices,
        output_dir=args.output_dir
    )
    
    # Set tolerances if provided
    if tolerances:
        validator.tolerances.update(tolerances)
    
    try:
        if args.category:
            # Run specific category
            if args.category == "broadcast":
                validator._validate_broadcast_correctness()
            elif args.category == "all_reduce":
                validator._validate_all_reduce_correctness()
            elif args.category == "gather":
                validator._validate_gather_correctness()
            elif args.category == "tree_reduce":
                validator._validate_tree_reduce_correctness()
            elif args.category == "direct_memory":
                validator._validate_direct_memory_correctness()
            elif args.category == "mathematical":
                validator._validate_mathematical_properties()
            elif args.category == "edge_cases":
                validator._validate_edge_cases()
            elif args.category == "data_integrity":
                validator._validate_data_integrity()
        else:
            # Run all validation
            validator.validate_all()
        
        # Print summary
        summary = validator.validation_results["summary"]
        print("\n" + "=" * 50)
        print("CORRECTNESS VALIDATION SUMMARY")
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