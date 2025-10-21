import torch
import numpy as np
import ctypes
from typing import List, Dict, Tuple, Optional
from ..util import log_tp
from .model_tp_cuda import _cudart

# CUDA P2P API constants
CUDA_SUCCESS = 0
CUDA_ERROR_PEER_ACCESS_NOT_SUPPORTED = 724
CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 725

class P2PTopology:
    """
    P2P topology detection and analysis for ExLlamaV3 tensor parallelism.
    
    This class detects P2P capabilities between GPU pairs and builds optimal
    communication topologies for tensor parallel operations.
    """
    
    def __init__(self, active_devices: List[int]):
        """
        Initialize P2P topology detection.
        
        Args:
            active_devices: List of active GPU device IDs
        """
        self.active_devices = active_devices
        self.num_devices = len(active_devices)
        self.device_to_index = {device: idx for idx, device in enumerate(active_devices)}
        self.index_to_device = {idx: device for idx, device in enumerate(active_devices)}
        
        # P2P capability matrix: p2p_matrix[i][j] = True if device i can access device j
        self.p2p_matrix = None
        self.topology_type = None
        self.communication_tree = None
        
        # Detect P2P capabilities on initialization
        self._detect_p2p_capabilities()
        
    def _detect_p2p_capabilities(self) -> None:
        """
        Detect P2P capabilities between all GPU pairs using CUDA APIs.
        """
        log_tp(None, "Detecting P2P capabilities between GPUs")
        
        # Initialize P2P matrix
        self.p2p_matrix = np.zeros((self.num_devices, self.num_devices), dtype=bool)
        
        # Set diagonal to True (devices can always access themselves)
        for i in range(self.num_devices):
            self.p2p_matrix[i][i] = True
        
        # Use PyTorch's built-in P2P detection which is more reliable
        try:
            # Check P2P capabilities using PyTorch
            for i, device_i in enumerate(self.active_devices):
                for j, device_j in enumerate(self.active_devices):
                    if i == j:
                        continue
                        
                    try:
                        # Use PyTorch to check P2P capability
                        with torch.cuda.device(device_i):
                            can_access = torch.cuda.can_device_access_peer(device_j)
                            if can_access:
                                self.p2p_matrix[i][j] = True
                                log_tp(device_i, f"P2P: Can access device {device_j}")
                            else:
                                log_tp(device_i, f"P2P: Cannot access device {device_j}")
                    except Exception as e:
                        log_tp(device_i, f"P2P detection error for device {device_j}: {e}")
                        
        except Exception as e:
            log_tp(None, f"PyTorch P2P detection failed: {e}")
            
            # Fallback to CUDA runtime API
            try:
                cudart = _cudart()
                
                # Get CUDA API functions
                cuda_device_can_access_peer = cudart.cudaDeviceCanAccessPeer
                cuda_device_can_access_peer.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
                cuda_device_can_access_peer.restype = ctypes.c_int
                
                # Test P2P capabilities between all pairs
                for i, device_i in enumerate(self.active_devices):
                    for j, device_j in enumerate(self.active_devices):
                        if i == j:
                            continue
                            
                        try:
                            # Check if device_i can access device_j
                            can_access = ctypes.c_int()
                            result = cuda_device_can_access_peer(
                                ctypes.c_int(device_i),
                                ctypes.c_int(device_j),
                                ctypes.byref(can_access)
                            )
                            
                            if result == CUDA_SUCCESS and can_access.value == 1:
                                self.p2p_matrix[i][j] = True
                                log_tp(device_i, f"P2P: Can access device {device_j}")
                            else:
                                log_tp(device_i, f"P2P: Cannot access device {device_j}")
                                
                        except Exception as e:
                            log_tp(device_i, f"P2P detection error for device {device_j}: {e}")
                            
            except Exception as e:
                log_tp(None, f"CUDA runtime P2P detection failed: {e}")
        
        # Analyze and build optimal topology
        self._analyze_topology()
        
    def _analyze_topology(self) -> None:
        """
        Analyze P2P matrix and determine optimal communication topology.
        """
        if self.is_fully_connected():
            self.topology_type = "fully_connected"
            log_tp(None, "P2P: All GPUs fully connected - using mesh topology")
        else:
            # Build tree topology for reduction operations
            self.communication_tree = self.find_optimal_tree()
            self.topology_type = "tree"
            log_tp(None, f"P2P: Using tree topology for communication")
            
    def is_fully_connected(self) -> bool:
        """
        Check if all GPUs can communicate directly with each other.
        
        Returns:
            True if fully connected, False otherwise
        """
        if self.p2p_matrix is None:
            return False
            
        # Check if all off-diagonal elements are True
        for i in range(self.num_devices):
            for j in range(self.num_devices):
                if i != j and not self.p2p_matrix[i][j]:
                    return False
        return True
        
    def find_optimal_tree(self) -> Dict:
        """
        Build optimal tree structure for reduction operations.
        
        Returns:
            Dictionary representing the tree topology
        """
        if self.p2p_matrix is None:
            return {}
            
        # Use binary tree as default for better scalability
        return self.build_binary_tree()
    
    def build_binary_tree(self) -> Dict:
        """
        Build binary tree structure for reduction operations.
        
        Returns:
            Dictionary representing the binary tree topology
        """
        if self.p2p_matrix is None:
            return {}
            
        tree = {
            "root": self.active_devices[0],
            "children": {},
            "parent": {},
            "depth": {},
            "tree_type": "binary"
        }
        
        # Build binary tree: parent = floor((i-1)/2)
        for i, device in enumerate(self.active_devices):
            if i > 0:
                parent_idx = (i - 1) // 2
                parent_device = self.active_devices[parent_idx]
                
                # Check if P2P is available between parent and child
                if self.can_access_peer(parent_device, device):
                    tree["parent"][device] = parent_device
                    tree["children"][parent_device] = tree["children"].get(parent_device, [])
                    tree["children"][parent_device].append(device)
                    tree["depth"][device] = tree["depth"].get(parent_device, 0) + 1
                else:
                    # Fallback: find alternative parent
                    alt_parent = self.find_alternative_parent(device, tree)
                    if alt_parent:
                        tree["parent"][device] = alt_parent
                        tree["children"][alt_parent] = tree["children"].get(alt_parent, [])
                        tree["children"][alt_parent].append(device)
                        tree["depth"][device] = tree["depth"].get(alt_parent, 0) + 1
        
        return tree
    
    def build_kary_tree(self, k: int = 4) -> Dict:
        """
        Build k-ary tree structure for reduction operations.
        
        Args:
            k: Number of children per node (default: 4)
            
        Returns:
            Dictionary representing the k-ary tree topology
        """
        if self.p2p_matrix is None:
            return {}
            
        tree = {
            "root": self.active_devices[0],
            "children": {},
            "parent": {},
            "depth": {},
            "tree_type": f"{k}-ary"
        }
        
        # Build k-ary tree: parent = floor((i-1)/k)
        for i, device in enumerate(self.active_devices):
            if i > 0:
                parent_idx = (i - 1) // k
                parent_device = self.active_devices[parent_idx]
                
                # Check if P2P is available between parent and child
                if self.can_access_peer(parent_device, device):
                    tree["parent"][device] = parent_device
                    tree["children"][parent_device] = tree["children"].get(parent_device, [])
                    tree["children"][parent_device].append(device)
                    tree["depth"][device] = tree["depth"].get(parent_device, 0) + 1
                else:
                    # Fallback: find alternative parent
                    alt_parent = self.find_alternative_parent(device, tree)
                    if alt_parent:
                        tree["parent"][device] = alt_parent
                        tree["children"][alt_parent] = tree["children"].get(alt_parent, [])
                        tree["children"][alt_parent].append(device)
                        tree["depth"][device] = tree["depth"].get(alt_parent, 0) + 1
        
        return tree
    
    def build_balanced_tree(self) -> Dict:
        """
        Build balanced tree structure for reduction operations.
        
        Returns:
            Dictionary representing the balanced tree topology
        """
        if self.p2p_matrix is None:
            return {}
            
        # For balanced tree, we want to minimize the maximum depth
        # This is more complex and requires finding optimal branching factor
        num_devices = self.num_devices
        
        # Calculate optimal branching factor
        if num_devices <= 4:
            k = 2  # Binary tree for small number of devices
        elif num_devices <= 16:
            k = 4  # 4-ary tree for medium number of devices
        else:
            k = 8  # 8-ary tree for large number of devices
        
        return self.build_kary_tree(k)
    
    def find_alternative_parent(self, device: int, tree: Dict) -> Optional[int]:
        """
        Find alternative parent for a device when direct P2P is not available.
        
        Args:
            device: Device ID that needs a parent
            tree: Current tree structure
            
        Returns:
            Alternative parent device ID or None if not found
        """
        device_idx = self.device_to_index[device]
        
        # Find devices that can access this device
        potential_parents = []
        for i, can_access in enumerate(self.p2p_matrix[:, device_idx]):
            if can_access and i != device_idx:
                potential_parents.append(self.index_to_device[i])
        
        # Choose the parent with minimum depth in current tree
        best_parent = None
        min_depth = float('inf')
        
        for parent in potential_parents:
            depth = tree["depth"].get(parent, 0)
            if depth < min_depth:
                min_depth = depth
                best_parent = parent
        
        return best_parent
    
    def get_tree_depth(self, tree: Dict) -> int:
        """
        Get the depth of the tree.
        
        Args:
            tree: Tree structure
            
        Returns:
            Maximum depth of the tree
        """
        if "depth" not in tree:
            return 0
        return max(tree["depth"].values()) if tree["depth"] else 0
    
    def get_tree_stats(self, tree: Dict) -> Dict:
        """
        Get statistics about the tree structure.
        
        Args:
            tree: Tree structure
            
        Returns:
            Dictionary with tree statistics
        """
        stats = {
            "tree_type": tree.get("tree_type", "unknown"),
            "num_devices": self.num_devices,
            "tree_depth": self.get_tree_depth(tree),
            "root": tree.get("root"),
            "num_leaves": 0,
            "branching_factors": []
        }
        
        # Count leaves and calculate branching factors
        for device in self.active_devices:
            children = tree["children"].get(device, [])
            if len(children) == 0:
                stats["num_leaves"] += 1
            else:
                stats["branching_factors"].append(len(children))
        
        if stats["branching_factors"]:
            stats["avg_branching_factor"] = sum(stats["branching_factors"]) / len(stats["branching_factors"])
            stats["max_branching_factor"] = max(stats["branching_factors"])
        else:
            stats["avg_branching_factor"] = 0
            stats["max_branching_factor"] = 0
        
        return stats
        
    def build_optimal_topology(self, operation_type: str, tensor_size: int = 0) -> Dict:
        """
        Determine best communication pattern for a given operation.
        
        Args:
            operation_type: Type of operation ("reduce", "broadcast", "gather")
            tensor_size: Size of tensor in bytes (for adaptive selection)
            
        Returns:
            Dictionary describing the optimal communication pattern
        """
        if self.p2p_matrix is None:
            return {"type": "fallback", "reason": "P2P not available"}
            
        if self.is_fully_connected():
            # For fully connected topologies, choose based on tensor size and device count
            if self.num_devices <= 4 or tensor_size < 1024 * 1024:  # 1MB
                return {
                    "type": "mesh",
                    "pattern": "all_to_all",
                    "reason": "Fully connected P2P available, small tensor or few devices"
                }
            else:
                # Use tree for larger tensors and more devices
                tree = self.build_balanced_tree()
                return {
                    "type": "tree",
                    "pattern": "balanced_tree",
                    "tree": tree,
                    "reason": "Tree reduction optimal for large tensors with many devices"
                }
        else:
            # For partial connectivity, always use tree
            if operation_type == "reduce":
                # Choose tree type based on connectivity ratio
                connectivity_ratio = self.get_connectivity_ratio()
                if connectivity_ratio > 0.7:
                    tree = self.build_binary_tree()
                elif connectivity_ratio > 0.4:
                    tree = self.build_kary_tree(4)
                else:
                    tree = self.build_balanced_tree()
                    
                return {
                    "type": "tree",
                    "pattern": "reduce_tree",
                    "tree": tree,
                    "reason": f"Tree reduction optimal for partial P2P (connectivity: {connectivity_ratio:.2f})"
                }
            elif operation_type == "broadcast":
                tree = self.build_binary_tree()
                return {
                    "type": "tree",
                    "pattern": "broadcast_tree",
                    "tree": tree,
                    "reason": "Tree broadcast optimal for partial P2P"
                }
            else:  # gather
                return {
                    "type": "hybrid",
                    "pattern": "tree_gather",
                    "tree": self.communication_tree,
                    "reason": "Hybrid gather for partial P2P"
                }
    
    def get_connectivity_ratio(self) -> float:
        """
        Calculate the connectivity ratio of the P2P topology.
        
        Returns:
            Ratio of connected pairs to total possible pairs
        """
        if self.p2p_matrix is None:
            return 0.0
            
        total_pairs = self.num_devices * (self.num_devices - 1)
        connected_pairs = np.sum(self.p2p_matrix) - self.num_devices  # Subtract diagonal
        
        return connected_pairs / total_pairs if total_pairs > 0 else 0.0
    
    def select_reduce_algorithm(self, tensor_size: int) -> str:
        """
        Select the best reduction algorithm based on topology and tensor size.
        
        Args:
            tensor_size: Size of tensor in bytes
            
        Returns:
            Algorithm name ("binary_tree", "kary_tree", "balanced_tree", "ring")
        """
        connectivity_ratio = self.get_connectivity_ratio()
        
        # For small number of devices, ring might be better
        if self.num_devices <= 4:
            return "ring"
        
        # For highly connected topologies, tree is better
        if connectivity_ratio > 0.7:
            if tensor_size > 10 * 1024 * 1024:  # 10MB
                return "balanced_tree"
            else:
                return "binary_tree"
        
        # For medium connectivity, use k-ary tree
        if connectivity_ratio > 0.4:
            return "kary_tree"
        
        # For low connectivity, fallback to ring
        return "ring"
    
    def get_tree_for_algorithm(self, algorithm: str) -> Dict:
        """
        Get the appropriate tree structure for a given algorithm.
        
        Args:
            algorithm: Algorithm name
            
        Returns:
            Tree structure dictionary
        """
        if algorithm == "binary_tree":
            return self.build_binary_tree()
        elif algorithm == "kary_tree":
            return self.build_kary_tree(4)
        elif algorithm == "balanced_tree":
            return self.build_balanced_tree()
        else:
            return self.communication_tree or {}
                
    def get_p2p_matrix(self) -> np.ndarray:
        """
        Get the P2P capability matrix.
        
        Returns:
            Boolean matrix where True indicates P2P capability
        """
        return self.p2p_matrix.copy() if self.p2p_matrix is not None else np.array([])
        
    def can_access_peer(self, device_a: int, device_b: int) -> bool:
        """
        Check if device_a can access device_b via P2P.
        
        Args:
            device_a: Source device ID
            device_b: Target device ID
            
        Returns:
            True if P2P access is possible
        """
        if self.p2p_matrix is None:
            return False
            
        if device_a not in self.device_to_index or device_b not in self.device_to_index:
            return False
            
        idx_a = self.device_to_index[device_a]
        idx_b = self.device_to_index[device_b]
        
        return self.p2p_matrix[idx_a][idx_b]
        
    def get_topology_summary(self) -> Dict:
        """
        Get summary of detected topology.
        
        Returns:
            Dictionary with topology information
        """
        if self.p2p_matrix is None:
            return {"status": "failed", "reason": "P2P detection failed"}
            
        total_pairs = self.num_devices * (self.num_devices - 1)
        connected_pairs = np.sum(self.p2p_matrix) - self.num_devices  # Subtract diagonal
        
        return {
            "status": "success",
            "num_devices": self.num_devices,
            "topology_type": self.topology_type,
            "connected_pairs": int(connected_pairs),
            "total_pairs": total_pairs,
            "connectivity_ratio": connected_pairs / total_pairs if total_pairs > 0 else 0,
            "is_fully_connected": self.is_fully_connected()
        }


def detect_p2p_capabilities(active_devices: List[int]) -> np.ndarray:
    """
    Detect P2P capabilities between active GPU devices.
    
    Args:
        active_devices: List of active GPU device IDs
        
    Returns:
        Boolean matrix of P2P capabilities
    """
    topology = P2PTopology(active_devices)
    return topology.get_p2p_matrix()


def build_optimal_topology(p2p_matrix: np.ndarray, operation_type: str) -> Dict:
    """
    Build optimal communication topology based on P2P capabilities.
    
    Args:
        p2p_matrix: Boolean matrix of P2P capabilities
        operation_type: Type of operation ("reduce", "broadcast", "gather")
        
    Returns:
        Dictionary describing the optimal communication pattern
    """
    # This is a simplified version - in practice you'd pass the full topology object
    if np.all(p2p_matrix):
        return {
            "type": "mesh",
            "pattern": "all_to_all",
            "reason": "Fully connected P2P available"
        }
    else:
        return {
            "type": "tree",
            "pattern": f"{operation_type}_tree",
            "reason": "Tree topology for partial P2P"
        }


def is_fully_connected(p2p_matrix: np.ndarray) -> bool:
    """
    Check if P2P matrix represents a fully connected topology.
    
    Args:
        p2p_matrix: Boolean matrix of P2P capabilities
        
    Returns:
        True if fully connected, False otherwise
    """
    if p2p_matrix.size == 0:
        return False
        
    n = p2p_matrix.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j and not p2p_matrix[i][j]:
                return False
    return True


def find_optimal_tree(p2p_matrix: np.ndarray) -> Dict:
    """
    Find optimal tree structure for reduction operations.
    
    Args:
        p2p_matrix: Boolean matrix of P2P capabilities
        
    Returns:
        Dictionary representing the tree topology
    """
    if p2p_matrix.size == 0:
        return {}
        
    n = p2p_matrix.shape[0]
    connection_counts = np.sum(p2p_matrix, axis=1)
    root_idx = np.argmax(connection_counts)
    
    tree = {
        "root": root_idx,
        "children": {},
        "parent": {}
    }
    
    visited = set([root_idx])
    queue = [root_idx]
    
    while queue:
        current_idx = queue.pop(0)
        
        for neighbor_idx in range(n):
            if neighbor_idx not in visited and p2p_matrix[current_idx][neighbor_idx]:
                tree["children"][current_idx] = tree["children"].get(current_idx, [])
                tree["children"][current_idx].append(neighbor_idx)
                tree["parent"][neighbor_idx] = current_idx
                visited.add(neighbor_idx)
                queue.append(neighbor_idx)
    
    return tree