import torch
import torch.distributed as dist
import time
import numpy as np
from .model_tp_cuda import cuda_host_register, cuda_host_unregister, CUDA_HOST_REGISTER_PORTABLE
import exllamav3_ext as ext
from multiprocessing import shared_memory, Barrier
from ..util import log_tp
from .model_tp_p2p import P2PTopology

# Import monitoring tools
try:
    from ..util.p2p_monitor import get_global_monitor
    from ..util.p2p_profiler import get_global_profiler
    from ..util.p2p_debug import get_global_debugger
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

GLOBALS_SIZE = 128*1024
SHBUF_SIZE = 16 * 1024 ** 2
SHBUF_SIZE_R = 17 * 128 * 1024
SHBUF_SIZE_S = 16 * 1024
MAX_CPU_REDUCE = SHBUF_SIZE_R // 17 // 256 * 256

class TPBackend:

    def __init__(self):
        pass

    def close(self):
        pass

    def fwd_barrier(self):
        raise NotImplementedError()


class TPBackendP2P:

    def __init__(
        self,
        device: int,
        active_devices: list[int],
        output_device: int,
        init_method: str,
        master: bool,
        uuid: str,
        shbuf_size: int = SHBUF_SIZE,
        auto_fallback: bool = True,
    ):
        self.device = device
        if device < 0:
            log_tp(device, f"P2P init: skip CPU process")
            return

        self.active_devices = active_devices
        self.world_size = len(active_devices)
        self.rank = active_devices.index(device)
        self.output_device = output_device
        self.uuid = uuid
        self.shbuf_size = shbuf_size

        log_tp(device, f"P2P init: world_size {self.world_size}, rank {self.rank}, device {device}")
        print(f" -- P2P init: world_size {self.world_size}, rank {self.rank}, device {device}")
        
        self.auto_fallback = auto_fallback
        
        # Skip P2P topology detection to avoid peer access conflicts
        # We'll assume P2P is available and let the memory pool initialization handle it
        if master:
            log_tp(device, "Skipping P2P topology detection to avoid conflicts")
            print(f" -- Skipping P2P topology detection to avoid conflicts")
            self.p2p_topology = None
            self.use_p2p = True
            # Initialize P2P memory pool
            self._init_p2p_memory_pool()
        else:
            # Non-master processes will get topology from shared memory
            self.p2p_topology = None
            self.use_p2p = True  # Assume P2P is available for non-master processes
            # Initialize P2P memory pool
            self._init_p2p_memory_pool()
        
        # Create fallback to native backend for operations not supported by P2P
        self.fallback = TPBackendNative(
            device,
            active_devices,
            output_device,
            init_method,
            master,
            uuid,
            shbuf_size
        )
        
        # Create abort flag for P2P operations
        if self.device >= 0:
            self.abort_flag = torch.zeros((1,), device=self.device, dtype=torch.int)
        else:
            self.abort_flag = None
        
        # Initialize monitoring tools if available
        self.monitor = None
        self.profiler = None
        self.debugger = None
        
        if MONITORING_AVAILABLE:
            self.monitor = get_global_monitor()
            self.profiler = get_global_profiler()
            self.debugger = get_global_debugger()
            
            # Set topology for monitoring
            if self.monitor and self.p2p_topology:
                self.monitor.set_topology(self.p2p_topology)

    def _init_p2p_memory_pool(self):
        """Initialize P2P memory pool for this device with adaptive sizing."""
        if self.device < 0 or not self.use_p2p:
            return
            
        try:
            # Adaptive pool sizing based on available memory and device count
            import torch
            total_memory = torch.cuda.get_device_properties(self.device).total_memory
            available_memory = total_memory - torch.cuda.memory_allocated(self.device)
            
            # Calculate pool size as 10% of available memory, with min/max limits
            min_pool_size = 32 * 1024 * 1024  # 32MB minimum
            max_pool_size = 512 * 1024 * 1024  # 512MB maximum
            pool_size = int(min(max(available_memory * 0.1, min_pool_size), max_pool_size))
            
            # Initialize memory pool
            ext.p2p_init_memory_pool(self.device, pool_size, self.abort_flag)
            log_tp(self.device, f"P2P memory pool initialized: {pool_size // 1024**2}MB ({pool_size} bytes)")
            
            # Initialize direct memory pool for P2P access with peer-specific sizing
            peer_devices = [d for d in self.active_devices if d != self.device]
            # Adjust pool size based on number of peers
            peer_pool_size = max(pool_size // len(peer_devices), min_pool_size // 2) if peer_devices else pool_size
            ext.p2p_init_direct_memory_pool(self.device, peer_pool_size, peer_devices, self.abort_flag)
            log_tp(self.device, f"P2P direct memory pool initialized: {peer_pool_size // 1024**2}MB with {len(peer_devices)} peers")
            
            # Enable P2P access for all peer devices (centralized management)
            try:
                ext.p2p_enable_all_peer_access(self.device, peer_devices, self.abort_flag)
                log_tp(self.device, f"P2P access enabled for {len(peer_devices)} peer devices")
            except Exception as e:
                log_tp(self.device, f"P2P access enabling failed: {e}")
                self.use_p2p = False
        except Exception as e:
            log_tp(self.device, f"P2P memory pool initialization failed: {e}")
            self.use_p2p = False

    def close(self):
        if self.device < 0:
            log_tp(self.device, f"P2P close: skip CPU process")
            return

        # Cleanup P2P memory pool
        if self.use_p2p:
            try:
                ext.p2p_cleanup_memory_pool(self.device, self.abort_flag)
                ext.p2p_cleanup_direct_memory_pool(self.device, self.abort_flag)
                log_tp(self.device, "P2P memory pool cleaned up")
            except Exception as e:
                log_tp(self.device, f"P2P memory pool cleanup failed: {e}")

        self.fallback.close()

    def fwd_barrier(self):
        if self.use_p2p:
            try:
                # Use P2P-aware barrier
                ext.p2p_device_barrier(self.active_devices, self.device, self.abort_flag)
            except Exception as e:
                log_tp(self.device, f"P2P barrier failed: {e}, using fallback")
                self.fallback.fwd_barrier()
        else:
            self.fallback.fwd_barrier()

    def broadcast(self, tensor: torch.Tensor, src_device: int):
        # Record operation start time for monitoring
        start_time = time.time()
        success = True
        
        if not self.use_p2p:
            self.fallback.broadcast(tensor, src_device)
            # Record fallback operation
            if self.monitor:
                end_time = time.time()
                self.monitor.record_operation(
                    operation_type="broadcast",
                    device_id=self.device,
                    peer_device=src_device,
                    tensor=tensor,
                    start_time=start_time,
                    end_time=end_time,
                    algorithm="fallback",
                    success=True
                )
            return
            
        try:
            # Log debug event
            if self.debugger:
                self.debugger.log_event(
                    event_type="operation_start",
                    operation_type="broadcast",
                    device_id=self.device,
                    peer_device=src_device,
                    tensor=tensor,
                    message=f"Starting P2P broadcast from device {src_device}"
                )
            
            # Check if P2P is available between source and this device
            # Since we skipped topology detection, we'll assume P2P is available
            if src_device != self.device:
                # Use direct memory copy for broadcast when possible
                if self.device != src_device:
                    # Create a temporary tensor on this device for direct copy
                    temp_tensor = torch.empty_like(tensor, device=self.device)
                    ext.p2p_copy_tensor_async(src_device, self.device, tensor, temp_tensor, self.abort_flag)
                    tensor.copy_(temp_tensor)
                    log_tp(self.device, f"P2P direct broadcast from device {src_device}")
                else:
                    log_tp(self.device, f"P2P broadcast: already on device {src_device}")
                
                # Fallback to traditional P2P broadcast for synchronization
                if tensor.numel() * tensor.element_size() <= 2048:
                    ext.p2p_broadcast_ll(self.fallback.ptr_g, self.active_devices, self.device,
                                       src_device, tensor, self.abort_flag)
                else:
                    ext.p2p_broadcast(self.fallback.ptr_g, self.active_devices, self.device,
                                    src_device, tensor, self.abort_flag)
            else:
                # Fallback to native broadcast
                log_tp(self.device, f"P2P not available from {src_device}, using fallback")
                self.fallback.broadcast(tensor, src_device)
                
        except Exception as e:
            success = False
            log_tp(self.device, f"P2P broadcast failed: {e}, using fallback")
            
            # Log debug error
            if self.debugger:
                self.debugger.log_error(
                    error_type="broadcast_error",
                    operation_type="broadcast",
                    device_id=self.device,
                    peer_device=src_device,
                    error_message=str(e),
                    context={"tensor_shape": tensor.shape, "tensor_dtype": str(tensor.dtype)}
                )
            
            self.fallback.broadcast(tensor, src_device)
        
        finally:
            # Record operation for monitoring
            if self.monitor:
                end_time = time.time()
                self.monitor.record_operation(
                    operation_type="broadcast",
                    device_id=self.device,
                    peer_device=src_device,
                    tensor=tensor,
                    start_time=start_time,
                    end_time=end_time,
                    algorithm="p2p_direct" if self.use_p2p else "fallback",
                    success=success
                )
            
            # Log debug event
            if self.debugger:
                self.debugger.log_event(
                    event_type="operation_end",
                    operation_type="broadcast",
                    device_id=self.device,
                    peer_device=src_device,
                    tensor=tensor,
                    message=f"Completed P2P broadcast: {'success' if success else 'failed'}"
                )

    def all_reduce(self, tensor: torch.Tensor, contribution: bool = True):
        # Record operation start time for monitoring
        start_time = time.time()
        success = True
        algorithm = None
        
        if not self.use_p2p:
            self.fallback.all_reduce(tensor, contribution)
            # Record fallback operation
            if self.monitor:
                end_time = time.time()
                self.monitor.record_operation(
                    operation_type="all_reduce",
                    device_id=self.device,
                    peer_device=None,
                    tensor=tensor,
                    start_time=start_time,
                    end_time=end_time,
                    algorithm="fallback",
                    success=True
                )
            return
            
        try:
            # Log debug event
            if self.debugger:
                self.debugger.log_event(
                    event_type="operation_start",
                    operation_type="all_reduce",
                    device_id=self.device,
                    tensor=tensor,
                    message="Starting P2P all_reduce"
                )
            
            # Get tensor size for adaptive algorithm selection
            tensor_size = tensor.numel() * tensor.element_size()
            
            # Since we skipped topology detection, use default algorithm
            algorithm = "ring"  # Default to ring algorithm
            connectivity_ratio = 1.0  # Assume full connectivity
            
            # Optimize tensor layout for better performance
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            
            # Use dtype optimization for better bandwidth
            original_dtype = tensor.dtype
            if tensor.dtype == torch.float32 and tensor_size > 1024 * 1024:  # > 1MB
                # Use half precision for large tensors to reduce bandwidth
                tensor = tensor.half()
                contribution = False  # Don't contribute twice
            
            log_tp(self.device, f"P2P all_reduce: algorithm={algorithm}, connectivity={connectivity_ratio:.2f}, size={tensor_size//1024}KB")
            
            # Use the selected algorithm
            if algorithm == "ring":
                # Ring-based all_reduce for partial connectivity or small device count
                ext.p2p_all_reduce_ring(self.fallback.ptr_g, self.active_devices, self.device,
                                      self.active_devices[0], tensor, self.abort_flag)
            elif algorithm in ["binary_tree", "kary_tree", "balanced_tree"]:
                # Tree-based all_reduce with adaptive selection
                tree_type = 0 if algorithm == "binary_tree" else (1 if algorithm == "kary_tree" else 2)
                ext.p2p_all_reduce_tree_adaptive(self.fallback.ptr_g, self.active_devices, self.device,
                                               self.active_devices[0], tensor, self.abort_flag,
                                               connectivity_ratio)
            else:
                # Fallback to original P2P all_reduce
                ext.p2p_all_reduce(self.fallback.ptr_g, self.active_devices, self.device,
                                 self.active_devices[0], tensor, self.abort_flag)
            
            # Convert back to original dtype if needed
            if original_dtype == torch.float32 and tensor.dtype != original_dtype:
                tensor = tensor.float()
            
            log_tp(self.device, f"P2P all_reduce completed using {algorithm}")
            
        except Exception as e:
            success = False
            log_tp(self.device, f"P2P all_reduce failed: {e}, using fallback")
            
            # Log debug error
            if self.debugger:
                self.debugger.log_error(
                    error_type="all_reduce_error",
                    operation_type="all_reduce",
                    device_id=self.device,
                    error_message=str(e),
                    context={
                        "tensor_shape": tensor.shape,
                        "tensor_dtype": str(tensor.dtype),
                        "algorithm": algorithm,
                        "connectivity_ratio": connectivity_ratio if 'connectivity_ratio' in locals() else 0.0
                    }
                )
            
            self.fallback.all_reduce(tensor, contribution)
        
        finally:
            # Record operation for monitoring
            if self.monitor:
                end_time = time.time()
                self.monitor.record_operation(
                    operation_type="all_reduce",
                    device_id=self.device,
                    peer_device=None,
                    tensor=tensor,
                    start_time=start_time,
                    end_time=end_time,
                    algorithm=algorithm or "fallback",
                    success=success
                )
            
            # Log debug event
            if self.debugger:
                self.debugger.log_event(
                    event_type="operation_end",
                    operation_type="all_reduce",
                    device_id=self.device,
                    tensor=tensor,
                    message=f"Completed P2P all_reduce: {'success' if success else 'failed'}",
                    details={"algorithm": algorithm}
                )

    def gather(
        self,
        tensor: torch.Tensor,
        out_tensor: torch.Tensor | None,
        gather_devices: torch.Tensor | None,
        out_device: int,
        ldims: list[int]
    ):
        # Record operation start time for monitoring
        start_time = time.time()
        success = True
        
        if not self.use_p2p:
            self.fallback.gather(tensor, out_tensor, gather_devices, out_device, ldims)
            # Record fallback operation
            if self.monitor:
                end_time = time.time()
                self.monitor.record_operation(
                    operation_type="gather",
                    device_id=self.device,
                    peer_device=out_device,
                    tensor=tensor,
                    start_time=start_time,
                    end_time=end_time,
                    algorithm="fallback",
                    success=True
                )
            return
            
        try:
            # Log debug event
            if self.debugger:
                self.debugger.log_event(
                    event_type="operation_start",
                    operation_type="gather",
                    device_id=self.device,
                    peer_device=out_device,
                    tensor=tensor,
                    message=f"Starting P2P gather to device {out_device}"
                )
            
            # Use P2P-optimized gather
            if gather_devices is None:
                gather_devices_list = self.active_devices
            else:
                gather_devices_list = gather_devices.tolist()
                
            # Since we skipped topology detection, assume we can use direct P2P gather
            can_use_direct = True
                    
            if can_use_direct and self.device != out_device:
                # Use direct memory copy for gather when possible
                if out_tensor is not None:
                    # Calculate offset for this device's contribution
                    offset = 0
                    for i, device in enumerate(gather_devices_list):
                        if device == self.device:
                            break
                        offset += ldims[i]
                    
                    # Copy directly to output tensor
                    slice_size = tensor.numel() * tensor.element_size()
                    ext.p2p_copy_tensor_with_offset(
                        self.device, out_device, tensor, out_tensor,
                        0, offset * tensor.element_size(), slice_size, self.abort_flag
                    )
                    log_tp(self.device, f"P2P direct gather copy to device {out_device}")
            
            # Use traditional P2P gather for synchronization
            ext.p2p_gather(self.fallback.ptr_g, self.active_devices, self.device,
                         out_device, tensor, out_tensor, ldims, self.abort_flag)
            log_tp(self.device, f"P2P gather to device {out_device}")
            
        except Exception as e:
            success = False
            log_tp(self.device, f"P2P gather failed: {e}, using fallback")
            
            # Log debug error
            if self.debugger:
                self.debugger.log_error(
                    error_type="gather_error",
                    operation_type="gather",
                    device_id=self.device,
                    peer_device=out_device,
                    error_message=str(e),
                    context={
                        "tensor_shape": tensor.shape,
                        "tensor_dtype": str(tensor.dtype),
                        "out_device": out_device,
                        "gather_devices": gather_devices_list if 'gather_devices_list' in locals() else []
                    }
                )
            
            self.fallback.gather(tensor, out_tensor, gather_devices, out_device, ldims)
        
        finally:
            # Record operation for monitoring
            if self.monitor:
                end_time = time.time()
                self.monitor.record_operation(
                    operation_type="gather",
                    device_id=self.device,
                    peer_device=out_device,
                    tensor=tensor,
                    start_time=start_time,
                    end_time=end_time,
                    algorithm="p2p_direct" if can_use_direct else "p2p_traditional",
                    success=success
                )
            
            # Log debug event
            if self.debugger:
                self.debugger.log_event(
                    event_type="operation_end",
                    operation_type="gather",
                    device_id=self.device,
                    peer_device=out_device,
                    tensor=tensor,
                    message=f"Completed P2P gather: {'success' if success else 'failed'}",
                    details={"can_use_direct": can_use_direct if 'can_use_direct' in locals() else False}
                )

    def run_cpu_reduce_jobs(self):
        pass

    def end_cpu_reduce_jobs(self):
        pass

    def copy_tensor_direct(self, src_device: int, dst_device: int, tensor: torch.Tensor):
        """Direct GPU-to-GPU tensor copy using P2P memory access."""
        # Record operation start time for monitoring
        start_time = time.time()
        success = True
        result_tensor = None
        
        if not self.use_p2p:
            # Fallback to traditional copy
            if self.device == dst_device:
                tensor_copy = torch.empty_like(tensor, device=self.device)
                tensor_copy.copy_(tensor)
                result_tensor = tensor_copy
            else:
                raise RuntimeError(f"Cannot copy from device {src_device} to {dst_device} without P2P")
            
            # Record fallback operation
            if self.monitor:
                end_time = time.time()
                self.monitor.record_operation(
                    operation_type="direct_copy",
                    device_id=self.device,
                    peer_device=dst_device if self.device == src_device else src_device,
                    tensor=tensor,
                    start_time=start_time,
                    end_time=end_time,
                    algorithm="fallback",
                    success=True
                )
            return result_tensor
        
        try:
            # Log debug event
            if self.debugger:
                self.debugger.log_event(
                    event_type="operation_start",
                    operation_type="direct_copy",
                    device_id=self.device,
                    peer_device=dst_device if self.device == src_device else src_device,
                    tensor=tensor,
                    message=f"Starting P2P direct copy: {src_device} -> {dst_device}"
                )
            
            if self.device == src_device:
                # We are the source device
                dst_tensor = torch.empty_like(tensor, device=dst_device)
                ext.p2p_copy_tensor_async(src_device, dst_device, tensor, dst_tensor, self.abort_flag)
                log_tp(self.device, f"P2P direct copy: {src_device} -> {dst_device}")
                result_tensor = dst_tensor
            elif self.device == dst_device:
                # We are the destination device
                src_tensor = torch.empty_like(tensor, device=src_device)
                ext.p2p_copy_tensor_async(src_device, dst_device, src_tensor, tensor, self.abort_flag)
                log_tp(self.device, f"P2P direct copy: {src_device} -> {dst_device}")
                result_tensor = tensor
            else:
                # We are neither source nor destination, use fallback
                result_tensor = self.fallback.copy_tensor_direct(src_device, dst_device, tensor)
                
        except Exception as e:
            success = False
            log_tp(self.device, f"P2P direct copy failed: {e}, using fallback")
            
            # Log debug error
            if self.debugger:
                self.debugger.log_error(
                    error_type="direct_copy_error",
                    operation_type="direct_copy",
                    device_id=self.device,
                    peer_device=dst_device if self.device == src_device else src_device,
                    error_message=str(e),
                    context={
                        "tensor_shape": tensor.shape,
                        "tensor_dtype": str(tensor.dtype),
                        "src_device": src_device,
                        "dst_device": dst_device
                    }
                )
            
            result_tensor = self.fallback.copy_tensor_direct(src_device, dst_device, tensor)
        
        finally:
            # Record operation for monitoring
            if self.monitor:
                end_time = time.time()
                self.monitor.record_operation(
                    operation_type="direct_copy",
                    device_id=self.device,
                    peer_device=dst_device if self.device == src_device else src_device,
                    tensor=tensor,
                    start_time=start_time,
                    end_time=end_time,
                    algorithm="p2p_direct",
                    success=success
                )
            
            # Log debug event
            if self.debugger:
                self.debugger.log_event(
                    event_type="operation_end",
                    operation_type="direct_copy",
                    device_id=self.device,
                    peer_device=dst_device if self.device == src_device else src_device,
                    tensor=tensor,
                    message=f"Completed P2P direct copy: {'success' if success else 'failed'}"
                )
        
        return result_tensor

    def measure_p2p_bandwidth(self, src_device: int, dst_device: int, size_mb: int = 64, num_iterations: int = 10):
        """Measure P2P bandwidth between two devices."""
        if not self.use_p2p:
            return 0.0
        
        try:
            size_bytes = size_mb * 1024 * 1024
            bandwidth = ext.p2p_measure_bandwidth(src_device, dst_device, size_bytes, num_iterations, self.abort_flag)
            log_tp(self.device, f"P2P bandwidth {src_device}->{dst_device}: {bandwidth:.2f} GB/s")
            return bandwidth
        except Exception as e:
            log_tp(self.device, f"P2P bandwidth measurement failed: {e}")
            return 0.0

    def measure_p2p_latency(self, src_device: int, dst_device: int, size_kb: int = 4, num_iterations: int = 100):
        """Measure P2P latency between two devices."""
        if not self.use_p2p:
            return 0.0
        
        try:
            size_bytes = size_kb * 1024
            latency = ext.p2p_measure_latency(src_device, dst_device, size_bytes, num_iterations, self.abort_flag)
            log_tp(self.device, f"P2P latency {src_device}->{dst_device}: {latency:.2f} μs")
            return latency
        except Exception as e:
            log_tp(self.device, f"P2P latency measurement failed: {e}")
            return 0.0

    def validate_p2p_memory_access(self, src_device: int, dst_device: int, size_bytes: int = 1024):
        """Validate that P2P memory access is working between two devices."""
        if not self.use_p2p or not self.p2p_topology:
            return False
        
        try:
            # Create test tensors
            src_tensor = torch.zeros(size_bytes // 4, dtype=torch.float32, device=src_device)
            dst_tensor = torch.zeros(size_bytes // 4, dtype=torch.float32, device=dst_device)
            
            # Test validation
            is_valid = ext.p2p_validate_memory_access(
                src_device, dst_device,
                src_tensor.data_ptr(), dst_tensor.data_ptr(),
                size_bytes, self.abort_flag
            )
            
            log_tp(self.device, f"P2P memory access validation {src_device}->{dst_device}: {'OK' if is_valid else 'FAILED'}")
            return is_valid
        except Exception as e:
            log_tp(self.device, f"P2P memory access validation failed: {e}")
            return False

    def get_memory_pool_stats(self):
        """Get detailed statistics about P2P memory pool usage."""
        if not self.use_p2p:
            return {}
        
        try:
            usage = ext.p2p_get_direct_pool_usage(self.device, self.abort_flag)
            total_size = ext.p2p_get_direct_pool_size(self.device, self.abort_flag)
            
            # Get GPU memory stats
            import torch
            gpu_props = torch.cuda.get_device_properties(self.device)
            total_memory = gpu_props.total_memory
            allocated_memory = torch.cuda.memory_allocated(self.device)
            cached_memory = torch.cuda.memory_reserved(self.device)
            
            stats = {
                'device': self.device,
                'pool_usage_bytes': usage,
                'pool_total_bytes': total_size,
                'pool_usage_percent': (usage / total_size * 100.0) if total_size > 0 else 0.0,
                'gpu_total_memory': total_memory,
                'gpu_allocated_memory': allocated_memory,
                'gpu_cached_memory': cached_memory,
                'gpu_free_memory': total_memory - allocated_memory,
                'pool_efficiency': (usage / allocated_memory * 100.0) if allocated_memory > 0 else 0.0
            }
            
            # Add topology information
            if self.p2p_topology:
                topology_stats = self.p2p_topology.get_topology_summary()
                stats.update({
                    'topology_connectivity_ratio': topology_stats.get('connectivity_ratio', 0),
                    'topology_is_fully_connected': topology_stats.get('is_fully_connected', False),
                    'topology_peer_count': topology_stats.get('peer_count', 0)
                })
            
            log_tp(self.device, f"P2P memory pool stats: {stats}")
            return stats
        except Exception as e:
            log_tp(self.device, f"P2P memory pool stats failed: {e}")
            return {}

    def enable_peer_access(self, peer_device: int):
        """Enable P2P access to a peer device."""
        if not self.use_p2p:
            return False
        
        try:
            ext.p2p_enable_peer_access(self.device, peer_device, self.abort_flag)
            log_tp(self.device, f"P2P peer access enabled: {peer_device}")
            return True
        except Exception as e:
            log_tp(self.device, f"P2P peer access enable failed: {e}")
            return False

    def disable_peer_access(self, peer_device: int):
        """Disable P2P access to a peer device."""
        if not self.use_p2p:
            return False
        
        try:
            ext.p2p_disable_peer_access(self.device, peer_device, self.abort_flag)
            log_tp(self.device, f"P2P peer access disabled: {peer_device}")
            return True
        except Exception as e:
            log_tp(self.device, f"P2P peer access disable failed: {e}")
            return False

    def is_peer_access_enabled(self, peer_device: int):
        """Check if P2P access to a peer device is enabled."""
        if not self.use_p2p:
            return False
        
        try:
            is_enabled = ext.p2p_is_peer_access_enabled(self.device, peer_device, self.abort_flag)
            log_tp(self.device, f"P2P peer access {peer_device}: {'enabled' if is_enabled else 'disabled'}")
            return is_enabled
        except Exception as e:
            log_tp(self.device, f"P2P peer access check failed: {e}")
            return False


class TPBackendNCCL:

    def __init__(
        self,
        device: int,
        active_devices: list[int],
        output_device: int,
        init_method: str,
        master: bool,
        uuid: str,
        shbuf_size: int = SHBUF_SIZE,
    ):
        self.device = device
        if device < 0:
            log_tp(device, f"NCCL init: skip CPU process")
            return

        self.active_devices = active_devices
        self.world_size = len(active_devices)
        self.rank = active_devices.index(device)

        log_tp(device, f"NCCL init: world_size {self.world_size}, rank {self.rank}, device {device}, init_method {init_method}")
        print(f" -- NCCL init: world_size {self.world_size}, rank {self.rank}, device {device}, init_method {init_method}")
        dist.init_process_group(
            "nccl",
            rank = self.rank,
            world_size = self.world_size,
            init_method = init_method,
        )
        self.mp_warmup_nccl(device)
        self.fallback = TPBackendNative(
            device,
            active_devices,
            output_device,
            init_method,
            master,
            uuid,
            shbuf_size
        )


    def mp_warmup_nccl(self, device):
        """
        NCCL does lazy initialization which causes the first reduction operation to take an exceedingly long time
        (20+ seconds). This seems to lead to race conditions or timeouts if it happens during a forward pass. Called
        by TP loader as soon as processes are spawned and process group is initialized.
        """
        print(f" -- NCCL warmup, device {device}, please wait...")
        x = torch.ones((6,), device = device)
        dist.all_reduce(x)
        print(f" -- Finished NCCL warmup, device {device}")


    def close(self):
        if self.device < 0:
            log_tp(self.device, f"NCCL close: skip CPU process")
            return

        dist.barrier()
        self.fallback.close()
        dist.destroy_process_group()


    def fwd_barrier(self):
        dist.barrier()


    def broadcast(self, tensor: torch.Tensor, src_device: int):
        self.fallback.broadcast(tensor, src_device)
        # src_rank = self.active_devices.index(src_device)
        # dist.broadcast(tensor, src = src_rank)


    def all_reduce(self, tensor: torch.Tensor, contribution: bool = True):
        if tensor.dtype == torch.float32:
            temp = tensor.to(torch.bfloat16)
            dist.all_reduce(temp, async_op = False)
            temp = temp.to(torch.float32)
            tensor.copy_(temp)
        else:
            dist.all_reduce(tensor, async_op = False)


    def gather(
        self,
        tensor: torch.Tensor,
        out_tensor: torch.Tensor | None,
        gather_devices: torch.Tensor | None,
        out_device: int,
        ldims: list[int]
    ):
        self.fallback.gather(tensor, out_tensor, gather_devices, out_device, ldims)
        # dst_rank = self.active_devices.index(out_device)
        # d_ldims = [0] * (max(self.active_devices) + 1)
        # for d, m in zip(gather_devices, ldims):
        #     d_ldims[d] = m
        # ldims = [d_ldims[d] for d in self.active_devices]
        #
        # if self.rank == dst_rank:
        #     od = 0
        #     for src, ldim in enumerate(ldims):
        #         if ldim == 0:
        #             continue
        #         out_slice = out_tensor[..., od : od + ldim]
        #         od += ldim
        #         if src == self.rank:
        #             out_slice.copy(tensor)
        #         else:
        #             # print(f"rank {self.rank} recv {out_slice.shape[-1]} from {src}")
        #             rbuf = torch.empty_like(out_slice)
        #             dist.recv(rbuf, src = src)
        #             out_slice.copy_(rbuf)
        # elif tensor.shape[-1] > 0:
        #     # print(f"rank {self.rank} send {tensor.shape[-1]} to {dst_rank}")
        #     dist.send(tensor, dst = dst_rank)


    def run_cpu_reduce_jobs(self):
        pass


    def end_cpu_reduce_jobs(self):
        pass


class TPBackendNative:

    def __init__(
        self,
        device: int,
        active_devices: list[int],
        output_device: int,
        init_method: str,
        master: bool,
        uuid: str,
        shbuf_size: int = SHBUF_SIZE,
        cpu: bool = False
    ):
        self.uuid = uuid
        self.shm_g_name = uuid + "_g"
        self.shm_b_name = uuid + "_b"
        self.shm_r_name = uuid + "_r"
        self.shm_s_name = uuid + "_s"
        self.device = device
        self.max_num_devices = max(active_devices) + 1
        self.active_devices = active_devices
        self.shbuf_size = shbuf_size
        self.master = master
        self.cpu = cpu
        self.cpu_is_pinned = False

        size_g = GLOBALS_SIZE
        size_b = self.shbuf_size
        size_r = SHBUF_SIZE_R
        size_s = SHBUF_SIZE_S

        if master:
            log_tp(device, f"Creating SHMs")
            self.shm_g = shared_memory.SharedMemory(create = True, size = size_g, name = self.shm_g_name)
            log_tp(device, f"Created SHM: {self.shm_g_name}, {size_g} bytes")
            self.shm_b = shared_memory.SharedMemory(create = True, size = size_b, name = self.shm_b_name)
            log_tp(device, f"Created SHM: {self.shm_b_name}, {size_b} bytes")
            self.shm_r = shared_memory.SharedMemory(create = True, size = size_r, name = self.shm_r_name)
            log_tp(device, f"Created SHM: {self.shm_r_name}, {size_r} bytes")
            self.shm_s = shared_memory.SharedMemory(create = True, size = size_s, name = self.shm_s_name)
            log_tp(device, f"Created SHM: {self.shm_s_name}, {size_s} bytes")
            self.buf_g = np.ndarray((size_g,), dtype = np.uint8, buffer = self.shm_g.buf)
            self.buf_b = np.ndarray((size_b,), dtype = np.uint8, buffer = self.shm_b.buf)
            self.buf_r = np.ndarray((size_r,), dtype = np.uint8, buffer = self.shm_r.buf)
            self.buf_s = np.ndarray((size_s,), dtype = np.uint8, buffer = self.shm_s.buf)
            self.buf_g[:] = 0
            self.buf_b[: size_b: 4096] = 0
            self.buf_r[:] = 0
            self.buf_s[:] = 0
        else:
            self.shm_g = None
            self.shm_b = None
            self.shm_r = None
            self.shm_s = None
            deadline = time.time() + 15
            log_tp(device, f"Opening SHMs")
            first_fnf = True
            while True:
                try:
                    if self.shm_g is None:
                        self.shm_g = shared_memory.SharedMemory(name = self.shm_g_name)
                        log_tp(device, f"Opened SHM {self.shm_g_name}")
                    if self.shm_b is None:
                        self.shm_b = shared_memory.SharedMemory(name = self.shm_b_name)
                        log_tp(device, f"Opened SHM {self.shm_b_name}")
                    if self.shm_r is None:
                        self.shm_r = shared_memory.SharedMemory(name = self.shm_r_name)
                        log_tp(device, f"Opened SHM {self.shm_r_name}")
                    if self.shm_s is None:
                        self.shm_s = shared_memory.SharedMemory(name = self.shm_s_name)
                        log_tp(device, f"Opened SHM {self.shm_s_name}")
                    break
                except FileNotFoundError:
                    if first_fnf:
                        log_tp(device, f"Waiting for SHM to appear")
                        first_fnf = False
                    if time.time() > deadline:
                        log_tp(device, f"Timeout opening SHM")
                        raise TimeoutError("Timeout waiting for master process to create SHM")
                    time.sleep(0.05)

        # Create local tensors/flags
        if self.device >= 0:
            self.abort_flag = torch.zeros((1,), device = self.device, dtype = torch.int)
        else:
            self.abort_flag = None

        # Create pinned, shared tensors
        def get_local_tensor(shm_buf, _buffer_size):
            np_view = np.ndarray(
                shape = (_buffer_size,),
                dtype = np.uint8,
                buffer = shm_buf,
                offset = 0,
            )
            return torch.as_tensor(np_view)
        self.tensor_g = get_local_tensor(self.shm_g.buf, size_g)
        self.tensor_b = get_local_tensor(self.shm_b.buf, size_b)
        self.tensor_r = get_local_tensor(self.shm_r.buf, size_r)
        self.tensor_s = get_local_tensor(self.shm_s.buf, size_s)
        self.ptr_g = self.tensor_g.data_ptr()
        self.ptr_b = self.tensor_b.data_ptr()
        self.ptr_r = self.tensor_r.data_ptr()
        self.ptr_s = self.tensor_s.data_ptr()
        if not self.cpu:
            log_tp(device, f"Host register G")
            cuda_host_register(self.ptr_g, self.tensor_g.numel(), flags = CUDA_HOST_REGISTER_PORTABLE)
            log_tp(device, f"Host register B")
            cuda_host_register(self.ptr_b, self.tensor_b.numel(), flags = CUDA_HOST_REGISTER_PORTABLE)
            log_tp(device, f"Host register R")
            cuda_host_register(self.ptr_r, self.tensor_r.numel(), flags = CUDA_HOST_REGISTER_PORTABLE)
            log_tp(device, f"Host register S")
            cuda_host_register(self.ptr_s, self.tensor_s.numel(), flags = CUDA_HOST_REGISTER_PORTABLE)

        # Init global context
        if master:
            log_tp(device, f"Initializing global context")
            ext.pg_init_context(self.ptr_g)


    def close(self):
        if not self.cpu:
            log_tp(self.device, f"Host unregister G")
            cuda_host_unregister(self.ptr_g)
            log_tp(self.device, f"Host unregister B")
            cuda_host_unregister(self.ptr_b)
            log_tp(self.device, f"Host unregister R")
            cuda_host_unregister(self.ptr_r)
            log_tp(self.device, f"Host unregister S")
            cuda_host_unregister(self.ptr_s)
        self.shm_g.close()
        log_tp(self.device, f"Closed {self.shm_g_name}")
        self.shm_b.close()
        log_tp(self.device, f"Closed {self.shm_b_name}")
        self.shm_r.close()
        log_tp(self.device, f"Closed {self.shm_r_name}")
        self.shm_s.close()
        log_tp(self.device, f"Closed {self.shm_s_name}")
        if self.master:
            log_tp(self.device, f"Master unlink G")
            self.shm_g.unlink()
            log_tp(self.device, f"Master unlink B")
            self.shm_b.unlink()
            log_tp(self.device, f"Master unlink R")
            self.shm_r.unlink()
            log_tp(self.device, f"Master unlink S")
            self.shm_s.unlink()


    def fwd_barrier(self):
        ext.pg_barrier(self.ptr_g, self.active_devices, self.device, self.abort_flag)


    def broadcast(self, tensor: torch.Tensor, src_device: int):
        if tensor.numel() * tensor.element_size() <= 2048:
            ext.pg_broadcast_ll(
                self.ptr_g,
                self.active_devices,
                self.device,
                src_device,
                tensor,
                self.ptr_s,
                SHBUF_SIZE_S,
                self.abort_flag
            )
        else:
            ext.pg_broadcast(
                self.ptr_g,
                self.active_devices,
                self.device,
                src_device,
                tensor,
                self.ptr_b,
                self.shbuf_size,
                self.abort_flag
            )


    def all_reduce(self, tensor: torch.Tensor, contribution: bool = True):
        """Optimized all_reduce with adaptive algorithm selection."""
        # Ensure tensor is contiguous for better performance
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        tensor_size = tensor.numel() * tensor.element_size()
        
        # Use CPU reduction for smaller tensors to avoid GPU overhead
        if tensor_size < MAX_CPU_REDUCE:
            ext.pg_all_reduce_cpu(
                self.ptr_g,
                self.active_devices,
                self.device,
                self.active_devices[0],
                tensor,
                contribution,
                self.ptr_r,
                SHBUF_SIZE_R,
                self.master,
                self.abort_flag
            )
        else:
            # Use GPU reduction for larger tensors
            # Optimize dtype for better bandwidth
            original_dtype = tensor.dtype
            if tensor.dtype == torch.float32 and tensor_size > 4 * 1024 * 1024:  # > 4MB
                tensor = tensor.half()
                contribution = False  # Don't contribute twice
            
            ext.pg_all_reduce(
                self.ptr_g,
                self.active_devices,
                self.device,
                self.active_devices[0],
                tensor,
                self.ptr_b,
                self.shbuf_size,
                self.abort_flag
            )
            
            # Convert back to original dtype if needed
            if original_dtype == torch.float32 and tensor.dtype != original_dtype:
                tensor = tensor.float()


    def gather(
        self,
        tensor: torch.Tensor,
        out_tensor: torch.Tensor | None,
        gather_devices: torch.Tensor | None,
        out_device: int,
        ldims: list[int]
    ):
        if out_device == self.device:
            assert out_tensor is not None, \
                f"Gather: Output device must supply output tensor"
            assert out_tensor.shape[-1] == sum(ldims), \
                f"Gather: Output tensor must match size of concatenated slices: {sum(ldims)}"

        ext.pg_gather(
            self.ptr_g,
            gather_devices,
            self.device,
            out_device,
            tensor,
            out_tensor,
            ldims,
            self.ptr_b,
            self.shbuf_size,
            self.abort_flag
        )


    def run_cpu_reduce_jobs(self):
        """Run CPU reduce jobs with optimized memory management."""
        if not self.cpu_is_pinned and self.cpu:
            # Pin CPU process for better performance
            try:
                import os
                # Set process priority
                os.nice(-5)  # Higher priority for CPU process
                self.cpu_is_pinned = True
            except:
                pass
        
        ext.run_cpu_reduce_jobs(
            self.ptr_g,
            self.ptr_r,
            SHBUF_SIZE_R,
        )


    def end_cpu_reduce_jobs(self):
        if self.master:
            ext.end_cpu_reduce_jobs(
                self.ptr_g,
            )