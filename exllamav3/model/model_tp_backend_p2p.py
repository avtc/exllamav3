import torch
import time
import numpy as np
from ctypes import POINTER, c_void_p, c_uint32, c_uint64
from .model_tp_cuda import (
    cuda_host_register,
    cuda_host_unregister,
    CUDA_HOST_REGISTER_PORTABLE,
    check_p2p_connectivity,
    enable_p2p_access,
    disable_p2p_access
)
from ..ext import exllamav3_ext as ext
from .model_tp_backend import TPBackend, SHBUF_SIZE, SHBUF_SIZE_R, SHBUF_SIZE_S
from multiprocessing import shared_memory
from ..util import log_tp

class TPBackendP2P(TPBackend):
    """
    P2P-optimized tensor parallel backend for fully connected GPU systems.
    
    This backend provides high-performance GPU-to-GPU communication using
    peer-to-peer (P2P) memory access when all GPUs in the system are fully
    connected. It offers significant performance improvements over traditional
    NCCL and native backends for tensor parallel inference.
    
    Key Features:
        - Direct GPU-to-GPU communication without CPU mediation
        - Optimized all-reduce, broadcast, and gather operations
        - Automatic P2P connectivity validation
        - Graceful fallback to other backends if P2P unavailable
        - Shared memory integration for large tensor transfers
    
    Performance Benefits:
        - Reduced latency by eliminating CPU bottlenecks
        - Higher throughput for fully connected GPU systems
        - Better scalability for 2-8 GPU configurations
        - Optimized for transformer model tensor parallel operations
    
    Usage:
        This backend is typically created through the factory function
        `create_tp_backend()` with backend_type='p2p' or 'auto'.
        
    Example:
        >>> from exllamav3.model.model_tp_backend import create_tp_backend
        >>> backend = create_tp_backend(
        ...     backend_type="p2p",
        ...     device=0,
        ...     active_devices=[0, 1, 2],
        ...     output_device=0,
        ...     init_method="tcp://127.0.0.1:29500",
        ...     master=True,
        ...     uuid="example-p2p-uuid"
        ... )
        >>> # Use backend for tensor parallel operations
        
    Note:
        - Requires full bidirectional P2P connectivity between all GPUs
        - Automatically enabled/disabled P2P access during initialization/cleanup
        - Uses shared memory buffers for communication coordination
        - Compatible with existing tensor parallel workflows
    """
    
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
        """
        Initialize P2P tensor parallel backend.
        
        Args:
            device: Current device ID (integer)
            active_devices: List of active device IDs participating in tensor parallelism
            output_device: Device ID where results will be collected
            init_method: Initialization method string (typically TCP endpoint)
            master: Whether this process is the master (controls shared memory creation)
            uuid: Unique identifier for shared memory allocation
            shbuf_size: Size of shared memory buffers in bytes (default: SHBUF_SIZE)
            
        Raises:
            RuntimeError: If P2P connectivity check fails or initialization fails
            
        Note:
            - Validates P2P connectivity between all devices during initialization
            - Automatically enables P2P access for all device pairs
            - Creates shared memory buffers for coordination
            - Initializes P2P context for optimized communication
        """
        self.device = device
        if device < 0:
            log_tp(device, f"P2P init: skip CPU process")
            return

        self.active_devices = active_devices
        self.world_size = len(active_devices)
        self.rank = active_devices.index(device)
        self.output_device = output_device
        self.init_method = init_method
        self.master = master
        self.uuid = uuid
        self.shbuf_size = shbuf_size
        
        # Validate P2P connectivity
        if not check_p2p_connectivity(active_devices):
            log_tp(device, f"P2P connectivity check failed for devices {active_devices}")
            raise RuntimeError(
                f"P2P backend requires full peer connectivity between all GPUs, "
                f"but connectivity check failed for devices {active_devices}"
            )
        
        log_tp(device, f"P2P init: world_size {self.world_size}, rank {self.rank}, device {device}")
        print(f" -- P2P init: world_size {self.world_size}, rank {self.rank}, device {device}")
        
        # Enable P2P access between all GPUs
        try:
            enable_p2p_access(active_devices)
            log_tp(device, f"P2P access enabled for all device pairs")
        except Exception as e:
            log_tp(device, f"Failed to enable P2P access: {e}")
            raise RuntimeError(f"Failed to enable P2P access: {e}")
        
        # Initialize shared memory buffers for P2P communication
        self._init_shmem_buffers()
        
        # Initialize P2P context
        self._init_p2p_context()
        
        # P2P-specific buffers
        self.p2p_buffer = None
        self.p2p_buffer_size = max(1024 * 1024, shbuf_size // 4)  # 1MB minimum or shbuf_size/4
        if device >= 0:
            self.p2p_buffer = torch.zeros(
                (self.p2p_buffer_size // 4,),  # Divide by 4 for float32
                dtype=torch.float32,
                device=device
            )
        
        # P2P context management
        self.p2p_context = None
        self.p2p_initialized = False
    
    def _init_shmem_buffers(self):
        """Initialize shared memory buffers for P2P communication."""
        self.shm_g_name = self.uuid + "_p2p_g"
        self.shm_b_name = self.uuid + "_p2p_b"
        self.shm_r_name = self.uuid + "_p2p_r"
        self.shm_s_name = self.uuid + "_p2p_s"
        
        size_g = 128 * 1024  # Globals buffer
        size_b = self.shbuf_size  # Buffer for P2P operations
        size_r = SHBUF_SIZE_R  # Reduction buffer
        size_s = SHBUF_SIZE_S  # Small buffer
        
        if self.master:
            log_tp(self.device, f"Creating P2P SHMs")
            self.shm_g = shared_memory.SharedMemory(create=True, size=size_g, name=self.shm_g_name)
            self.shm_b = shared_memory.SharedMemory(create=True, size=size_b, name=self.shm_b_name)
            self.shm_r = shared_memory.SharedMemory(create=True, size=size_r, name=self.shm_r_name)
            self.shm_s = shared_memory.SharedMemory(create=True, size=size_s, name=self.shm_s_name)
            log_tp(self.device, f"Created P2P SHMs: {self.shm_g_name}, {self.shm_b_name}, {self.shm_r_name}, {self.shm_s_name}")
        else:
            # Connect to existing shared memory
            deadline = time.time() + 15
            log_tp(self.device, f"Opening P2P SHMs")
            first_fnf = True
            
            while True:
                try:
                    if not hasattr(self, 'shm_g') or self.shm_g is None:
                        self.shm_g = shared_memory.SharedMemory(name=self.shm_g_name)
                        log_tp(self.device, f"Opened P2P SHM {self.shm_g_name}")
                    if not hasattr(self, 'shm_b') or self.shm_b is None:
                        self.shm_b = shared_memory.SharedMemory(name=self.shm_b_name)
                        log_tp(self.device, f"Opened P2P SHM {self.shm_b_name}")
                    if not hasattr(self, 'shm_r') or self.shm_r is None:
                        self.shm_r = shared_memory.SharedMemory(name=self.shm_r_name)
                        log_tp(self.device, f"Opened P2P SHM {self.shm_r_name}")
                    if not hasattr(self, 'shm_s') or self.shm_s is None:
                        self.shm_s = shared_memory.SharedMemory(name=self.shm_s_name)
                        log_tp(self.device, f"Opened P2P SHM {self.shm_s_name}")
                    break
                except FileNotFoundError:
                    if first_fnf:
                        log_tp(self.device, f"Waiting for P2P SHM to appear")
                        first_fnf = False
                    if time.time() > deadline:
                        log_tp(self.device, f"Timeout opening P2P SHM")
                        raise TimeoutError("Timeout waiting for master process to create P2P SHM")
                    time.sleep(0.05)
        
        # Create local tensors from shared memory
        def get_local_tensor(shm_buf, buffer_size):
            np_view = np.ndarray(
                shape=(buffer_size,),
                dtype=np.uint8,
                buffer=shm_buf,
                offset=0,
            )
            return torch.as_tensor(np_view)
        
        self.tensor_g = get_local_tensor(self.shm_g.buf, size_g)
        self.tensor_b = get_local_tensor(self.shm_b.buf, size_b)
        self.tensor_r = get_local_tensor(self.shm_r.buf, size_r)
        self.tensor_s = get_local_tensor(self.shm_s.buf, size_s)
        
        # Get pointers for CUDA operations
        self.ptr_g = self.tensor_g.data_ptr()
        self.ptr_b = self.tensor_b.data_ptr()
        self.ptr_r = self.tensor_r.data_ptr()
        self.ptr_s = self.tensor_s.data_ptr()
        
        # Register host memory for CUDA access
        if not self.device < 0:
            log_tp(self.device, f"Host register P2P G")
            cuda_host_register(self.ptr_g, self.tensor_g.numel(), flags=CUDA_HOST_REGISTER_PORTABLE)
            log_tp(self.device, f"Host register P2P B")
            cuda_host_register(self.ptr_b, self.tensor_b.numel(), flags=CUDA_HOST_REGISTER_PORTABLE)
            log_tp(self.device, f"Host register P2P R")
            cuda_host_register(self.ptr_r, self.tensor_r.numel(), flags=CUDA_HOST_REGISTER_PORTABLE)
            log_tp(self.device, f"Host register P2P S")
            cuda_host_register(self.ptr_s, self.tensor_s.numel(), flags=CUDA_HOST_REGISTER_PORTABLE)
    
    def _init_p2p_context(self):
        """Initialize P2P communication context using new P2P kernels."""
        if self.master:
            log_tp(self.device, f"Initializing P2P context")
            ext.pg_init_context(self.ptr_g)
            
        # Initialize P2P context using new P2P utilities
        try:
            # Convert device IDs to appropriate type for P2P operations
            # Use c_uint64 as uintptr_t equivalent for 64-bit systems
            uintptr_devices = [c_uint64(d) for d in self.active_devices]
            
            # Initialize P2P context
            self.p2p_context = ext.init_p2p_context(uintptr_devices, self.p2p_buffer_size)
            self.p2p_initialized = True
            log_tp(self.device, f"P2P context initialized successfully")
            
        except Exception as e:
            log_tp(self.device, f"Failed to initialize P2P context: {e}")
            # Fail fast if P2P connectivity was detected but initialization failed
            raise RuntimeError(
                f"P2P connectivity was detected and enabled, but P2P context initialization failed. "
                f"This indicates a system configuration issue that prevents P2P communication. "
                f"Error: {e}"
            ) from e
    
    def close(self):
        """
        Clean up P2P resources and disable P2P access.
        
        This method should be called when the P2P backend is no longer needed.
        It properly cleans up all resources including:
        - Destroys P2P context
        - Disables P2P access between all GPUs
        - Unregisters host memory
        - Closes and unlinks shared memory buffers
        
        Note:
            - Safe to call multiple times (idempotent)
            - Automatically handles cleanup errors gracefully
            - Should be called during model unloading or process shutdown
        """
        if self.device < 0:
            log_tp(self.device, f"P2P close: skip CPU process")
            return
        
        # Destroy P2P context if initialized
        if self.p2p_initialized and self.p2p_context is not None:
            try:
                ext.destroy_p2p_context(self.p2p_context)
                log_tp(self.device, f"P2P context destroyed")
            except Exception as e:
                log_tp(self.device, f"Failed to destroy P2P context: {e}")
            finally:
                self.p2p_context = None
                self.p2p_initialized = False
        
        # Disable P2P access
        try:
            disable_p2p_access(self.active_devices)
            log_tp(self.device, f"P2P access disabled for all device pairs")
        except Exception as e:
            log_tp(self.device, f"Failed to disable P2P access: {e}")
        
        # Unregister host memory
        if not self.device < 0:
            log_tp(self.device, f"Host unregister P2P G")
            cuda_host_unregister(self.ptr_g)
            log_tp(self.device, f"Host unregister P2P B")
            cuda_host_unregister(self.ptr_b)
            log_tp(self.device, f"Host unregister P2P R")
            cuda_host_unregister(self.ptr_r)
            log_tp(self.device, f"Host unregister P2P S")
            cuda_host_unregister(self.ptr_s)
        
        # Close shared memory
        if hasattr(self, 'shm_g'):
            self.shm_g.close()
            log_tp(self.device, f"Closed {self.shm_g_name}")
        if hasattr(self, 'shm_b'):
            self.shm_b.close()
            log_tp(self.device, f"Closed {self.shm_b_name}")
        if hasattr(self, 'shm_r'):
            self.shm_r.close()
            log_tp(self.device, f"Closed {self.shm_r_name}")
        if hasattr(self, 'shm_s'):
            self.shm_s.close()
            log_tp(self.device, f"Closed {self.shm_s_name}")
        
        # Unlink shared memory (master only)
        if self.master:
            if hasattr(self, 'shm_g'):
                self.shm_g.unlink()
                log_tp(self.device, f"Master unlink {self.shm_g_name}")
            if hasattr(self, 'shm_b'):
                self.shm_b.unlink()
                log_tp(self.device, f"Master unlink {self.shm_b_name}")
            if hasattr(self, 'shm_r'):
                self.shm_r.unlink()
                log_tp(self.device, f"Master unlink {self.shm_r_name}")
            if hasattr(self, 'shm_s'):
                self.shm_s.unlink()
                log_tp(self.device, f"Master unlink {self.shm_s_name}")
        
    
    def fwd_barrier(self):
        """
        P2P-optimized barrier synchronization using new P2P kernels.
        
        This method implements a barrier synchronization across all processes
        using P2P-optimized kernels. It ensures all processes reach this
        point before proceeding, which is essential for tensor parallel
        operations.
        
        Note:
            - Uses P2P-specific barrier implementation for better performance
            - Handles abort flags for error detection
            - Automatically manages device context switching
        """
        # Use P2P-optimized barrier
        uintptr_devices = [c_uint64(d) for d in self.active_devices]
        abort_flag = torch.zeros((1,), device=self.device, dtype=torch.int32)
        ext.pg_barrier_full_p2p(
            self.p2p_context,
            uintptr_devices,
            self.device,
            abort_flag
        )
    
    def broadcast(self, tensor: torch.Tensor, src_device: int):
        """
        P2P-optimized broadcast operation using new P2P kernels.
        
        Broadcasts a tensor from the source device to all other devices
        in the tensor parallel group using direct P2P communication.
        
        Args:
            tensor: PyTorch tensor to broadcast (must be on current device)
            src_device: Source device ID from which to broadcast the tensor
            
        Note:
            - Uses P2P-optimized direct transfer for large tensors
            - Falls back to shared memory broadcast for small tensors (< 2KB)
            - Automatically handles device context switching
            - Supports non-contiguous tensors through automatic copying
        """
        # For small tensors, use shared memory broadcast
        if tensor.numel() * tensor.element_size() <= 2048:
            ext.pg_broadcast_ll(
                self.ptr_g,
                self.active_devices,
                self.device,
                src_device,
                tensor,
                self.ptr_s,
                SHBUF_SIZE_S,
                None
            )
        else:
            # Use P2P-optimized direct transfer with new kernels
            uintptr_devices = [c_uint64(d) for d in self.active_devices]
            abort_flag = torch.zeros((1,), device=self.device, dtype=torch.int32)
            ext.pg_broadcast_full_p2p(
                self.p2p_context,
                uintptr_devices,
                self.device,
                src_device,
                tensor,
                self.ptr_b,
                self.shbuf_size,
                abort_flag
            )
    
    
    def all_reduce(self, tensor: torch.Tensor, contribution: bool = True):
        """
        P2P-optimized all-reduce operation using new P2P kernels.
        
        Performs an all-reduce operation across all devices in the tensor
        parallel group using direct P2P communication. This is a key
        operation for tensor parallel inference.
        
        Args:
            tensor: PyTorch tensor to all-reduce (modified in-place)
            contribution: Whether this tensor contributes to the reduction
                         (currently always True, but reserved for future use)
            
        Note:
            - Uses P2P-optimized all-reduce implementation for best performance
            - Supports float32 tensors (automatically handles bfloat16 conversion)
            - Operates in-place on the input tensor
            - Automatically manages device context and error handling
        """
        # Use P2P-optimized all-reduce with new kernels
        uintptr_devices = [c_uint64(d) for d in self.active_devices]
        abort_flag = torch.zeros((1,), device=self.device, dtype=torch.int32)
        ext.pg_all_reduce_full_p2p(
            self.p2p_context,
            uintptr_devices,
            self.device,
            self.active_devices[0],  # master_device
            tensor,
            self.ptr_b,
            self.shbuf_size,
            abort_flag
        )
    
    
    def gather(
        self,
        tensor: torch.Tensor,
        out_tensor: torch.Tensor | None,
        gather_devices: torch.Tensor | None,
        out_device: int,
        ldims: list[int]
    ):
        """
        P2P-optimized gather operation using new P2P kernels.
        
        Gathers tensors from all devices to the output device using
        direct P2P communication. This is used to collect results
        from tensor parallel operations.
        
        Args:
            tensor: Tensor to gather (from current device)
            out_tensor: Output tensor (must be provided if on output device)
            gather_devices: Devices to gather from (tensor of device IDs)
            out_device: Output device ID where results should be collected
            ldims: Local dimensions for each device (list of integers)
            
        Raises:
            ValueError: If output tensor dimensions don't match expected size
            
        Note:
            - If current device is output device: receives from all other devices
            - If current device is not output device: sends tensor to output device
            - Uses P2P-optimized gather implementation for performance
            - Handles tensor dimension validation automatically
        """
        if out_device == self.device:
            # This is the output device - receive from all others using P2P
            if out_tensor is None:
                raise ValueError("Gather: Output device must supply output tensor")
            
            if out_tensor.shape[-1] != sum(ldims):
                raise ValueError(f"Gather: Output tensor must match size of concatenated slices: {sum(ldims)}")
            
            # Use P2P gather with new kernels
            uintptr_devices = [c_uint64(d) for d in self.active_devices]
            abort_flag = torch.zeros((1,), device=self.device, dtype=torch.int32)
            ext.pg_gather_full_p2p(
                self.p2p_context,
                uintptr_devices,
                self.device,
                out_device,
                tensor,
                out_tensor,
                ldims,
                self.ptr_b,
                self.shbuf_size,
                abort_flag
            )
        else:
            # This is not the output device - send to output device using P2P
            self._p2p_send_to_output(tensor, out_device)
    
    def _p2p_gather(self, tensor: torch.Tensor, out_tensor: torch.Tensor, 
                   gather_devices: torch.Tensor, ldims: list[int]):
        """P2P-optimized gather implementation."""
    
    

