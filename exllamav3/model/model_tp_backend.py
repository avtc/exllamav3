import torch
import torch.distributed as dist
import time
import numpy as np
from .model_tp_cuda import (
    cuda_host_register,
    cuda_host_unregister,
    CUDA_HOST_REGISTER_PORTABLE,
    check_p2p_connectivity,
    enable_p2p_access,
    disable_p2p_access
)
from ..ext import exllamav3_ext as ext
from multiprocessing import shared_memory, Barrier
from ..util import log_tp

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
        # if tensor.numel() * 2 < MAX_CPU_REDUCE:
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
        # else:
        #     ext.pg_all_reduce(
        #         self.ptr_g,
        #         self.active_devices,
        #         self.device,
        #         self.active_devices[0],
        #         tensor,
        #         self.ptr_b,
        #         self.shbuf_size,
        #         self.abort_flag
        #     )


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
        # if not self.cpu_is_pinned:
        #     set_process_priority_and_affinity()
        #     self.cpu_is_pinned = True
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


# Import TPBackendP2P for P2P connectivity support
try:
    from .model_tp_backend_p2p import TPBackendP2P
    TPBackendP2P_AVAILABLE = True
except ImportError:
    TPBackendP2P_AVAILABLE = False
    TPBackendP2P = None


def create_tp_backend(
    backend_type: str,
    device: int,
    active_devices: list[int],
    output_device: int,
    init_method: str,
    master: bool,
    uuid: str,
    shbuf_size: int = SHBUF_SIZE,
) -> TPBackend:
    """
    Factory function to create tensor parallel backend with automatic P2P detection.
    
    This function provides a unified interface for creating different types of tensor
    parallel backends. It automatically handles P2P connectivity detection
    when backend_type is set to 'auto'.
    
    Args:
        backend_type: Type of backend ('nccl', 'native', 'p2p', 'auto')
            - 'auto': Automatically selects best available backend (P2P if available, then NCCL)
            - 'p2p': Force P2P backend (requires full P2P connectivity)
            - 'nccl': Force NCCL backend (standard multi-GPU communication)
            - 'native': Force native backend (CPU-mediated communication)
        device: Current device ID (integer)
        active_devices: List of active device IDs to use for tensor parallelism
        output_device: Output device ID where results will be collected
        init_method: Initialization method string (typically TCP endpoint)
        master: Whether this is the master process (controls shared memory creation)
        uuid: Unique identifier for shared memory allocation
        shbuf_size: Shared buffer size in bytes (default: SHBUF_SIZE)
        
    Returns:
        TPBackend: Initialized backend instance ready for tensor parallel operations
        
    Raises:
        ValueError: If backend type is not supported or invalid
        RuntimeError: If P2P connectivity check fails for P2P backend
        RuntimeError: If backend initialization fails for any reason
        
    Example:
        >>> # Automatic backend selection
        >>> backend = create_tp_backend(
        ...     backend_type="auto",
        ...     device=0,
        ...     active_devices=[0, 1, 2],
        ...     output_device=0,
        ...     init_method="tcp://127.0.0.1:29500",
        ...     master=True,
        ...     uuid="example-uuid"
        ... )
        >>> print(type(backend).__name__)
        TPBackendP2P  # If P2P connectivity is available
        
        >>> # Explicit P2P backend
        >>> backend = create_tp_backend(
        ...     backend_type="p2p",
        ...     device=0,
        ...     active_devices=[0, 1],
        ...     output_device=0,
        ...     init_method="tcp://127.0.0.1:29500",
        ...     master=True,
        ...     uuid="example-uuid"
        ... )
        
    Note:
        - The 'auto' backend type performs automatic P2P connectivity detection
        - P2P backend requires all GPUs to have bidirectional P2P access
        - NCCL backend falls back to TPBackendNative for certain operations
        - Native backend uses CPU-mediated communication between GPUs
    """
    if backend_type == "auto":
        # Auto-detect best backend based on system capabilities
        if device >= 0 and len(active_devices) > 1 and TPBackendP2P_AVAILABLE:
            # Check P2P connectivity first
            if check_p2p_connectivity(active_devices):
                print(f" -- Auto-detected P2P connectivity, using TPBackendP2P")
                return TPBackendP2P(
                    device=device,
                    active_devices=active_devices,
                    output_device=output_device,
                    init_method=init_method,
                    master=master,
                    uuid=uuid,
                    shbuf_size=shbuf_size
                )
            else:
                print(f" -- P2P connectivity not available, using TPBackendNCCL")
                return TPBackendNCCL(
                    device=device,
                    active_devices=active_devices,
                    output_device=output_device,
                    init_method=init_method,
                    master=master,
                    uuid=uuid,
                    shbuf_size=shbuf_size
                )
        else:
            print(f" -- Using TPBackendNCCL (no P2P or single device)")
            return TPBackendNCCL(
                device=device,
                active_devices=active_devices,
                output_device=output_device,
                init_method=init_method,
                master=master,
                uuid=uuid,
                shbuf_size=shbuf_size
            )
    
    elif backend_type == "p2p":
        if not TPBackendP2P_AVAILABLE:
            raise ValueError("TPBackendP2P is not available")
        if device < 0:
            raise ValueError("P2P backend requires GPU device")
        if not check_p2p_connectivity(active_devices):
            raise RuntimeError(
                f"P2P backend requires full peer connectivity between all GPUs, "
                f"but connectivity check failed for devices {active_devices}"
            )
        print(f" -- Using TPBackendP2P (explicit)")
        return TPBackendP2P(
            device=device,
            active_devices=active_devices,
            output_device=output_device,
            init_method=init_method,
            master=master,
            uuid=uuid,
            shbuf_size=shbuf_size
        )
    
    elif backend_type == "nccl":
        print(f" -- Using TPBackendNCCL (explicit)")
        return TPBackendNCCL(
            device=device,
            active_devices=active_devices,
            output_device=output_device,
            init_method=init_method,
            master=master,
            uuid=uuid,
            shbuf_size=shbuf_size
        )
    
    elif backend_type == "native":
        print(f" -- Using TPBackendNative (explicit)")
        return TPBackendNative(
            device=device,
            active_devices=active_devices,
            output_device=output_device,
            init_method=init_method,
            master=master,
            uuid=uuid,
            shbuf_size=shbuf_size
        )
    
    else:
        raise ValueError(f"Unsupported backend type: {backend_type}")


def get_available_backends() -> list[str]:
    """
    Get list of available backend types.
    
    This function returns a list of backend types that are currently available
    in the system. The list includes P2P backend if it's available,
    and always includes standard backends like NCCL and native.
    
    Returns:
        list[str]: List of available backend types, sorted with P2P first if available
        
    Example:
        >>> backends = get_available_backends()
        >>> print(backends)
        ['p2p', 'nccl', 'native', 'auto']  # If P2P is available
        >>> # or
        ['nccl', 'native', 'auto']  # If P2P is not available
        
    Note:
        - P2P backend is placed first in the list when available for auto-detection
        - 'auto' is always included as it represents automatic backend selection
        - The returned list can be used for UI options or configuration validation
    """
    backends = ["nccl", "native"]
    if TPBackendP2P_AVAILABLE:
        backends.insert(0, "p2p")  # P2P has priority for auto-detection
    backends.append("auto")
    return backends