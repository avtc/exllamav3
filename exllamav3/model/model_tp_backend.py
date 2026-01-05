import torch
import torch.distributed as dist
import time
import numpy as np
import os
from .model_tp_cuda import cuda_host_register, cuda_host_unregister, CUDA_HOST_REGISTER_PORTABLE
from ..ext import exllamav3_ext as ext
from multiprocessing import shared_memory
from ..util import log_tp

GLOBALS_SIZE = 128*1024
SHBUF_SIZE = 16 * 1024 ** 2
SHBUF_SIZE_S = 16 * 1024

# Default CPU reduce buffer size (will be scaled by multiplier)
DEFAULT_SHBUF_SIZE_R = 17 * 128 * 1024  # 2.1 MB


def get_cpu_reduce_buffer_size(multiplier: int = 1) -> int:
    """
    Calculate CPU all-reduce buffer size based on multiplier.

    OPT3: Larger buffer allows batching of multiple all-reduces,
    reducing CPU round-trip overhead.

    Args:
        multiplier: Buffer size multiplier (default: 1)

    Returns:
        Buffer size in bytes
    """
    return multiplier * DEFAULT_SHBUF_SIZE_R


# Initial buffer size (will be recalculated when backend is initialized)
SHBUF_SIZE_R = get_cpu_reduce_buffer_size(multiplier=1)
MAX_CPU_REDUCE = SHBUF_SIZE_R // 17 // 256 * 256


class OptimizationFlags:
    """
    Enable/disable specific optimizations for multi-GPU tensor parallelism.

    All flags can be controlled via environment variables:
    - EXLLAMA_TP_GPU_REDUCE: Enable GPU all-reduce (default: 1)
    - EXLLAMA_TP_GPU_REDUCE_THRESH: Threshold in elements (default: 65536)
    - EXLLAMA_TP_FUSED_REDUCE: Enable fused all-reduce (default: 1)
    - EXLLAMA_TP_CPU_BUFFER_MULT: CPU buffer multiplier (default: 4)
    """

    # OPT1: Use GPU all-reduce instead of CPU
    # GPU all-reduce stays entirely on GPU (faster with P2P)
    # CPU all-reduce goes through RAM (slower, but works without P2P)
    ENABLE_GPU_ALL_REDUCE = os.getenv("EXLLAMA_TP_GPU_REDUCE", "1") == "1"

    # Threshold for choosing GPU vs CPU all-reduce (in number of elements)
    # 0 = always use GPU, 65536 = 256KB for fp16 (default)
    GPU_ALL_REDUCE_THRESHOLD = int(os.getenv("EXLLAMA_TP_GPU_REDUCE_THRESH", "65536"))

    # OPT2: Fuse attention and MoE all-reduce into single operation
    # Combines 2 all-reduces per layer (attn + MLP/MoE) into 1 all-reduce
    # Reduces all-reduce frequency by ~50%
    ENABLE_FUSED_ALL_REDUCE = os.getenv("EXLLAMA_TP_FUSED_REDUCE", "1") == "1"

    # OPT3: CPU all-reduce buffer multiplier (when GPU path can't be used)
    # Larger buffer allows batching of multiple reductions, reducing CPU round-trips
    # Multiplier of 4 = 8.4 MB buffer (default: 4)
    # Larger buffers help when GPU all-reduce can't be used for small tensors
    CPU_REDUCE_BUFFER_MULTIPLIER = int(os.getenv("EXLLAMA_TP_CPU_BUFFER_MULT", "4"))

    # OPT5: Batched sampling to reduce GPU-CPU synchronization overhead
    # Processes all sequences in a job in a single batch instead of looping
    # Reduces GPU→CPU syncs from O(sequences) to O(1) per job
    # Expected speedup: 20-30% for multi-sequence jobs
    ENABLE_BATCHED_SAMPLING = os.getenv("EXLLAMA_BATCHED_SAMPLING", "1") == "1"

    @classmethod
    def log_settings(cls):
        """Log current optimization settings"""
        log_tp(-1, f"TP Optimization Settings:")
        log_tp(-1, f"  GPU all-reduce: {cls.ENABLE_GPU_ALL_REDUCE}")
        log_tp(-1, f"  GPU threshold: {cls.GPU_ALL_REDUCE_THRESHOLD} elements")
        log_tp(-1, f"  Fused all-reduce: {cls.ENABLE_FUSED_ALL_REDUCE}")
        buffer_mb = get_cpu_reduce_buffer_size(cls.CPU_REDUCE_BUFFER_MULTIPLIER) / (1024*1024)
        log_tp(-1, f"  CPU buffer: {buffer_mb:.1f} MB ({cls.CPU_REDUCE_BUFFER_MULTIPLIER}x default)")
        log_tp(-1, f"  Batched sampling: {cls.ENABLE_BATCHED_SAMPLING}")


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

    # Class-level counters for tracking all-reduce usage
    _gpu_reduce_count = 0
    _cpu_reduce_count = 0
    _fused_reduce_count = 0
    _total_bytes_gpu = 0
    _total_bytes_cpu = 0
    _total_bytes_fused = 0

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

        # OPT3: Calculate dynamic CPU reduce buffer size
        # Get buffer multiplier from OptimizationFlags
        buffer_multiplier = OptimizationFlags.CPU_REDUCE_BUFFER_MULTIPLIER

        size_g = GLOBALS_SIZE
        size_b = self.shbuf_size
        size_r = get_cpu_reduce_buffer_size(buffer_multiplier)  # Dynamic based on multiplier
        size_s = SHBUF_SIZE_S

        # Store buffer size as instance variable for use in all_reduce_cpu
        self.shbuf_size_r = size_r

        if master:
            # Log optimization settings
            OptimizationFlags.log_settings()

            log_tp(device, f"Creating SHMs")
            self.shm_g = shared_memory.SharedMemory(create = True, size = size_g, name = self.shm_g_name)
            log_tp(device, f"Created SHM: {self.shm_g_name}, {size_g} bytes")
            self.shm_b = shared_memory.SharedMemory(create = True, size = size_b, name = self.shm_b_name)
            log_tp(device, f"Created SHM: {self.shm_b_name}, {size_b} bytes")
            self.shm_r = shared_memory.SharedMemory(create = True, size = size_r, name = self.shm_r_name)
            log_tp(device, f"Created SHM: {self.shm_r_name}, {size_r} bytes (OPT3: {buffer_multiplier}x buffer)")
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


    def all_reduce(self, tensor: torch.Tensor, contribution: bool = True, is_fused: bool = False):
        """
        All-reduce operation with configurable CPU/GPU backend.

        GPU path (OPT1): Direct GPU-to-GPU via P2P/PCIe, no CPU involvement
        CPU path: GPU → RAM → CPU → RAM → GPU (slower, but works without P2P)
        Fused (OPT2): Combines attention + MLP/MoE all-reduce into one operation

        Backend selection is controlled by:
        - OptimizationFlags.ENABLE_GPU_ALL_REDUCE
        - OptimizationFlags.GPU_ALL_REDUCE_THRESHOLD

        Args:
            tensor: Input tensor to all-reduce
            contribution: Whether this rank contributes to the sum (CPU path only)
            is_fused: True if this is a fused attention+MLP all-reduce (OPT2)
        """

        # Determine if we should use GPU all-reduce
        use_gpu_reduce = (
            OptimizationFlags.ENABLE_GPU_ALL_REDUCE and
            tensor.numel() >= OptimizationFlags.GPU_ALL_REDUCE_THRESHOLD
        )

        tensor_bytes = tensor.numel() * tensor.element_size()

        if use_gpu_reduce:
            # GPU-based ring all-reduce (direct GPU-to-GPU, can use P2P)
            # This is the fast path! No CPU involvement.
            # Only limitation: requires P2P or NVLink for optimal performance

            # Track statistics
            if is_fused:
                TPBackendNative._fused_reduce_count += 1
                TPBackendNative._total_bytes_fused += tensor_bytes
            else:
                TPBackendNative._gpu_reduce_count += 1
                TPBackendNative._total_bytes_gpu += tensor_bytes

            # Log first few calls for debugging
            total_calls = TPBackendNative._gpu_reduce_count + TPBackendNative._fused_reduce_count
            if total_calls <= 5 or (is_fused and TPBackendNative._fused_reduce_count <= 3):
                fused_str = "FUSED " if is_fused else ""
                log_tp(self.device, f"All-reduce: {fused_str}GPU path ({tensor.numel()} elems, {tensor_bytes//1024} KB)")

            ext.pg_all_reduce(
                self.ptr_g,
                self.active_devices,
                self.device,
                self.active_devices[0],
                tensor,
                self.ptr_b,  # Use larger SHBUF (16 MB vs 2.1 MB)
                self.shbuf_size,
                self.abort_flag
            )
        else:
            # CPU-based all-reduce (original behavior)
            # Goes through RAM: GPU → RAM → CPU (sum) → RAM → GPU
            # Use this as fallback when P2P is not available or for small tensors

            # Track statistics
            TPBackendNative._cpu_reduce_count += 1
            TPBackendNative._total_bytes_cpu += tensor_bytes

            # Log first few calls for debugging
            if TPBackendNative._cpu_reduce_count <= 5:
                reason = "disabled" if not OptimizationFlags.ENABLE_GPU_ALL_REDUCE else "too small"
                log_tp(self.device, f"All-reduce: CPU path ({tensor.numel()} elems, {tensor_bytes//1024} KB, reason: {reason})")

            ext.pg_all_reduce_cpu(
                self.ptr_g,
                self.active_devices,
                self.device,
                self.active_devices[0],
                tensor,
                contribution,
                self.ptr_r,
                self.shbuf_size_r,  # Use instance variable (OPT3: dynamic buffer size)
                self.master,
                self.abort_flag
            )

    @classmethod
    def get_all_reduce_stats(cls):
        """Get statistics on all-reduce usage"""
        total = cls._gpu_reduce_count + cls._cpu_reduce_count + cls._fused_reduce_count
        if total == 0:
            return {"total": 0, "gpu": 0, "cpu": 0, "fused": 0, "gpu_pct": 0, "cpu_pct": 0, "fused_pct": 0}

        gpu_pct = 100 * cls._gpu_reduce_count / total
        cpu_pct = 100 * cls._cpu_reduce_count / total
        fused_pct = 100 * cls._fused_reduce_count / total

        return {
            "total": total,
            "gpu": cls._gpu_reduce_count,
            "cpu": cls._cpu_reduce_count,
            "fused": cls._fused_reduce_count,
            "gpu_pct": gpu_pct,
            "cpu_pct": cpu_pct,
            "fused_pct": fused_pct,
            "total_mb_gpu": cls._total_bytes_gpu / (1024*1024),
            "total_mb_cpu": cls._total_bytes_cpu / (1024*1024),
            "total_mb_fused": cls._total_bytes_fused / (1024*1024),
        }

    @classmethod
    def log_all_reduce_stats(cls):
        """Log all-reduce statistics"""
        stats = cls.get_all_reduce_stats()
        if stats["total"] > 0:
            log_tp(-1, f"All-reduce statistics:")
            log_tp(-1, f"  Total calls: {stats['total']}")
            log_tp(-1, f"  GPU path: {stats['gpu']} ({stats['gpu_pct']:.1f}%) - {stats['total_mb_gpu']:.1f} MB")
            log_tp(-1, f"  FUSED path: {stats['fused']} ({stats['fused_pct']:.1f}%) - {stats['total_mb_fused']:.1f} MB")
            log_tp(-1, f"  CPU path: {stats['cpu']} ({stats['cpu_pct']:.1f}%) - {stats['total_mb_cpu']:.1f} MB")

            # Calculate reduction efficiency
            if stats["fused"] > 0:
                # Without fusion: would be 2x calls (attn + MLP)
                unfused_calls = stats["gpu"] + stats["cpu"] + 2 * stats["fused"]
                actual_calls = stats["total"]
                efficiency = 100 * (1 - actual_calls / unfused_calls)
                log_tp(-1, f"  Fusion efficiency: {efficiency:.1f}% reduction in all-reduce calls")



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
            self.shbuf_size_r,  # Use instance variable (OPT3: dynamic buffer size)
        )


    def end_cpu_reduce_jobs(self):
        if self.master:
            ext.end_cpu_reduce_jobs(
                self.ptr_g,
            )