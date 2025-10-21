from __future__ import annotations
import torch
import numpy as np
from multiprocessing import shared_memory
import uuid
import threading
from collections import defaultdict
from .model_tp_cuda import cuda_host_register, cuda_host_unregister, CUDA_HOST_REGISTER_PORTABLE

DEFAULT_BUFFER_SIZE = 2 * 1024 ** 3
MIN_BUFFER_SIZE = 64 * 1024 ** 2  # 64MB minimum
MAX_BUFFER_SIZE = 8 * 1024 ** 3  # 8GB maximum
BUFFER_GROWTH_FACTOR = 1.5  # Buffer growth factor when resizing

_torch_dtypes = {
    "torch.uint8": torch.uint8,
    "torch.int8": torch.int8,
    "torch.int16": torch.int16,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.bfloat16": torch.bfloat16,
}

# Memory pool for efficient buffer management
class MemoryPool:
    """Thread-safe memory pool for shared memory operations."""
    
    def __init__(self, initial_size: int = 64 * 1024 * 1024):
        self.pools = defaultdict(list)  # size -> list of free buffers
        self.lock = threading.Lock()
        self.initial_size = initial_size
        self.stats = {
            'allocations': 0,
            'reuses': 0,
            'total_bytes': 0
        }
    
    def get_buffer(self, size: int) -> int:
        """Get a buffer of at least the requested size."""
        with self.lock:
            # Find the smallest pool that can accommodate the request
            for pool_size in sorted(self.pools.keys()):
                if pool_size >= size and self.pools[pool_size]:
                    buffer = self.pools[pool_size].pop()
                    self.stats['reuses'] += 1
                    return buffer
            
            # No suitable buffer found, allocate new one
            self.stats['allocations'] += 1
            self.stats['total_bytes'] += size
            return size
    
    def return_buffer(self, buffer_size: int, actual_size: int):
        """Return a buffer to the pool."""
        with self.lock:
            self.pools[buffer_size].append(actual_size)
    
    def get_stats(self) -> dict:
        """Get memory pool statistics."""
        with self.lock:
            return self.stats.copy()

# Global memory pool instance
_memory_pool = MemoryPool()

class SMProducer:
    def __init__(
        self,
        shm_name: str | None = None,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        adaptive_sizing: bool = True,
        min_buffer_size: int = MIN_BUFFER_SIZE,
        max_buffer_size: int = MAX_BUFFER_SIZE,
    ):
        self.shm_name = shm_name or uuid.uuid4().hex
        self.initial_buffer_size = max(min_buffer_size, min(buffer_size, max_buffer_size))
        self.buffer_size = self.initial_buffer_size
        self.adaptive_sizing = adaptive_sizing
        self.min_buffer_size = min_buffer_size
        self.max_buffer_size = max_buffer_size
        
        # Memory management
        self.allocations = []  # Track allocations for defragmentation
        self.free_blocks = []  # Free blocks for reuse
        self.peak_usage = 0
        self.resize_count = 0
        
        # Performance tracking
        self.stats = {
            'total_sent': 0,
            'total_bytes': 0,
            'buffer_resizes': 0,
            'fragmentations': 0,
            'zero_copy_ops': 0
        }

        # Create SHM handle and numpy buffer
        self.shm = shared_memory.SharedMemory(create = True, size = self.buffer_size, name = self.shm_name)
        self.buf = np.ndarray((self.buffer_size,), dtype = np.uint8, buffer = self.shm.buf)
        self.buf_is_pinned = False
        self.next_offset = 0

        # Pre-touch buffer to avoid page faults later
        self.buf[: self.buffer_size: 4096] = 0

    def export(self):
        return {
            "shm_name": self.shm_name,
            "buffer_size": self.buffer_size,
        }

    def _resize_buffer_if_needed(self, required_size: int) -> bool:
        """Resize buffer if needed for adaptive sizing."""
        if not self.adaptive_sizing:
            return False
            
        if required_size > self.buffer_size:
            # Calculate new buffer size
            new_size = min(
                int(required_size * BUFFER_GROWTH_FACTOR),
                self.max_buffer_size
            )
            
            if new_size > self.buffer_size:
                # Create new shared memory
                new_shm = shared_memory.SharedMemory(create=True, size=new_size, name=self.shm_name + "_resize")
                new_buf = np.ndarray((new_size,), dtype=np.uint8, buffer=new_shm.buf)
                
                # Copy existing data
                new_buf[:self.buffer_size] = self.buf
                
                # Update references
                self.shm.close()
                self.shm.unlink()
                self.shm = new_shm
                self.buf = new_buf
                self.buffer_size = new_size
                self.resize_count += 1
                self.stats['buffer_resizes'] += 1
                
                # Pre-touch new buffer
                self.buf[self.buffer_size:new_size:4096] = 0
                return True
        return False
    
    def _defragment_if_needed(self):
        """Defragment buffer if fragmentation is high."""
        if len(self.free_blocks) > 10 and self.next_offset > self.buffer_size * 0.8:
            # Compact allocations
            self.stats['fragmentations'] += 1
            self.allocations.sort(key=lambda x: x[0])  # Sort by offset
            
            # Rebuild buffer
            new_offset = 0
            for offset, size in self.allocations:
                if offset != new_offset:
                    # Move data
                    self.buf[new_offset:new_offset+size] = self.buf[offset:offset+size]
                new_offset += size
            
            self.next_offset = new_offset
            self.free_blocks = []
    
    def send(self, tensor: torch.Tensor | None) -> dict:
        """Send tensor with optimized memory management."""
        self.stats['total_sent'] += 1

        # None tensor
        if tensor is None:
            return {
                "method": "none_tensor",
            }

        # Bytes to export
        nbytes = tensor.element_size() * tensor.numel()
        nbytes_align = (nbytes + 127) // 128 * 128
        self.stats['total_bytes'] += nbytes

        # Check if we can use zero-copy for CUDA tensors
        if tensor.is_cuda and tensor.is_contiguous():
            try:
                # Try to share CUDA memory directly
                tensor.share_memory_()
                self.stats['zero_copy_ops'] += 1
                return {
                    "method": "share_memory",
                    "shared_tensor": tensor,
                }
            except:
                pass  # Fall back to regular buffer

        # Check if buffer needs resizing
        if self.next_offset + nbytes_align >= self.buffer_size:
            # Try defragmentation first
            self._defragment_if_needed()
            
            # Check again after defragmentation
            if self.next_offset + nbytes_align >= self.buffer_size:
                # Try to resize buffer
                if not self._resize_buffer_if_needed(self.next_offset + nbytes_align):
                    # Fall back to slow sharing if buffer can't be resized
                    tensor.share_memory_()
                    return {
                        "method": "share_memory",
                        "shared_tensor": tensor,
                    }

        # Allocate space
        offset = self.next_offset
        self.next_offset += nbytes_align
        self.allocations.append((offset, nbytes_align))
        
        # Update peak usage
        if self.next_offset > self.peak_usage:
            self.peak_usage = self.next_offset

        tensor_d = tensor.view((1,)) if len(tensor.shape) == 0 else tensor
        # Optimized copy to shared buffer
        if tensor_d.is_cuda:
            # For CUDA tensors, move to CPU efficiently
            t_cpu = tensor_d.cpu().contiguous()
        else:
            t_cpu = tensor_d.contiguous()
        
        # Use memoryview for faster copying
        src_mv = memoryview(t_cpu.numpy())
        dst_mv = memoryview(self.buf)[offset:offset+nbytes]
        dst_mv[:] = src_mv.cast('B')

        # Data is now buffered in shared memory space, store metadata and offset
        return {
            "method": "buffer",
            "offset": offset,
            "nbytes": nbytes,
            "dtype": str(tensor.dtype),
            "shape": tuple(tensor.shape),
        }

    def clear(self):
        """Clear buffer and reset allocations."""
        self.next_offset = 0
        self.allocations = []
        self.free_blocks = []

    def get_stats(self) -> dict:
        """Get performance statistics."""
        stats = self.stats.copy()
        stats.update({
            'buffer_size': self.buffer_size,
            'peak_usage': self.peak_usage,
            'peak_usage_percent': (self.peak_usage / self.buffer_size * 100) if self.buffer_size > 0 else 0,
            'resize_count': self.resize_count,
            'num_allocations': len(self.allocations),
            'num_free_blocks': len(self.free_blocks),
        })
        return stats

    def close(self):
        self.shm.close()
        try:
            self.shm.unlink()
        except:
            pass  # Already unlinked


class SMConsumer:

    def __init__(
        self,
        producer_imp: dict | SMProducer,
        device: int | None = None,
        pin_memory: bool = False,
        enable_zero_copy: bool = True,
        cache_tensors: bool = True,
    ):
        self.pin_memory = pin_memory
        self.device = device
        self.enable_zero_copy = enable_zero_copy
        self.cache_tensors = cache_tensors
        
        # Performance tracking
        self.stats = {
            'total_recv': 0,
            'total_bytes': 0,
            'zero_copy_hits': 0,
            'cache_hits': 0,
            'gpu_transfers': 0,
        }
        
        # Tensor cache for frequently accessed tensors
        self.tensor_cache = {} if cache_tensors else None
        self.cache_max_size = 100
        
        if device is not None:
            torch.cuda.set_device(self.device)

        def get_local_tensor(shm_buf, _buffer_size):
            # Create local uint8 tensor mapping the entire shared buffer
            np_view = np.ndarray(
                shape = (_buffer_size,),
                dtype = np.uint8,
                buffer = shm_buf,
                offset = 0,
            )
            return torch.as_tensor(np_view)

        # Remote process consumer
        if isinstance(producer_imp, dict):
            self.producer = None
            self.shm_name = producer_imp["shm_name"]
            self.buffer_size = producer_imp["buffer_size"]
            self.shm = shared_memory.SharedMemory(name = self.shm_name)
            self.arena = get_local_tensor(self.shm.buf, self.buffer_size)

            # Optionally pin memory
            if pin_memory and device is not None:
                torch.cuda.set_device(self.device)
                cuda_host_register(self.arena.data_ptr(), self.arena.numel(), flags = CUDA_HOST_REGISTER_PORTABLE)

        # Local consumer
        else:
            assert isinstance(producer_imp, SMProducer)
            self.producer = producer_imp
            self.shm_name = producer_imp.shm_name
            self.buffer_size = producer_imp.buffer_size
            self.shm = producer_imp.shm
            self.arena = get_local_tensor(self.shm.buf, self.buffer_size)

            # Optionally pin memory
            if pin_memory and device is not None:
                torch.cuda.set_device(self.device)
                assert not self.producer.buf_is_pinned, "Only one local consumer can pin arena"
                cuda_host_register(self.arena.data_ptr(), self.arena.numel(), flags = CUDA_HOST_REGISTER_PORTABLE)
                self.producer.buf_is_pinned = True

    def _get_cache_key(self, imp: dict) -> str:
        """Generate cache key for tensor metadata."""
        if imp["method"] == "buffer":
            return f"{imp['offset']}_{imp['nbytes']}_{imp['dtype']}_{imp['shape']}"
        return None
    
    def recv(
        self,
        imp: dict,
        cuda: bool = False,
        slice_dim: int | None = None,
        first: int | None = None,
        last: int | None = None,
    ) -> torch.Tensor | None:
        """Receive tensor with optimized memory management."""
        self.stats['total_recv'] += 1

        if cuda and self.device is not None:
            torch.cuda.set_device(self.device)

        # Send was None
        if imp["method"] == "none_tensor":
            return None

        # Check cache first
        cache_key = self._get_cache_key(imp)
        if self.tensor_cache and cache_key and cache_key in self.tensor_cache:
            self.stats['cache_hits'] += 1
            tensor = self.tensor_cache[cache_key].clone()
        else:
            # Fallback method
            if imp["method"] == "share_memory":
                tensor = imp["shared_tensor"]
                if self.enable_zero_copy and tensor.is_cuda and cuda and tensor.device.index == self.device:
                    self.stats['zero_copy_hits'] += 1
                    # Zero-copy path - tensor is already on the right device
                    pass
                else:
                    # Need to copy/move tensor
                    if cuda and self.device is not None:
                        tensor = tensor.to(self.device, non_blocking=self.pin_memory)
                        self.stats['gpu_transfers'] += 1

            # Construct Torch tensor in shared memory
            else:
                offset = imp["offset"]
                nbytes = imp["nbytes"]
                dtype = _torch_dtypes[imp["dtype"]]
                shape = imp["shape"]
                
                # Optimized tensor creation
                tensor_view = self.arena.narrow(0, offset, nbytes).view(dtype)
                tensor = tensor_view.view(shape)
                
                self.stats['total_bytes'] += nbytes
                
                # Cache the tensor if enabled
                if self.tensor_cache and cache_key:
                    if len(self.tensor_cache) < self.cache_max_size:
                        self.tensor_cache[cache_key] = tensor.clone()

        # Slice before cloning (more efficient)
        if slice_dim is not None:
            tensor = tensor.narrow(slice_dim, first, last - first)

        # Move to GPU or clone to unshared memory
        if cuda and self.device is not None and not tensor.is_cuda:
            tensor = tensor.to(
                self.device,
                non_blocking = self.pin_memory,
                copy = True,
                memory_format = torch.contiguous_format
            )
            self.stats['gpu_transfers'] += 1
        elif not cuda or tensor.is_cuda:
            # Only clone if necessary
            if not tensor.is_contiguous():
                tensor = tensor.clone(memory_format = torch.contiguous_format)

        return tensor
    
    def recv_batch(self, batch_imp: list[dict], cuda: bool = False) -> list[torch.Tensor | None]:
        """Receive multiple tensors efficiently in batch."""
        results = []
        for imp in batch_imp:
            results.append(self.recv(imp, cuda))
        return results
    
    def clear_cache(self):
        """Clear tensor cache."""
        if self.tensor_cache:
            self.tensor_cache.clear()
    
    def get_stats(self) -> dict:
        """Get performance statistics."""
        stats = self.stats.copy()
        if self.tensor_cache:
            stats['cache_size'] = len(self.tensor_cache)
            stats['cache_hit_rate'] = (stats['cache_hits'] / stats['total_recv'] * 100) if stats['total_recv'] > 0 else 0
        stats['zero_copy_rate'] = (stats['zero_copy_hits'] / stats['total_recv'] * 100) if stats['total_recv'] > 0 else 0
        return stats


    def close(self):
        if self.pin_memory and self.device is not None:
            try:
                cuda_host_unregister(self.arena.data_ptr())
            except:
                pass  # Already unregistered
        
        if self.producer is not None:
            self.shm.close()
        else:
            try:
                self.shm.close()
            except:
                pass  # Already closed
        
        # Clear cache
        if self.tensor_cache:
            self.tensor_cache.clear()
