import ctypes
from functools import lru_cache
import os, glob

CUDA_SUCCESS = 0
CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 34
CUDA_HOST_REGISTER_PORTABLE = 1
CUDA_ERROR_CUDART_UNLOADING = 13
CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 719

# P2P memory constants
CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 217
CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 218
CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 219
CUDA_ERROR_INVALID_DEVICE = 10

@lru_cache(maxsize = 1)
def _cudart():

    # Windows: Try to find cudart64_*.dll in common paths
    # TODO: Test that this actually works
    if os.name == "nt":
        candidates = [f"cudart64_{v}.dll" for v in ("130","120","118","117","116","110","101","100")]
        for p in os.getenv("PATH","").split(os.pathsep):
            candidates += glob.glob(os.path.join(p, "cudart64_*.dll"))
        last_err = None
        for name in candidates:
            try:
                return ctypes.WinDLL(name)  # __stdcall
            except OSError as e:
                last_err = e
        raise OSError("Could not load cudart64_*.dll; ensure CUDA runtime is on PATH") from last_err

    # Linux: try unversioned and common SONAMEs, then ctypes.util.find_library
    else:
        for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11"):
            try:
                return ctypes.CDLL(name)  # cdecl
            except OSError:
                pass
        from ctypes.util import find_library
        path = find_library("cudart")
        if path:
            return ctypes.CDLL(path)
        raise OSError("Could not load libcudart; set LD_LIBRARY_PATH to your CUDA runtime")


def _cuda_error_string(code: int) -> str:
    lib = _cudart()
    fn = lib.cudaGetErrorString
    fn.argtypes = [ctypes.c_int]
    fn.restype  = ctypes.c_char_p
    return fn(code).decode(errors="replace")


def cuda_host_register(ptr: int, nbytes: int, flags: int = 0) -> None:
    lib = _cudart()
    fn = lib.cudaHostRegister
    fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
    fn.restype  = ctypes.c_int
    err = fn(ctypes.c_void_p(ptr), ctypes.c_size_t(nbytes), ctypes.c_uint(flags))
    if err not in (CUDA_SUCCESS, CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED):
        raise RuntimeError(f"cudaHostRegister({hex(ptr)}, {nbytes}) failed: {err} ({_cuda_error_string(err)})")


def cuda_host_unregister(ptr: int) -> None:
    lib = _cudart()
    fn = lib.cudaHostUnregister
    fn.argtypes = [ctypes.c_void_p]
    fn.restype  = ctypes.c_int
    err = fn(ctypes.c_void_p(ptr))
    # During teardown (or if already unregistered) treat as benign:
    if err not in (CUDA_SUCCESS, CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED, CUDA_ERROR_CUDART_UNLOADING):
        raise RuntimeError(f"cudaHostUnregister({hex(ptr)}) failed: {err} ({_cuda_error_string(err)})")


# P2P memory utility functions
def cuda_device_enable_peer_access(peer_device: int, flags: int = 0) -> None:
    """Enable peer access to another GPU device."""
    # First check if peer access is already enabled
    current_device = cuda_get_device()
    can_access = ctypes.c_int()
    try:
        cuda_device_can_access_peer(ctypes.byref(can_access), current_device, peer_device)
        if can_access.value != 0:
            # P2P access is already available, no need to enable it again
            return
    except RuntimeError:
        pass  # Continue with enabling if check fails
    
    lib = _cudart()
    fn = lib.cudaDeviceEnablePeerAccess
    fn.argtypes = [ctypes.c_int, ctypes.c_uint]
    fn.restype = ctypes.c_int
    err = fn(ctypes.c_int(peer_device), ctypes.c_uint(flags))
    if err not in (CUDA_SUCCESS, CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED):
        raise RuntimeError(f"cudaDeviceEnablePeerAccess({peer_device}) failed: {err} ({_cuda_error_string(err)})")


def cuda_device_disable_peer_access(peer_device: int) -> None:
    """Disable peer access to another GPU device."""
    lib = _cudart()
    fn = lib.cudaDeviceDisablePeerAccess
    fn.argtypes = [ctypes.c_int]
    fn.restype = ctypes.c_int
    err = fn(ctypes.c_int(peer_device))
    if err not in (CUDA_SUCCESS, CUDA_ERROR_PEER_ACCESS_NOT_ENABLED):
        raise RuntimeError(f"cudaDeviceDisablePeerAccess({peer_device}) failed: {err} ({_cuda_error_string(err)})")


def cuda_device_can_access_peer(can_access_ptr: int, device: int, peer_device: int) -> int:
    """Check if device can access peer device memory."""
    lib = _cudart()
    fn = lib.cudaDeviceCanAccessPeer
    fn.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    fn.restype = ctypes.c_int
    err = fn(ctypes.POINTER(ctypes.c_int)(can_access_ptr), ctypes.c_int(device), ctypes.c_int(peer_device))
    if err != CUDA_SUCCESS:
        raise RuntimeError(f"cudaDeviceCanAccessPeer({device}, {peer_device}) failed: {err} ({_cuda_error_string(err)})")
    return err


def cuda_memcpy_peer_async(dst_ptr: int, dst_device: int, src_ptr: int, src_device: int, count: int, stream: int = 0) -> None:
    """Copy memory from one GPU device to another asynchronously."""
    lib = _cudart()
    fn = lib.cudaMemcpyPeerAsync
    fn.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    err = fn(ctypes.c_void_p(dst_ptr), ctypes.c_int(dst_device), ctypes.c_void_p(src_ptr), ctypes.c_int(src_device), ctypes.c_size_t(count), ctypes.c_void_p(stream))
    if err != CUDA_SUCCESS:
        raise RuntimeError(f"cudaMemcpyPeerAsync failed: {err} ({_cuda_error_string(err)})")


def cuda_memcpy_peer(dst_ptr: int, dst_device: int, src_ptr: int, src_device: int, count: int) -> None:
    """Copy memory from one GPU device to another synchronously."""
    lib = _cudart()
    fn = lib.cudaMemcpyPeer
    fn.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    fn.restype = ctypes.c_int
    err = fn(ctypes.c_void_p(dst_ptr), ctypes.c_int(dst_device), ctypes.c_void_p(src_ptr), ctypes.c_int(src_device), ctypes.c_size_t(count))
    if err != CUDA_SUCCESS:
        raise RuntimeError(f"cudaMemcpyPeer failed: {err} ({_cuda_error_string(err)})")


def cuda_memcpy_2d_peer_async(dst_ptr: int, dst_pitch: int, src_ptr: int, src_pitch: int, width: int, height: int, dst_device: int, src_device: int, stream: int = 0) -> None:
    """Copy 2D memory from one GPU device to another asynchronously."""
    lib = _cudart()
    fn = lib.cudaMemcpy2DPeerAsync
    fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    err = fn(ctypes.c_void_p(dst_ptr), ctypes.c_size_t(dst_pitch), ctypes.c_void_p(src_ptr), ctypes.c_size_t(src_pitch), ctypes.c_size_t(width), ctypes.c_size_t(height), ctypes.c_int(dst_device), ctypes.c_int(src_device), ctypes.c_void_p(stream))
    if err != CUDA_SUCCESS:
        raise RuntimeError(f"cudaMemcpy2DPeerAsync failed: {err} ({_cuda_error_string(err)})")


def cuda_get_device_count() -> int:
    """Get the number of CUDA devices."""
    lib = _cudart()
    fn = lib.cudaGetDeviceCount
    fn.argtypes = [ctypes.POINTER(ctypes.c_int)]
    fn.restype = ctypes.c_int
    count = ctypes.c_int()
    err = fn(ctypes.byref(count))
    if err != CUDA_SUCCESS:
        raise RuntimeError(f"cudaGetDeviceCount failed: {err} ({_cuda_error_string(err)})")
    return count.value


def cuda_set_device(device: int) -> None:
    """Set the current CUDA device."""
    lib = _cudart()
    fn = lib.cudaSetDevice
    fn.argtypes = [ctypes.c_int]
    fn.restype = ctypes.c_int
    err = fn(ctypes.c_int(device))
    if err != CUDA_SUCCESS:
        raise RuntimeError(f"cudaSetDevice({device}) failed: {err} ({_cuda_error_string(err)})")


def cuda_get_device() -> int:
    """Get the current CUDA device."""
    lib = _cudart()
    fn = lib.cudaGetDevice
    fn.argtypes = [ctypes.POINTER(ctypes.c_int)]
    fn.restype = ctypes.c_int
    device = ctypes.c_int()
    err = fn(ctypes.byref(device))
    if err != CUDA_SUCCESS:
        raise RuntimeError(f"cudaGetDevice failed: {err} ({_cuda_error_string(err)})")
    return device.value


def cuda_device_synchronize() -> None:
    """Synchronize the current CUDA device."""
    lib = _cudart()
    fn = lib.cudaDeviceSynchronize
    fn.argtypes = []
    fn.restype = ctypes.c_int
    err = fn()
    if err != CUDA_SUCCESS:
        raise RuntimeError(f"cudaDeviceSynchronize failed: {err} ({_cuda_error_string(err)})")


def cuda_device_reset() -> None:
    """Reset the current CUDA device."""
    lib = _cudart()
    fn = lib.cudaDeviceReset
    fn.argtypes = []
    fn.restype = ctypes.c_int
    err = fn()
    if err != CUDA_SUCCESS:
        raise RuntimeError(f"cudaDeviceReset failed: {err} ({_cuda_error_string(err)})")


# Optimized memory management utilities
class OptimizedMemoryManager:
    """Optimized memory management for CPU/GPU operations."""
    
    def __init__(self):
        self.pinned_buffers = {}  # size -> buffer pool
        self.max_pinned_memory = 1024 * 1024 * 1024  # 1GB max pinned memory
        self.current_pinned = 0
        self.buffer_alignment = 256  # 256-byte alignment for optimal performance
    
    def get_pinned_buffer(self, size: int) -> int:
        """Get or create a pinned memory buffer."""
        aligned_size = ((size + self.buffer_alignment - 1) // self.buffer_alignment) * self.buffer_alignment
        
        # Check if we have a buffer of this size
        if aligned_size in self.pinned_buffers and self.pinned_buffers[aligned_size]:
            return self.pinned_buffers[aligned_size].pop()
        
        # Check if we have enough memory for a new buffer
        if self.current_pinned + aligned_size > self.max_pinned_memory:
            # Try to free some buffers
            self._cleanup_buffers()
            if self.current_pinned + aligned_size > self.max_pinned_memory:
                raise RuntimeError("Not enough pinned memory available")
        
        # Allocate new pinned buffer
        try:
            import torch
            buffer = torch.empty(aligned_size, dtype=torch.uint8, pin_memory=True)
            self.current_pinned += aligned_size
            
            # Initialize buffer pool for this size
            if aligned_size not in self.pinned_buffers:
                self.pinned_buffers[aligned_size] = []
            
            return buffer.data_ptr()
        except Exception as e:
            raise RuntimeError(f"Failed to allocate pinned buffer: {e}")
    
    def return_pinned_buffer(self, ptr: int, size: int):
        """Return a pinned buffer to the pool."""
        aligned_size = ((size + self.buffer_alignment - 1) // self.buffer_alignment) * self.buffer_alignment
        if aligned_size in self.pinned_buffers:
            self.pinned_buffers[aligned_size].append(ptr)
    
    def _cleanup_buffers(self):
        """Clean up buffer pools to free memory."""
        # Keep only the most recently used buffers
        for size in list(self.pinned_buffers.keys()):
            if len(self.pinned_buffers[size]) > 2:
                # Remove excess buffers
                excess = len(self.pinned_buffers[size]) - 2
                for _ in range(excess):
                    self.pinned_buffers[size].pop()
                    self.current_pinned -= size
    
    def get_stats(self) -> dict:
        """Get memory manager statistics."""
        return {
            'current_pinned_mb': self.current_pinned // (1024 * 1024),
            'max_pinned_mb': self.max_pinned_memory // (1024 * 1024),
            'buffer_pools': {size: len(buffers) for size, buffers in self.pinned_buffers.items()},
            'total_buffers': sum(len(buffers) for buffers in self.pinned_buffers.values())
        }

# Global memory manager instance
_memory_manager = OptimizedMemoryManager()

# P2P memory utility class with optimizations
class P2PMemoryUtils:
    """Utility class for P2P memory operations with optimizations."""
    
    @staticmethod
    def enable_peer_access(device: int, peer_device: int, flags: int = 0) -> bool:
        """Enable peer access between devices."""
        try:
            cuda_set_device(device)
            cuda_device_enable_peer_access(peer_device, flags)
            return True
        except RuntimeError:
            return False
    
    @staticmethod
    def disable_peer_access(device: int, peer_device: int) -> bool:
        """Disable peer access between devices."""
        try:
            cuda_set_device(device)
            cuda_device_disable_peer_access(peer_device)
            return True
        except RuntimeError:
            return False
    
    @staticmethod
    def can_access_peer(device: int, peer_device: int) -> bool:
        """Check if device can access peer device."""
        try:
            can_access = ctypes.c_int()
            cuda_device_can_access_peer(ctypes.byref(can_access), device, peer_device)
            return can_access.value != 0
        except RuntimeError:
            return False
    
    @staticmethod
    def copy_tensor_async(src_device: int, dst_device: int, src_ptr: int, dst_ptr: int, size: int, stream: int = 0) -> bool:
        """Copy tensor data between devices asynchronously."""
        try:
            cuda_memcpy_peer_async(dst_ptr, dst_device, src_ptr, src_device, size, stream)
            return True
        except RuntimeError:
            return False
    
    @staticmethod
    def copy_tensor_sync(src_device: int, dst_device: int, src_ptr: int, dst_ptr: int, size: int) -> bool:
        """Copy tensor data between devices synchronously."""
        try:
            cuda_memcpy_peer(dst_ptr, dst_device, src_ptr, src_device, size)
            return True
        except RuntimeError:
            return False
    
    @staticmethod
    def copy_tensor_2d_async(src_device: int, dst_device: int, src_ptr: int, dst_ptr: int,
                           src_pitch: int, dst_pitch: int, width: int, height: int, stream: int = 0) -> bool:
        """Copy 2D tensor data between devices asynchronously."""
        try:
            cuda_memcpy_2d_peer_async(dst_ptr, dst_pitch, src_ptr, src_pitch, width, height, dst_device, src_device, stream)
            return True
        except RuntimeError:
            return False
    
    @staticmethod
    def copy_tensor_optimized(src_device: int, dst_device: int, src_ptr: int, dst_ptr: int, size: int) -> bool:
        """Copy tensor data between devices with optimal method selection."""
        # For small tensors, use synchronous copy
        if size < 64 * 1024:  # < 64KB
            return P2PMemoryUtils.copy_tensor_sync(src_device, dst_device, src_ptr, dst_ptr, size)
        else:
            # For larger tensors, use asynchronous copy
            return P2PMemoryUtils.copy_tensor_async(src_device, dst_device, src_ptr, dst_ptr, size)
    
    @staticmethod
    def get_device_count() -> int:
        """Get the number of available CUDA devices."""
        try:
            return cuda_get_device_count()
        except RuntimeError:
            return 0
    
    @staticmethod
    def synchronize_device(device: int) -> bool:
        """Synchronize a specific device."""
        try:
            current_device = cuda_get_device()
            cuda_set_device(device)
            cuda_device_synchronize()
            cuda_set_device(current_device)
            return True
        except RuntimeError:
            return False
    
    @staticmethod
    def reset_device(device: int) -> bool:
        """Reset a specific device."""
        try:
            current_device = cuda_get_device()
            cuda_set_device(device)
            cuda_device_reset()
            cuda_set_device(current_device)
            return True
        except RuntimeError:
            return False
    
    @staticmethod
    def get_device_memory_info(device: int) -> dict:
        """Get detailed memory information for a device."""
        try:
            import torch
            cuda_set_device(device)
            props = torch.cuda.get_device_properties(device)
            total_memory = props.total_memory
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            free = total_memory - allocated
            
            return {
                'device': device,
                'total_memory': total_memory,
                'allocated_memory': allocated,
                'reserved_memory': reserved,
                'free_memory': free,
                'utilization_percent': (allocated / total_memory * 100) if total_memory > 0 else 0
            }
        except Exception as e:
            return {'device': device, 'error': str(e)}
