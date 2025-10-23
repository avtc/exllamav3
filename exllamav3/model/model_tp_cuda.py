import ctypes
from functools import lru_cache
import os, glob

CUDA_SUCCESS = 0
CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 34
CUDA_HOST_REGISTER_PORTABLE = 1
CUDA_ERROR_CUDART_UNLOADING = 13
CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 719

# P2P connectivity error codes
CUDA_ERROR_P2P_UNACCESSIBLE = 701

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


def check_p2p_connectivity(active_devices: list[int]) -> bool:
    """
    Check if all GPUs in the system are fully peer connected.
    
    This function verifies that every GPU in the provided list can establish
    peer-to-peer communication with every other GPU in at least one direction.
    This is a prerequisite for using the P2P backend.
    
    Args:
        active_devices: List of GPU device IDs to check for P2P connectivity
        
    Returns:
        bool: True if all GPUs are fully P2P connected, False otherwise.
              Returns True for single device or empty list.
        
    Raises:
        RuntimeError: If PyTorch CUDA operations fail during connectivity checks
        
    Example:
        >>> devices = [0, 1, 2]
        >>> if check_p2p_connectivity(devices):
        ...     print("P2P backend can be used")
        ... else:
        ...     print("Using fallback backend")
        
    Note:
        - For single device or empty lists, returns True (no P2P needed)
        - Uses PyTorch's built-in P2P checking for better integration
        - PyTorch automatically manages P2P access when needed
    """
    import torch
    
    # Check if we have at least 2 devices for P2P to make sense
    if len(active_devices) < 2:
        return True
    
    # Get device count even for single device to validate the device list
    try:
        device_count = torch.cuda.device_count()
        if device_count < len(active_devices):
            return False
    except Exception as e:
        raise RuntimeError(f"Failed to get device count: {e}")
    
    # Check P2P connectivity between all pairs using PyTorch
    for i, device_a in enumerate(active_devices):
        for j, device_b in enumerate(active_devices):
            if i == j:
                continue
                
            # Check P2P access in one direction only using PyTorch
            try:
                # Use PyTorch's built-in P2P checking
                can_access = torch.cuda.can_device_access_peer(device_a, device_b)
                
                if not can_access:
                    return False
                    
            except Exception as e:
                raise RuntimeError(f"Failed to check P2P access from device {device_a} to {device_b}: {e}")
    
    return True

