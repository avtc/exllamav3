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
        RuntimeError: If CUDA API calls fail during connectivity checks
        
    Example:
        >>> devices = [0, 1, 2]
        >>> if check_p2p_connectivity(devices):
        ...     print("P2P backend can be used")
        ... else:
        ...     print("Using fallback backend")
        
    Note:
        - For single device or empty lists, returns True (no P2P needed)
        - Checks unidirectional connectivity between all GPU pairs
        - May require elevated privileges on some systems
    """
    lib = _cudart()
    
    # Get device count even for single device to validate the device list
    try:
        cuda_get_device_count = lib.cudaGetDeviceCount
        cuda_get_device_count.argtypes = []
        cuda_get_device_count.restype = ctypes.c_int
        device_count = cuda_get_device_count()
        
        if device_count < len(active_devices):
            return False
            
    except Exception as e:
        raise RuntimeError(f"Failed to get device count: {e}")
    
    # Check if we have at least 2 devices for P2P to make sense
    if len(active_devices) < 2:
        return True
    
    # Get number of available devices first
    try:
        cuda_get_device_count = lib.cudaGetDeviceCount
        cuda_get_device_count.argtypes = []
        cuda_get_device_count.restype = ctypes.c_int
        device_count = cuda_get_device_count()
        
        if device_count < len(active_devices):
            return False
            
    except Exception as e:
        raise RuntimeError(f"Failed to get device count: {e}")
    
    # Check P2P connectivity between all pairs (unidirectional only)
    for i, device_a in enumerate(active_devices):
        for j, device_b in enumerate(active_devices):
            if i == j:
                continue
                
            # Check P2P access in one direction only
            try:
                # Check device_a -> device_b
                cuda_set_device = lib.cudaSetDevice
                cuda_set_device.argtypes = [ctypes.c_int]
                cuda_set_device.restype = ctypes.c_int
                err = cuda_set_device(device_a)
                if err != CUDA_SUCCESS:
                    raise RuntimeError(f"Failed to set device {device_a}: {_cuda_error_string(err)}")
                
                cuda_can_access_peer = lib.cudaDeviceCanAccessPeer
                cuda_can_access_peer.argtypes = [ctypes.c_int, ctypes.c_int]
                cuda_can_access_peer.restype = ctypes.c_int
                can_access = cuda_can_access_peer(device_a, device_b)
                
                if can_access != 1:
                    return False
                    
            except Exception as e:
                raise RuntimeError(f"Failed to check P2P access from device {device_a} to {device_b}: {e}")
    
    return True


def enable_p2p_access(active_devices: list[int]) -> None:
    """
    Enable P2P access between all GPUs in the system.
    
    This function enables peer-to-peer memory access between all pairs of GPUs
    in the provided list. This is required before using P2P communication
    primitives.
    
    Args:
        active_devices: List of GPU device IDs to enable P2P access for
        
    Raises:
        RuntimeError: If CUDA API calls fail or if P2P access cannot be enabled
        
    Example:
        >>> devices = [0, 1, 2]
        >>> enable_p2p_access(devices)
        >>> # Now P2P communication can be used between these devices
        
    Note:
        - Must be called after verifying P2P connectivity with check_p2p_connectivity()
        - Enables bidirectional access between all GPU pairs
        - May require system-level configuration changes on some platforms
        - Safe to call multiple times (idempotent with respect to already enabled access)
    """
    lib = _cudart()
    
    # Enable P2P access between all pairs
    for i, device_a in enumerate(active_devices):
        for j, device_b in enumerate(active_devices):
            if i == j:
                continue
                
            try:
                # Set current device for API calls
                cuda_set_device = lib.cudaSetDevice
                cuda_set_device.argtypes = [ctypes.c_int]
                cuda_set_device.restype = ctypes.c_int
                
                # Enable P2P access from device_a to device_b
                err = cuda_set_device(device_a)
                if err != CUDA_SUCCESS:
                    raise RuntimeError(f"Failed to set device {device_a}: {_cuda_error_string(err)}")
                
                cuda_enable_peer_access = lib.cudaDeviceEnablePeerAccess
                cuda_enable_peer_access.argtypes = [ctypes.c_int, ctypes.c_uint]
                cuda_enable_peer_access.restype = ctypes.c_int
                err = cuda_enable_peer_access(device_b, 0)
                
                # CUDA_ERROR_P2P_UNACCESSIBLE is expected if already enabled
                if err not in (CUDA_SUCCESS, CUDA_ERROR_P2P_UNACCESSIBLE):
                    raise RuntimeError(f"Failed to enable P2P access from device {device_a} to {device_b}: {err} ({_cuda_error_string(err)})")
                    
            except Exception as e:
                raise RuntimeError(f"Failed to enable P2P access from device {device_a} to {device_b}: {e}")


def disable_p2p_access(active_devices: list[int]) -> None:
    """
    Disable P2P access between all GPUs in the system.
    
    This function disables peer-to-peer memory access between all pairs of GPUs
    in the provided list. This should be called when cleaning up P2P resources.
    
    Args:
        active_devices: List of GPU device IDs to disable P2P access for
        
    Raises:
        RuntimeError: If CUDA API calls fail
        
    Example:
        >>> devices = [0, 1, 2]
        >>> disable_p2p_access(devices)
        >>> # P2P access is now disabled between these devices
        
    Note:
        - Typically called during cleanup or when switching backends
        - Safe to call even if P2P access was not previously enabled
        - Disables bidirectional access between all GPU pairs
        - May require system restart to fully take effect in some cases
    """
    lib = _cudart()
    
    # Disable P2P access between all pairs
    for i, device_a in enumerate(active_devices):
        for j, device_b in enumerate(active_devices):
            if i == j:
                continue
                
            try:
                # Set current device for API calls
                cuda_set_device = lib.cudaSetDevice
                cuda_set_device.argtypes = [ctypes.c_int]
                cuda_set_device.restype = ctypes.c_int
                
                # Disable P2P access from device_a to device_b
                err = cuda_set_device(device_a)
                if err != CUDA_SUCCESS:
                    raise RuntimeError(f"Failed to set device {device_a}: {_cuda_error_string(err)}")
                
                cuda_disable_peer_access = lib.cudaDeviceDisablePeerAccess
                cuda_disable_peer_access.argtypes = [ctypes.c_int]
                cuda_disable_peer_access.restype = ctypes.c_int
                err = cuda_disable_peer_access(device_b)
                
                # CUDA_ERROR_P2P_UNACCESSIBLE is expected if already disabled
                if err not in (CUDA_SUCCESS, CUDA_ERROR_P2P_UNACCESSIBLE):
                    raise RuntimeError(f"Failed to disable P2P access from device {device_a} to {device_b}: {err} ({_cuda_error_string(err)})")
                    
            except Exception as e:
                raise RuntimeError(f"Failed to disable P2P access from device {device_a} to {device_b}: {e}")
