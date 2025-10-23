# P2P Backend Troubleshooting Guide

This guide provides solutions for common issues encountered when using the P2P backend in ExLlamaV3. Follow these steps to diagnose and resolve problems.

## Table of Contents

- [Quick Start Checklist](#quick-start-checklist)
- [Common Issues](#common-issues)
  - [P2P Connectivity Problems](#p2p-connectivity-problems)
  - [Performance Issues](#performance-issues)
  - [Memory Issues](#memory-issues)
  - [Backend Selection Problems](#backend-selection-problems)
- [Debugging Techniques](#debugging-techniques)
- [Hardware Compatibility](#hardware-compatibility)
- [Environment Configuration](#environment-configuration)
- [Advanced Troubleshooting](#advanced-troubleshooting)

## Quick Start Checklist

Before diving into detailed troubleshooting, check these common issues:

1. **CUDA Version**: Ensure you have CUDA 11.0 or later
2. **GPU Drivers**: Update to latest NVIDIA drivers
3. **P2P Connectivity**: Verify with `check_p2p_connectivity()`
4. **Environment Variables**: Set `EXLLAMA_P2P_VERBOSE=1` for debugging
5. **Permissions**: Ensure proper GPU access permissions
6. **Memory**: Check available GPU memory (at least 16GB recommended)

## Common Issues

### P2P Connectivity Problems

#### Issue: "P2P connectivity not available between all GPUs"

**Symptoms:**
- P2P backend fails to initialize
- Automatic backend selection falls back to NCCL
- Error messages about peer access

**Causes:**
- Missing P2P support between GPU pairs
- Insufficient GPU permissions
- Hardware limitations
- Driver issues

**Solutions:**

1. **Check P2P support manually:**
   ```python
   from exllamav3.model.model_tp_cuda import check_p2p_connectivity
   devices = [0, 1, 2]  # Your GPU IDs
   print("P2P available:", check_p2p_connectivity(devices))
   ```

2. **Verify GPU permissions:**
   ```bash
   nvidia-smi -i 0 -q | grep -i "access"
   nvidia-smi -i 1 -q | grep -i "access"
   ```

3. **Check GPU architecture compatibility:**
   ```bash
   nvidia-smi --query-gpu=name,memory.total,architecture --format=csv
   ```
   - Ensure all GPUs support P2P (most modern GPUs do)
   - Mixed architectures may have limited P2P support

4. **Note on P2P access:** PyTorch automatically manages P2P access between GPUs. Manual enabling is no longer required.

#### Issue: "Failed to enable P2P access"

**Symptoms:**
- P2P access fails during initialization
- Permission denied errors
- CUDA API errors

**Solutions:**

1. **Run with elevated privileges:**
   ```bash
   sudo python your_script.py  # Linux/macOS
   # or
   python your_script.py  # Windows (run as Administrator)
   ```

2. **Check system configuration:**
   ```bash
   # Linux: Check IOMMU groups
   grep -i "iommu" /proc/cmdline
   # Check kernel parameters
   cat /proc/cmdline | grep -i "vfio"
   ```

3. **Verify GPU topology:**
   ```bash
   nvidia-smi topo -m
   ```
   Look for "P2P" and "NVLink" connections between GPUs

### Performance Issues

#### Issue: P2P backend slower than expected

**Symptoms:**
- Throughput lower than NCCL backend
- High latency in communication operations
- Inconsistent performance

**Causes:**
- Suboptimal buffer sizes
- Memory bandwidth limitations
- Contention with other processes
- Incorrect configuration

**Solutions:**

1. **Adjust buffer sizes:**
   ```python
   # Set larger buffer size
   os.environ["EXLLAMA_P2P_BUFFER_SIZE"] = "67108864"  # 64MB
   
   # Test different sizes
   buffer_sizes = ["16777216", "33554432", "67108864", "134217728"]
   for size in buffer_sizes:
       os.environ["EXLLAMA_P2P_BUFFER_SIZE"] = size
       # Run benchmark
   ```

2. **Optimize memory usage:**
   ```python
   # Enable memory pinning for better performance
   model = Model.load(
       model_dir,
       tp_backend="p2p",
       devices=[0, 1],
       max_batch_size=1,
       max_seq_len=2048
   )
   ```

3. **Monitor system resources:**
   ```bash
   # Monitor GPU usage during inference
   nvidia-smi dmon -i 0,1 -s u -c 10
   
   # Monitor CPU and memory
   top -p $(pgrep -f your_script)
   ```

4. **Check for resource contention:**
   ```bash
   # Check GPU processes
   nvidia-smi -l 1
   
   # Check system load
   uptime
   ```

#### Issue: Inconsistent performance between runs

**Symptoms:**
- Performance varies significantly between identical runs
- High standard deviation in benchmark results
- Sudden performance drops

**Causes:**
- Caching effects
- Background processes
- Thermal throttling
- Memory fragmentation

**Solutions:**

1. **Warm up the system:**
   ```python
   def warm_up_cuda(devices, steps=10):
       for device in devices:
           torch.cuda.set_device(device)
           for _ in range(steps):
               x = torch.randn(1024, 1024, device=device, dtype=torch.float16)
               y = x @ x
               del x, y
               torch.cuda.synchronize()
   
   # Call before benchmarking
   warm_up_cuda([0, 1])
   ```

2. **Monitor thermal status:**
   ```bash
   nvidia-smi -q -d TEMPERATURE
   ```

3. **Control background processes:**
   ```bash
   # Check background GPU usage
   nvidia-smi -l 1
   
   # Kill competing processes if necessary
   kill -9 $(pgrep -f competing_process)
   ```

### Memory Issues

#### Issue: Out of memory errors with P2P backend

**Symptoms:**
- CUDA out of memory errors
- Shared memory allocation failures
- Buffer size too large errors

**Causes:**
- Insufficient GPU memory
- Large buffer sizes
- Memory fragmentation
- Shared memory limits

**Solutions:**

1. **Reduce buffer sizes:**
   ```python
   # Start with smaller buffer
   os.environ["EXLLAMA_P2P_BUFFER_SIZE"] = "16777216"  # 16MB
   
   # Monitor memory usage
   def print_memory_usage():
       for i in range(torch.cuda.device_count()):
           allocated = torch.cuda.memory_allocated(i) / 1024**3
           cached = torch.cuda.memory_reserved(i) / 1024**3
           print(f"GPU {i}: {allocated:.2f}GB allocated, {cached:.2f}GB cached")
   
   print_memory_usage()
   ```

2. **Optimize model loading:**
   ```python
   # Use smaller batch sizes
   model = Model.load(
       model_dir,
       tp_backend="p2p",
       devices=[0, 1],
       max_batch_size=1,  # Reduce from default
       max_seq_len=1024   # Reduce sequence length
   )
   ```

3. **Clear memory between runs:**
   ```python
   def cleanup_memory():
       torch.cuda.empty_cache()
       torch.cuda.reset_peak_memory_stats()
       import gc
       gc.collect()
   
   # Call after each model run
   cleanup_memory()
   ```

#### Issue: Shared memory allocation failures

**Symptoms:**
- Shared memory creation errors
- Permission denied for shared memory
- Buffer allocation failures

**Solutions:**

1. **Check shared memory limits:**
   ```bash
   # Linux
   cat /proc/sys/kernel/shmmax
   cat /proc/sys/kernel/shmall
   
   # Increase limits if necessary
   sudo sysctl -w kernel.shmmax=4294967296
   sudo sysctl -w kernel.shmall=4194304
   ```

2. **Use explicit buffer sizes:**
   ```python
   # Explicitly specify smaller buffer size
   model = Model.load(
       model_dir,
       tp_backend="p2p",
       devices=[0, 1],
       shbuf_size=8388608  # 8MB
   )
   ```

### Backend Selection Problems

#### Issue: Auto backend doesn't select P2P when expected

**Symptoms:**
- Auto backend selects NCCL instead of P2P
- P2P connectivity is available but not used
- Performance worse than expected

**Causes:**
- P2P detection logic issues
- Fallback behavior too aggressive
- Configuration conflicts

**Solutions:**

1. **Check available backends:**
   ```python
   from exllamav3.model.model_tp_backend import get_available_backends
   backends = get_available_backends()
   print("Available backends:", backends)
   ```

2. **Force P2P backend:**
   ```python
   # Explicitly use P2P
   model = Model.load(
       model_dir,
       tp_backend="p2p",  # Force P2P
       devices=[0, 1]
   )
   ```

3. **Debug P2P detection:**
   ```python
   from exllamav3.model.model_tp_cuda import check_p2p_connectivity
   
   devices = [0, 1]
   print("P2P connectivity check:")
   for i, dev_a in enumerate(devices):
       for j, dev_b in enumerate(devices):
           if i != j:
               torch.cuda.set_device(dev_a)
               can_access = torch.cuda.can_cast(dev_a, dev_b)
               print(f"  GPU {dev_a} -> GPU {dev_b}: {can_access}")
   ```

## Debugging Techniques

### Enable Verbose Logging

Enable detailed P2P backend logging:

```bash
export EXLLAMA_P2P_VERBOSE=1
python your_script.py
```

### Debug P2P Connectivity

Create a debug script to test P2P functionality:

```python
#!/usr/bin/env python3
import torch
from exllamav3.model.model_tp_cuda import (
    check_p2p_connectivity
)

def debug_p2p_connectivity():
    devices = list(range(torch.cuda.device_count()))
    print(f"Found {len(devices)} CUDA devices: {devices}")
    
    # Check basic connectivity
    print("\n=== P2P Connectivity Check ===")
    try:
        connected = check_p2p_connectivity(devices)
        print(f"Full P2P connectivity: {connected}")
        
        # Check pairwise connections
        for i, dev_a in enumerate(devices):
            for j, dev_b in enumerate(devices):
                if i != j:
                    torch.cuda.set_device(dev_a)
                    try:
                        can_access = torch.cuda.can_cast(dev_a, dev_b)
                        print(f"  GPU {dev_a} -> GPU {dev_b}: {can_access}")
                    except Exception as e:
                        print(f"  GPU {dev_a} -> GPU {dev_b}: Error - {e}")
        
        # Note: PyTorch automatically manages P2P access
        print("\n=== P2P Access Note ===")
        if connected:
            print("✅ P2P connectivity available - PyTorch will automatically manage access")
        else:
            print("⚠️  P2P connectivity not available")
            
    except Exception as e:
        print(f"❌ P2P connectivity check failed: {e}")

if __name__ == "__main__":
    debug_p2p_connectivity()
```

### Performance Profiling

Profile P2P backend performance:

```python
import torch
import time
from exllamav3 import Generator, Job, model_init

def profile_p2p_performance():
    devices = [0, 1]
    
    # Load model
    model, config, cache, tokenizer = model_init.init_from_dict({
        'model_dir': '/path/to/model',
        'devices': devices,
        'tp_backend': 'p2p',
        'max_batch_size': 1,
        'max_seq_len': 2048,
        'cache': True
    })
    
    generator = Generator(model, cache, tokenizer)
    
    # Test different tensor sizes
    tensor_sizes = [256, 512, 1024, 2048]
    
    for size in tensor_sizes:
        print(f"\nTesting tensor size: {size}")
        
        # Create test input
        test_prompt = "Hello world. " * (size // 12)
        input_ids = tokenizer.encode(test_prompt, add_bos=True)
        
        if len(input_ids) > size:
            input_ids = input_ids[:size]
        else:
            padding = size - len(input_ids)
            input_ids = torch.cat([input_ids, torch.zeros(padding, dtype=torch.long)])
        
        # Profile inference
        start_time = time.perf_counter()
        
        job = Job(
            input_ids=input_ids,
            max_new_tokens=100,
            temperature=0.7
        )
        generator.enqueue(job)
        results = list(generator.iterate())
        
        end_time = time.perf_counter()
        
        if results:
            tokens_generated = results[-1].get("new_tokens", 0)
            inference_time = end_time - start_time
            throughput = tokens_generated / inference_time
            
            print(f"  Time: {inference_time:.3f}s")
            print(f"  Tokens: {tokens_generated}")
            print(f"  Throughput: {throughput:.2f} tokens/s")
    
    generator.close()
    model.unload()

if __name__ == "__main__":
    profile_p2p_performance()
```

## Hardware Compatibility

### Supported GPU Architectures

The P2P backend supports most modern NVIDIA GPUs with P2P capabilities:

**Fully Supported:**
- NVIDIA Ampere (RTX 30xx, A100)
- NVIDIA Turing (RTX 20xx)
- NVIDIA Volta (V100)
- NVIDIA Pascal (P100, GTX 10xx)

**Limited Support:**
- NVIDIA Maxwell (GTX 9xx) - Basic P2P support
- Mixed architectures - May have reduced performance

**Check GPU compatibility:**
```bash
nvidia-smi --query-gpu=name,architecture,pstate --format=csv
```

### Multi-GPU Setup Guidelines

**Optimal Configurations:**
- 2-4 identical GPUs for best performance
- Same GPU model and memory size recommended
- Sufficient cooling to prevent thermal throttling
- NVLink connection for best performance

**Minimum Requirements:**
- CUDA 11.0 or later
- NVIDIA drivers 450.80.02 or later
- At least 16GB GPU memory per device
- P2P connectivity between all GPUs

### Power and Cooling

**Power Requirements:**
- High-wattage power supply (750W+ for 4 GPUs)
- Separate power circuits for GPU systems
- Stable voltage regulation

**Cooling Requirements:**
- Adequate case airflow
- GPU fans at appropriate speeds
- Monitor temperatures during heavy load:
  ```bash
  nvidia-smi -q -d TEMPERATURE
  ```

## Environment Configuration

### System Requirements

**Linux (Recommended):**
- Ubuntu 20.04 LTS or later
- Linux kernel 5.4 or later
- systemd for process management

**Windows:**
- Windows 10/11 with WSL2 (recommended)
- Or native Windows with proper CUDA installation

**macOS:**
- Not officially supported for P2P functionality
- Use Linux virtual machine if needed

### Environment Variables

**P2P-specific variables:**
```bash
# Enable verbose P2P logging
export EXLLAMA_P2P_VERBOSE=1

# Set P2P buffer size (bytes)
export EXLLAMA_P2P_BUFFER_SIZE=33554432  # 32MB

# Set master address for TCP communication
export EXLLAMA_MASTER_ADDR=127.0.0.1

# Set master port for TCP communication
export EXLLAMA_MASTER_PORT=29500
```

**CUDA environment variables:**
```bash
# Set CUDA device selection
export CUDA_VISIBLE_DEVICES=0,1,2

# Enable CUDA error checking
export CUDA_LAUNCH_BLOCKING=1

# Set CUDA memory pool allocator
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.9
```

### Docker Configuration

**Dockerfile for P2P testing:**
```dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Set environment variables
ENV EXLLAMA_P2P_VERBOSE=1
ENV EXLLAMA_P2P_BUFFER_SIZE=33554432

WORKDIR /app
COPY . /app

CMD ["python3", "your_script.py"]
```

## Advanced Troubleshooting

### Kernel Module Issues

**Check kernel modules:**
```bash
# Load required modules
sudo modprobe nvidia
sudo modprobe nvidia_drm
sudo modprobe nvidia_uvm

# Check loaded modules
lsmod | grep nvidia

# Check kernel messages
dmesg | grep -i nvidia
```

**IOMMU configuration (Linux):**
```bash
# Check IOMMU groups
for d in /sys/kernel/iommu_groups/*/devices/*; do 
    n=${d#*/iommu_groups/*}; n=${n%%/*}
    printf 'IOMMU Group %s ' "$n"
    lspci -nns "${d##*/}" 
done

# Enable IOMMU if needed
sudo nano /etc/default/grub
# Add intel_iommu=on or amd_iommu=1 to GRUB_CMDLINE_LINUX
sudo update-grub
sudo reboot
```

### Performance Analysis

**Use NVIDIA Nsight Systems:**
```bash
# Install Nsight Systems
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y nsys

# Profile your application
nsys profile -o profile ./your_script.py

# Analyze results
nsys stats profile.nsys-rep
```

**Use PyTorch profiling:**
```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CUDA], profile_memory=True) as prof:
    with record_function("model_inference"):
        # Your model inference code here
        pass

print(prof.key_averages())
```

### System Monitoring

**Real-time monitoring script:**
```python
#!/usr/bin/env python3
import time
import torch
import psutil
import subprocess

def monitor_system():
    print("Starting system monitoring...")
    
    try:
        while True:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            # GPU usage
            gpu_info = []
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                cached = torch.cuda.memory_reserved(i) / 1024**3
                
                # Get GPU utilization using nvidia-smi
                try:
                    result = subprocess.run(
                        f"nvidia-smi --query-gutilization.gpu --format=csv,noheader,nounits -i {i}",
                        shell=True, capture_output=True, text=True
                    )
                    gpu_util = float(result.stdout.strip()) if result.stdout.strip() else 0
                except:
                    gpu_util = 0
                
                gpu_info.append(f"GPU{i}: {gpu_util:.1f}% util, {allocated:.2f}GB alloc")
            
            # Print status
            print(f"\rCPU: {cpu_percent:.1f}% | Mem: {memory_percent:.1f}% | {' | '.join(gpu_info)}", end="")
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    monitor_system()
```

## Getting Help

### Community Resources

- **ExLlamaV3 Discord**: Join for community support
- **GitHub Issues**: Report bugs and request features
- **NVIDIA Developer Forum**: For hardware-specific issues

### Bug Reports

When reporting bugs, include:

1. **System Information:**
   ```python
   import torch
   import platform
   
   print(f"Python: {platform.python_version()}")
   print(f"PyTorch: {torch.__version__}")
   print(f"CUDA: {torch.version.cuda}")
   print(f"GPU Count: {torch.cuda.device_count()}")
   for i in range(torch.cuda.device_count()):
       print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
   ```

2. **Error Messages:** Full stack trace
3. **Configuration:** Model size, batch size, backend type
4. **Reproduction Steps:** Minimal example that reproduces the issue

### Performance Benchmarks

When reporting performance issues, include:

1. **Hardware specifications**
2. **Software versions** (OS, drivers, CUDA, PyTorch)
3. **Benchmark results** from `p2p_benchmark.py`
4. **Comparison with other backends** if applicable
5. **System load** during testing

---

*This troubleshooting guide will be updated as new issues are discovered and resolved. Check back for the latest solutions and debugging techniques.*