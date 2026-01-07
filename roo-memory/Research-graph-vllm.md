# vLLM CUDA Graph Implementation Research

## Overview

Research into vLLM's CUDA graph capture implementation for MLP and attention layers to understand their approach and compare with ExLlamaV3's implementation.

## Key Architecture

### Python-Native Approach

vLLM uses **PyTorch's built-in `torch.cuda.graph()`** API, NOT manual CUDA graph capture at C++ level.

**Files:**
- `vllm/compilation/cuda_graph.py` - CUDA graph wrapper
- `vllm/compilation/wrapper.py` - torch.compile wrapper
- `vllm/compilation/backends.py` - Compiler manager
- `vllm/compilation/decorators.py` - Layer decorators

### Two-Level Graph System

```python
# Level 1: torch.compile with fullgraph=True
self._compiled_callable = torch.compile(
    self.forward,
    fullgraph=True,      # Capture entire layer as single graph
    dynamic=False,
    backend=backend,     # Usually "inductor"
)

# Level 2: Wrap with CUDA graph capture/replay
CUDAGraphWrapper(runnable, vllm_config, runtime_mode)
```

### Key Components

#### 1. CUDAGraphWrapper (`cuda_graph.py:137-302`)

```python
class CUDAGraphWrapper:
    """
    Wraps a runnable to add CUDA graph capturing and replaying ability.

    Workflow:
    1. At initialization, a runtime mode is assigned (FULL or PIECEWISE)
    2. At runtime, receives batch_descriptor from forward context
    3. If batch_descriptor not in cache → capture new graph
    4. If batch_descriptor in cache → replay existing graph
    """
```

**Key features:**
- Captures graphs per `BatchDescriptor` (shape signature)
- Uses `torch.cuda.CUDAGraph()` for capture
- Replays graphs with `.replay()` method
- Weak references for outputs to save memory
- Graph pool for reduced memory allocation

#### 2. TorchCompileWithNoGuardsWrapper (`wrapper.py:82-320`)

```python
class TorchCompileWithNoGuardsWrapper:
    def __init__(self):
        self._compiled_callable = torch.compile(
            self.forward,
            fullgraph=True,      # Capture entire layer
            dynamic=False,
            backend=backend,
        )
```

**Key features:**
- Uses `fullgraph=True` to capture entire layer as single graph
- Drops all guards for performance (no recompilation checks)
- Supports AOT compilation
- NVTX tracing for profiling

## What Gets Captured in Graphs

### vLLM Approach: **ALL Operations**

Since vLLM uses `torch.compile` with `fullgraph=True`, the entire forward pass is captured:

✅ **In the graph (via torch.compile + Inductor):**
- GEMM operations (matrix multiplications)
- RMS norm / Layer norm
- RoPE (rotary position embedding)
- Activations (SiLU, GeLU, etc.)
- Reshape, transpose, view operations
- Everything in the forward pass

### Graph Capture Modes

vLLM supports different CUDA graph modes:

1. **FULL Mode**: Entire model as one graph
2. **PIECEWISE Mode**: Each layer as separate graph
3. **NONE Mode**: No CUDA graph (eager execution)

## Implementation Details

### Graph Capture Process

```python
# From cuda_graph.py:248-288
with torch.cuda.graph(cudagraph, pool=self.graph_pool):
    output = self.runnable(*args, **kwargs)
    if self.cudagraph_options.weak_ref_output:
        output = weak_ref_tensors(output)
```

### Graph Replay

```python
# From cuda_graph.py:301-302
entry.cudagraph.replay()
return entry.output
```

### Batch Descriptors

Graphs are keyed by `BatchDescriptor` which includes:
- Number of tokens
- Batch size
- Other shape information

This allows vLLM to maintain multiple graphs for different input shapes.

## Comparison: vLLM vs ExLlamaV3

| Aspect | vLLM | ExLlamaV3 (Current) |
|--------|------|---------------------|
| **Level** | Python (`torch.cuda.graph`) | C++ (manual CUDA graph API) |
| **Graph Content** | Entire compiled layer (Inductor) | Only GEMM operations |
| **Ops in Graph** | ALL (GEMM + norms + RoPE + activations) | GEMM only |
| **Norms/RoPE** | In graph (via torch.compile) | Eager (outside graph) |
| **Backend** | PyTorch Inductor | Custom kernels |
| **Memory** | Graph pool, weak refs | Manual temp tensors |
| **Portability** | Requires PyTorch 2.x | Works with older PyTorch |
| **Complexity** | High (compilation manager, backends) | Low (direct C++) |

## Advantages of vLLM Approach

### Pros
1. **More in graph**: Captures norms, RoPE, activations - potentially larger speedup
2. **Automatic optimization**: Inductor fuses operations
3. **Python-level**: No C++ code needed for graph capture
4. **Flexible**: Can switch between eager, compiled, and CUDA graph modes

### Cons
1. **PyTorch 2.x required**: Needs recent PyTorch version
2. **Complex infrastructure**: Compilation manager, backends, decorators
3. **Inductor overhead**: First run has compilation cost
4. **Memory**: Graph pool management adds complexity

## Advantages of ExLlamaV3 Approach

### Pros
1. **Simpler**: Direct C++ CUDA graph capture
2. **Compatible**: Works with older PyTorch versions
3. **Predictable**: Only GEMM in graph (easy to reason about)
4. **Low overhead**: No compilation step
5. **Custom kernels**: Optimized for specific hardware

### Cons
1. **Less in graph**: Only GEMM, norms/RoPE are eager
2. **C++ maintenance**: Need to maintain C++ code
3. **Manual optimization**: No automatic fusion from Inductor

## Key Learnings for ExLlamaV3

### Current Implementation is Sound

The ExLlamaV3 approach of capturing only GEMM in CUDA graphs is **consistent with the pattern used in `mlp.cpp`**:

```cpp
// ExLlamaV3 pattern (attn.cpp, mlp.cpp)
if (!graph.ready) {
    graph.capture_begin();
    // Capture only GEMM operations
    q_proj->run_gr(x, temp_q, &graph);
    k_proj->run_gr(x, temp_k, &graph);
    v_proj->run_gr(x, temp_v, &graph);
    graph.capture_end();
}
graph.launch(args, stream);

// Norms run eagerly
rms_norm(temp_q, q_norm_weight, temp_q, norm_epsilon, 0.0, false);
```

This is a **conservative but safe approach** that:
- Avoids complexity of capturing variable operations (RoPE with changing positions)
- Works without PyTorch 2.x Inductor
- Is easier to debug and maintain

### Potential Improvements

Based on vLLM research, potential enhancements for ExLlamaV3:

1. **Capture RMS Norm in Graph**
   - RMS norm has fixed operations (element-wise)
   - Only input tensor pointer changes
   - Could use pointer update mechanism like GEMM

2. **Static RoPE Capture**
   - For fixed position encodings (e.g., during prefill)
   - Pre-compute sin/cos for max sequence length
   - Capture in graph when position doesn't change

3. **Graph Pool**
   - vLLM uses `torch.cuda.graph.Pool` for reduced memory allocation
   - ExLlamaV3 could implement similar pool for multiple graphs

4. **Multiple Graphs per Layer**
   - vLLM maintains graphs for different batch descriptors
   - ExLlamaV3 could capture graphs for common sequence lengths

## File References

### vLLM Key Files

```
vllm/compilation/
├── cuda_graph.py           # CUDA graph wrapper implementation
├── wrapper.py              # torch.compile wrapper
├── backends.py             # Compiler manager
├── decorators.py           # Layer decorators
├── base_static_graph.py    # Base static graph class
├── compiler_interface.py   # Compiler interface
└── fusion*.py              # Various fusion passes
```

### ExLlamaV3 Key Files

```
exllamav3/exllamav3_ext/libtorch/
├── attn.cpp                # Attention CUDA graph implementation
├── attn.h                  # Attention CUDA graph header
├── mlp.cpp                 # MLP CUDA graph implementation
├── mlp.h                   # MLP CUDA graph header
└── graph.cuh               # CUDA graph utilities

exllamav3/modules/
└── attn.py                 # Python attention layer with BC optimization
```

## Conclusion

vLLM's CUDA graph implementation is **more comprehensive** but also **more complex**, relying on PyTorch 2.x's Inductor compiler to capture entire layers as graphs.

ExLlamaV3's current approach is **simpler and more portable**, capturing only GEMM operations in graphs at the C++ level. This is a reasonable trade-off that:
- Provides significant speedup from reduced kernel launch overhead
- Maintains compatibility with older PyTorch versions
- Keeps code complexity manageable
- Follows the established pattern in `mlp.cpp`

The main opportunity for improvement would be to **capture RMS norm in the graph** since it's a fixed operation with only the input pointer changing (similar to GEMM).
