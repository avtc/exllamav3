# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ExLlamaV3 is a high-performance inference library for running local LLMs on consumer GPUs. It features the EXL3 quantization format (based on QTIP), tensor-parallel and expert-parallel inference, continuous batching, and multimodal model support.

**Language:** Python with CUDA/C++ extensions (via PyTorch)
**Primary use case:** Fast, memory-efficient LLM inference on consumer hardware (RTX 3080/4090, etc.)

## Common Development Commands

### Installation and Building

```bash
# Install from source (after installing PyTorch separately)
pip install -r requirements.txt
pip install .

# Environment variables for building
MAX_JOBS=4                    # Limit compilation parallelism
EXLLAMA_NOCOMPILE=1          # Skip pre-compilation (JIT at runtime)
EXLLAMA_VERBOSE=1            # Verbose build output
```

### Model Conversion

```bash
# Convert HF model to EXL3 format
python convert.py -i <input_dir> -o <output_dir> -w <working_dir> -b <bitrate>

# Resume interrupted quantization
python convert.py -w <working_dir> -r

# Multi-GPU quantization
python convert.py -i <input_dir> -o <output_dir> -w <working_dir> -b <bitrate> -d 0,1,2

# View all options
python convert.py -h
```

The working directory (`-w`) needs enough free space to store a full copy of the output model. See `doc/convert.md` for detailed conversion options.

### Testing

```bash
# Run specific test
pytest tests/test_sampler.py

# Run all tests
pytest tests/

# Run tests for specific module
pytest tests/test_cache_rotate.py tests/test_kv_quant.py
```

### Running Examples

```bash
# Chat interface
python examples/chat.py -m <model_dir> -mode llama3

# View all example scripts
ls examples/
```

### OpenSpec (Specification-Driven Development)

```bash
# List active changes
openspec list

# List specifications
openspec list --specs

# Validate a change
openspec validate <change-id> --strict

# Show change or spec details
openspec show <item>

# Archive completed change
openspec archive <change-id> --yes
```

See `openspec/AGENTS.md` for complete OpenSpec workflow documentation.

## Architecture Overview

### Core Components

**Model Layer** (`exllamav3/model/`)
- `model.py`: Main `Model` class supporting tensor parallelism
- `config.py`: Base `Config` class for model configuration
- Models are loaded from HuggingFace-style configs and safetensors weights

**Architecture Implementations** (`exllamav3/architecture/`)
- Each supported model family has its own file (e.g., `llama.py`, `mistral.py`, `qwen3.py`)
- Pattern: `LlamaConfig(Config)` + `LlamaModel(Model)` classes
- `arch_string` class attribute identifies the HF model architecture
- New model support: Copy an existing architecture and adapt

**Generation Engine** (`exllamav3/generator/`)
- `generator.py`: `Generator` class - main synchronous inference engine
- `async_generator.py`: `AsyncGenerator` - asynchronous streaming generation
- `job.py`: `Job` and `AsyncJob` classes encapsulate generation requests
- `sampler/`: Sampling strategies (TopKSampler, TopPSampler, ComboSampler, CustomSampler)
- `filter/`: Output filters (grammar constraints, JSON schema, etc.)

**Quantization System** (`exllamav3/conversion/`)
- `convert_model.py`: Main conversion script entry point
- `allocation.py`: Bitrate allocation across tensors
- `optimize_model.py`: QTIP-based quantization optimization
- `compile.py`: Compile quantized model to EXL3 format

**Cache Management** (`exllamav3/cache/`)
- KV cache implementation with optional quantization
- `CacheLayer_fp16`: Full-precision cache
- `CacheLayer_quant`: 2-8 bit quantized cache

**Neural Network Modules** (`exllamav3/modules/`)
- `linear.py`: `Linear` - quantized linear layers with EXL3 GEMM kernel
- `attn.py`: `Attention` - fused attention implementations (SDPA, flash-attn)
- `mlp.py`: `MLP`, `GatedMLP` - feedforward layers
- `transformer.py`: `TransformerBlock` - transformer layer combining attn + MLP
- `block_sparse_mlp.py`: `BlockSparseMLP` - MoE expert layers

**Tokenizer** (`exllamav3/tokenizer/`)
- Wraps HuggingFace tokenizers
- Multimodal embedding support for vision-language models

### Loading Models

The `model_init` module provides a convenience API for examples:

```python
from exllamav3 import Generator, Job, model_init

# Add CLI arguments
parser = argparse.ArgumentParser()
model_init.add_args(parser)

# Initialize model from args
model, config, cache, tokenizer = model_init.init(args)

# Create generator
generator = Generator(model=model, cache=cache, tokenizer=tokenizer)
```

### Multi-GPU Support

- **Tensor Parallelism**: Model weights split across GPUs (`-tp` flag)
- **Expert Parallelism**: MoE experts distributed across GPUs
- Configurable parallelism per layer type (`-tp_attn`, `-tp_mlp`, `-tp_moe`, `-tp_linear`)
- Two backends: `native` (default) or `nccl`

## Quantization Details

### EXL3 Format

- Based on QTIP (Cornell RelaxML)
- Variable bitrate: 1-8 bits per weight
- Output layer typically quantized to 3-6 bits (`-hb` flag)
- Marlin-inspired GEMM kernel for efficient inference
- Models largely retain original file structure (unlike EXL2)

### Bitrate Selection

Lower bitwidths trade quality for VRAM savings:
- 4.0-6.0 bpw: Good quality for most models
- 2.0-3.0 bpw: Usable but noticeable degradation
- <2.0 bpw: Experimental, often incoherent

## Adding Model Support

To add support for a new model architecture:

1. Create `exllamav3/architecture/<model_name>.py`
2. Define config class inheriting from `Config`:
   ```python
   class NewModelConfig(Config):
       arch_string = "NewModelForCausalLM"  # HF architecture identifier
   ```
3. Define model class inheriting from `Model`:
   ```python
   class NewModel(Model):
       config_class = NewModelConfig
   ```
4. Implement `__init__` to build module graph from config
5. Import in `exllamav3/architecture/__init__.py`

Reference existing implementations (e.g., `llama.py`) for patterns.

## Key Dependencies

- **PyTorch** (>=2.6.0): Core framework with CUDA extensions
- **Flash Attention** (>=2.7.4): Optimized attention implementation
- **Tokenizers** (>=0.21.1): HuggingFace tokenizer library
- **SafeTensors**: Safe tensor loading/saving
- **kbnf/formatron**: Grammar parsing for structured generation
- **Rich**: Terminal output formatting
- **Pydantic**: Data validation

## Development Notes

### Qwen3-Next Support

Experimental support for Qwen3-Next models:
- Requires [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) (needs Triton)
- [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) recommended but not required
- No tensor/expert parallelism support currently

### File Structure Patterns

- Modules use `__init__` to expose public API
- Architecture-specific code isolated in `architecture/` directory
- CUDA/C++ extensions in `exllamav3/exllamav3_ext/`
- Examples show typical usage patterns

### Performance Considerations

- Optimized for Ada Lovelace (RTX 4090) GPUs
- Ampere (RTX 3090) performance improving but not optimal
- Memory-bound at 4bpw under optimal conditions
- Multi-GPU quantization can reduce conversion time significantly

## Integration Points

### TabbyAPI

Official and recommended backend server providing:
- OpenAI-compatible API
- HF model downloading
- Embedding model support
- HF Jinja2 chat template support

Repository: https://github.com/theroyallab/tabbyAPI/

### HF Transformers Plugin

See `examples/transformers_integration.py` for HF Transformers integration.

## Documentation

- `doc/exl3.md`: EXL3 format details and benchmarks
- `doc/convert.md`: Conversion script documentation
- `README.md`: Overview, installation, and usage examples
- `openspec/AGENTS.md`: OpenSpec specification-driven development workflow
