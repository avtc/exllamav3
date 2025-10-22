
# <img src="doc/cat.png" width="40"> ExLlamaV3

ExLlamaV3 is an inference library for running local LLMs on modern consumer GPUs. Headline features:

- New [EXL3](doc/exl3.md) quantization format based on QTIP
- Flexible tensor-parallel and expert-parallel inference for consumer hardware setups
- **P2P-optimized backend** for high-performance GPU-to-GPU communication
- OpenAI-compatible server provided via [TabbyAPI](https://github.com/theroyallab/tabbyAPI/)
- Continuous, dynamic batching
- HF Transformers plugin (see [here](examples/transformers_integration.py))
- HF model support (see [supported architectures](#architecture-support))
- Speculative decoding
- 2-8 bit cache quantization
- Multimodal support

The official and recommended backend server for ExLlamaV3 is [TabbyAPI](https://github.com/theroyallab/tabbyAPI/), which provides an OpenAI-compatible API for local or remote inference, with extended features like HF model downloading, embedding model support and support for HF Jinja2 chat templates.

## P2P Backend Support

ExLlamaV3 includes a **P2P (Peer-to-Peer) backend** for optimized tensor parallel inference on systems with fully connected GPUs. This backend provides direct GPU-to-GPU communication capabilities that can significantly improve performance for multi-GPU setups.

### Benefits of P2P Backend

- **Reduced latency**: Direct GPU-to-GPU communication eliminates CPU bottlenecks
- **Higher throughput**: Optimized communication primitives for fully connected GPU systems
- **Automatic detection**: Seamless integration with existing tensor parallel workflows
- **Fallback support**: Graceful degradation to NCCL or native backends when P2P is unavailable

### When to Use P2P vs Other Backends

| Backend | Best For | Requirements | Performance |
|---------|----------|--------------|-------------|
| **P2P** | Fully connected multi-GPU systems | All GPUs can access each other via P2P | **Highest** - Direct GPU communication |
| **NCCL** | Mixed connectivity systems | NCCL support and standard GPU setup | **High** - Optimized for general multi-GPU |
| **Native** | Limited connectivity or debugging | No special requirements | **Medium** - CPU-mediated communication |

### P2P Backend Configuration

The P2P backend can be configured using the `tp_backend` parameter when loading models:

```python
# Automatic P2P detection (recommended)
model = Model.load(
    model_dir,
    tp_backend="auto",  # Automatically detects and uses P2P if available
    devices=[0, 1, 2]  # Multiple GPU devices
)

# Explicit P2P usage
model = Model.load(
    model_dir,
    tp_backend="p2p",  # Force P2P backend
    devices=[0, 1, 2]
)
```

### Installation and Setup

The P2P backend is included with ExLlamaV3 and requires:

- **CUDA 11.0 or later**: For GPU P2P capabilities
- **Fully connected GPUs**: All GPUs must be able to access each other via P2P
- **NVIDIA drivers**: Up-to-date drivers supporting peer-to-peer communication

#### Verifying P2P Connectivity

You can test P2P connectivity with the built-in utility:

```python
from exllamav3.model.model_tp_cuda import check_p2p_connectivity

devices = [0, 1, 2]  # Your GPU device IDs
if check_p2p_connectivity(devices):
    print("P2P connectivity available!")
else:
    print("P2P connectivity not available - using fallback backend")
```

#### Environment Variables

- `EXLLAMA_P2P_VERBOSE`: Enable verbose P2P logging (set to "1" for debugging)
- `EXLLAMA_P2P_BUFFER_SIZE`: Configure P2P buffer size in bytes (default: 16MB)

### Performance Considerations

- **Best performance** on systems with 2-8 fully connected GPUs
- **Optimal for** transformer models with frequent tensor parallel operations
- **Memory overhead**: Slightly higher than NCCL due to direct P2P buffers
- **Scalability**: Tested on systems up to 8 GPUs

### Troubleshooting

If you encounter issues with the P2P backend, see [P2P Troubleshooting Guide](docs/p2p_troubleshooting.md) for common solutions and debugging tips.

### ⚠️ Important

- **Qwen3-Next** support is currently experimental and still requires some optimization, so don't expect
  optimal performance just yet. [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) is required
  and this in turn requires Triton. [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) is supported and 
  recommended but not required.
- **Qwen3-Next** currently does not support tensor/expert parallelism.

## Architecture support

- **AFM** (ArceeForCausalLM)
- **Apertus** (ApertursForCausalLM)
- **Command-R** etc. (CohereForCausalLM)
- **Command-A**, **Command-R7B**, **Command-R+** etc. (Cohere2ForCausalLM)
- **DeciLM**, **Nemotron** (DeciLMForCausalLM)
- **dots.llm1** (Dots1ForCausalLM)
- **ERNIE 4.5** (Ernie4_5_ForCausalLM, Ernie4_5_MoeForCausalLM)
- **EXAONE 4.0** (Exaone4ForCausalLM)
- **Gemma 2** (Gemma2ForCausalLM)
- **Gemma 3** (Gemma3ForCausalLM, Gemma3ForConditionalGeneration) *- multimodal*
- **GLM 4**, **GLM 4.5**, **GLM 4.5-Air**, (Glm4ForCausalLM, Glm4MoeForCausalLM)
- **Llama**, **Llama 2**, **Llama 3**, **Llama 3.1-Nemotron** etc. (LlamaForCausalLM)
- **MiMo-RL** (MiMoForCausalLM)
- **Mistral**, **Mistral 3** etc. (MistralForCausalLM, Mistral3ForConditionalGeneration) *- multimodal*
- **Mixtral** (MixtralForCausalLM)
- **Phi3**, **Phi4** (Phi3ForCausalLM)
- **Qwen 2**, **Qwen 2.5** (Qwen2ForCausalLM)
- **Qwen 3** (Qwen3ForCausalLM, Qwen3MoeForCausalLM)
- **Qwen 3-Next** (Qwen3NextForCausalLM)
- **Seed-OSS** (SeedOssForCausalLM)
- **SmolLM** (SmolLM3ForCausalLM)

Always adding more, stay tuned.


## What's missing?

Currently on the to-do list:

- Lots of optimization
- LoRA support
- ROCm support
- More sampling functions
- More quantization modes (FP4 etc.)

As for what is implemented, expect that some things may be a little broken at first. Please be patient, raise issues and/or contribute. 👉👈 


## How to?

[TabbyAPI](https://github.com/theroyallab/tabbyAPI/) has a startup script that manages and installs prerequisites if you want to get started quickly with inference in an OAI-compatible client. 

Otherwise, start by making sure you have the appropriate version of [PyTorch](https://pytorch.org/get-started/locally/) installed (CUDA 12.4 or later) since the Torch dependency is not automatically handled by `pip`. Then pick a method below:

### Method 1: Installing from prebuilt wheel (recommended if you're unsure)

Pick a wheel from the [releases page](https://github.com/turboderp-org/exllamav3/releases), then e.g.:

```sh
pip install https://github.com/turboderp-org/exllamav3/releases/download/v0.0.6/exllamav3-0.0.6+cu128.torch2.8.0-cp313-cp313-linux_x86_64.whl
```

### Method 2: Installing from PyPi:

```sh
pip install exllamav3
```
Note that the PyPi package does not contain a prebuilt extension and requires the CUDA toolkit and build prerequisites (i.e. VS Build Tools on Windows, gcc on Linux, `python-dev` headers etc.).    

### Method 3: Building from source

```sh
# Clone the repo
git clone https://github.com/turboderp-org/exllamav3
cd exllamav3

# (Optional) switch to dev branch for latest in-progress features
git checkout dev

# Install requirements (make sure you install Torch separately)
pip install -r requirements.txt
```

At this point you should be able to run the conversion, eval and example scripts from the main repo directory, e.g. `python convert.py -i ...`

To install the library for the active venv, run from the repo directory:

```sh
pip install .
```

Relevant env variables for building:
- `MAX_JOBS`: by default ninja may launch too many processes and run out of system memory for compilation. Set this to a reasonable value like 4 in that case.  
- `EXLLAMA_NOCOMPILE`: set to install the library without compiling the C++/CUDA extension. Torch will build/load it at runtime instead.


## Conversion

To convert a model to EXL3 format, use:

```sh
# Convert model
python convert.py -i <input_dir> -o <output_dir> -w <working_dir> -b <bitrate>

# Resume an interrupted quant job
python convert.py -w <working_dir> -r

# More options
python convert.py -h
```

The working directory is temporary storage for state checkpoints and for storing quantized tensors until the converted model can be compiled. It should have enough free space to store an entire copy of the output model. Note that while EXL2 conversion by default resumes an interrupted job when pointed to an existing folder, EXL3 needs you to explicitly resume with the `-r`/`--resume` argument.    

See [here](doc/convert.md) for more information.


## Examples

A number of example scripts are provided to showcase the features of the backend and generator. Some of them have hardcoded model paths and should be edited before you run them, but there is a simple CLI chatbot that you can start with:

```sh
python examples/chat.py -m <input_dir> -mode <prompt_mode> 

# E.g.:
python examples/chat.py -m /mnt/models/llama3.1-8b-instruct-exl3 -mode llama3

# Wealth of options
python examples/chat.py -h
```

## EXL3 quantization

<div align="center">
    <a href="doc/exl3.md" target="_blank">
        <img src="doc/llama31_8b_instruct_bpw.png" width="640">
    </a>
</div>

Despite their amazing achievements, most SOTA quantization techniques remain cumbersome or even prohibitively expensive to use. For instance, **AQLM** quantization of a 70B model takes around **720 GPU-hours** on an A100 server, costing $850 US at the time of writing. ExLlamaV3 aims to address this with the **EXL3** format, which is a streamlined variant of [**QTIP**](https://github.com/Cornell-RelaxML/qtip) from Cornell RelaxML. The conversion process is designed to be simple and efficient and requires only an input model (in HF format) and a target bitrate. By computing Hessians on the fly and thanks to a fused Viterbi kernel, the quantizer can convert a model in a single step, taking a couple of minutes for smaller models, up to a few hours for larger ones (70B+) (on a single RTX 4090 or equivalent GPU.)

The [Marlin](https://github.com/IST-DASLab/marlin)-inspired GEMM kernel achieves roughly memory-bound latency under optimal conditions (4bpw, RTX 4090), though it still needs some work to achieve the same efficiency on Ampere GPUs and to remain memory-bound at lower bitrates.

Since converted models largely retain the original file structure (unlike **EXL2** which renames some tensors in its quest to turn every model into a Llama variant), it will be possible to extend **EXL3** support to other frameworks like HF Transformers and vLLM.

There are some benchmark results [here](doc/exl3.md), and a full writeup on the format is coming soon.

Fun fact: Llama-3.1-70B-EXL3 is coherent at 1.6 bpw. With the output layer quantized to 3 bpw and a 4096-token cache, inference is possible in under 16 GB of VRAM. 


### Community

You are always welcome to join the [ExLlama discord server](https://discord.gg/NSFwVuCjRq) ←🎮  


### 🤗 HuggingFace repos

A selection of EXL3-quantized models is available [here](https://huggingface.co/collections/turboderp/exl3-models-67f2dfe530f05cb9f596d21a). Also shout out the following lovely people:
 
- [ArtusDev](https://huggingface.co/ArtusDev)
- [MikeRoz](https://huggingface.co/MikeRoz) 
- [MetaphoricalCode](https://huggingface.co/MetaphoricalCode) 
- [Ready.Art](https://huggingface.co/ReadyArt) 
- [isogen](https://huggingface.co/isogen/models)


## Acknowledgements

This project owes its existence to a wonderful community of FOSS developers and some very generous supporters (🐈❤️!) The following projects in particular deserve a special mention:

- [TabbyAPI](https://github.com/theroyallab/tabbyAPI/)
- [PyTorch](https://github.com/pytorch/pytorch)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [QTIP](https://github.com/Cornell-RelaxML/qtip)
- [Transformers](https://github.com/huggingface/transformers)
- [Marlin](https://github.com/IST-DASLab/marlin)
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)