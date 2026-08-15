# Setup Guide

These steps are meant to run on **your own machine** (CPU-only), not in
any sandbox. Everything here is local — no accounts, no API keys.

## 1. Prerequisites

- Python 3.10+
- A C++ compiler toolchain (needed to build `llama-cpp-python`):
  - Linux: `sudo apt install build-essential cmake`
  - macOS: `xcode-select --install`
  - Windows: install "Desktop development with C++" via Visual Studio
    Build Tools, or use WSL2 (recommended — simpler on Windows)
- ~10GB free disk space (model file + dependencies)

## 2. Create a virtual environment

```bash
cd local-agent
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install --upgrade pip
```

## 3. Install PyTorch (CPU-only build) — do this FIRST

`sentence-transformers` depends on PyTorch. If you `pip install` it
directly, pip will default to the full CUDA-enabled build, which is
several GB of unnecessary NVIDIA libraries on a CPU-only machine.
Install the CPU-only build explicitly instead:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 4. Install the rest of the dependencies

```bash
pip install -r requirements.txt
```

Note: `llama-cpp-python` compiles the llama.cpp C++ backend from
source during this step. That's normal and expected — it can take
5–15 minutes depending on your machine. It is not stuck; let it run.

**Performance tip:** if you know your CPU's core count, you can speed
up both the build and later inference:

```bash
CMAKE_ARGS="-DGGML_NATIVE=ON" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

This lets llama.cpp compile with optimizations specific to your CPU
(AVX2/AVX-512 etc. if available), which noticeably improves inference
speed.

## 5. Download a local model (GGUF format)

We're using **Qwen2.5-7B-Instruct**, quantized to Q4_K_M (a good
balance of quality and speed/size for CPU). Download it manually from
Hugging Face and place it in the `models/` folder:

```bash
mkdir -p models
# Download Qwen2.5-7B-Instruct-Q4_K_M.gguf (~4.7GB) from:
# https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF
# and save it as models/model.gguf
```

If you have the `huggingface_hub` CLI installed, this does it in one
command:

```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF \
    qwen2.5-7b-instruct-q4_k_m.gguf --local-dir models
mv models/qwen2.5-7b-instruct-q4_k_m.gguf models/model.gguf
```

If 7B feels too slow once you benchmark it (step 6), Qwen2.5-3B-Instruct-GGUF
is a lighter drop-in replacement — same steps, smaller download.

## 6. Benchmark on your machine

Before building anything on top, confirm the model loads and check
real generation speed:

```bash
python -c "
import time
from agent.llm import LocalLLM, LLMConfig

llm = LocalLLM(LLMConfig(model_path='models/model.gguf'))
start = time.time()
output = llm.generate('Explain what a vector database is in two sentences.', max_tokens=100)
elapsed = time.time() - start
print(output)
print(f'\\n{elapsed:.1f}s for ~100 tokens')
"
```

A few tokens/second is normal and workable for this kind of agent. If
it's dramatically slower than that, the CPU-native build flag in step
4 usually helps a lot.

## 7. Run the tests

```bash
pytest tests/ -v
```

All tests except model-dependent ones should pass without needing the
model file. (We validated all of these already in development — see
docs/architecture.md for what's been tested.)

## 8. Try it out

```bash
# Add a document to the knowledge base
python -m agent.cli add /path/to/some_document.pdf

# See what's in the knowledge base
python -m agent.cli list

# Chat with it
python -m agent.cli chat --model models/model.gguf --show-reasoning
```

Inside the chat session:
- Type `cite on` / `cite off` to toggle source citations
- Type `exit` to quit
