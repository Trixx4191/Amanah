# Local Document-Learning Agent

A fully local AI agent that learns from PDFs, Word docs, and text
files, reasons through questions rather than just pattern-matching,
and keeps track of conversation context — all on CPU, with no API
keys and no cloud dependencies.

## Quick start

See **[docs/setup.md](docs/setup.md)** for full setup instructions —
installing dependencies correctly (CPU-only PyTorch), downloading a
local model, and running your first query.

```bash
python -m agent.cli add /path/to/document.pdf
python -m agent.cli chat --model models/model.gguf --show-reasoning
```

## How it works

See **[docs/architecture.md](docs/architecture.md)** for the full
design writeup: how each component works, why it's built the way it
is, and what's been tested vs. what needs your own machine to verify.

See **[docs/project_plan.md](docs/project_plan.md)** for the original
planning document and the decisions that shaped this build.

## Project structure

```
agent/
  ingestion.py       Parses PDF/DOCX/TXT/MD into text + metadata
  chunking.py         Splits text into overlapping, coherent chunks
  embeddings.py        Local embedding model wrapper
  vector_store.py      From-scratch cosine-similarity vector index
  llm.py                Local GGUF model wrapper (llama.cpp)
  memory.py             Conversation memory with summarization
  reasoning.py          The Thought/Action/Observation agent loop
  knowledge_base.py     Ties ingestion → chunking → embedding → storage
  cli.py                Command-line interface
tests/                  19 passing tests covering everything except
                        actual model generation
docs/                   Setup guide, architecture writeup, project plan
data/knowledge_base/    Where the persisted vector index lives
models/                 Where you place the downloaded .gguf model file
```

## Status

Ingestion, chunking, the vector store, and the reasoning loop's
control logic are built and fully tested (`pytest tests/` — 19/19
passing). The CLI and knowledge base manager are wired up and tested
for the non-model-dependent paths (add/list/remove/persistence).

Not yet verified: actual model loading and generation quality — that
needs to run on your own hardware. See docs/setup.md step 6 for a
benchmark script to run once the model is downloaded.
