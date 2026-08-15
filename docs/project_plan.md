# Local Document-Learning Agent — Project Plan

## 1. What we're building

A fully local AI agent that:
- Ingests PDFs, Word docs, and text files the user uploads
- Builds a persistent, shared knowledge base from everything it's been given
- Answers questions and performs tasks grounded in that knowledge
- **Reasons** through multi-step problems (not single-shot pattern completion)
- Remembers conversation context across turns
- Runs entirely on a CPU-only machine — no GPU, no API keys, no cloud calls, no remote servers

Everything except the base language model weights and a couple of well-established data-structure libraries (numpy, etc.) is built by us, so every piece is understood and explainable — this is designed to be something you can walk through confidently in an interview.

## 2. Key decisions made

| Decision | Choice | Why |
|---|---|---|
| Hardware target | CPU only | No dedicated GPU available |
| "From scratch" scope | Build the full pipeline ourselves (ingestion, chunking, embedding index, retrieval, reasoning loop, memory) on top of an open-weight local model — not implementing/training a transformer from zero | Realistic, still deeply educational, produces something that actually works well |
| Knowledge base structure | Single shared knowledge base across all uploaded documents | Simpler mental model; the agent can draw connections across sources |
| Model size | ~7B parameters, quantized | Prioritizes reasoning quality over raw speed |
| Interface | CLI first, then a local web UI | Get the core loop working and testable before investing in UI |
| Source citation | Toggleable on/off | Sometimes you want clean prose, sometimes you want to verify against the source |
| Document removal | Required from day one | Needed to keep the knowledge base manageable and correct |
| Unsupported-answer handling | Agent must flag low-confidence / not-in-docs answers rather than guess | Trust and correctness matter more than always sounding confident |

## 3. System architecture

```
                     ┌─────────────────────┐
   Upload PDF/DOCX/  │  1. Ingestion Layer  │
   TXT/MD files  ───▶│  (parsers per type)  │
                     └──────────┬──────────┘
                                ▼
                     ┌─────────────────────┐
                     │  2. Chunking Engine  │
                     │ (semantic splitting) │
                     └──────────┬──────────┘
                                ▼
                     ┌─────────────────────┐
                     │  3. Embedding Model  │
                     │  (local, CPU-based)  │
                     └──────────┬──────────┘
                                ▼
                     ┌─────────────────────┐
                     │ 4. Vector Store      │
                     │ (local index on disk)│
                     └──────────┬──────────┘
                                ▲
                                │ retrieval
        User query ──▶ ┌────────────────────────┐
                        │ 5. Reasoning Agent Loop │
                        │  (Thought/Action/Obs)   │
                        │  tools: search_docs,    │
                        │  (later: calculator,    │
                        │   summarizer, etc.)     │
                        └───────────┬─────────────┘
                                    ▼
                        ┌────────────────────────┐
                        │ 6. Local LLM (7B, GGUF) │
                        │   via llama.cpp         │
                        └───────────┬─────────────┘
                                    ▼
                        ┌────────────────────────┐
                        │ 7. Memory Manager       │
                        │ (rolling buffer +       │
                        │  summarization)         │
                        └────────────────────────┘
                                    ▼
                              Response to user
```

## 4. Component breakdown

### 4.1 Ingestion Layer
- PDF parsing via `pymupdf` (fast, handles text + layout reasonably well on CPU)
- DOCX parsing via `python-docx`
- Plain text / Markdown read natively
- Normalizes everything to `(text, source_metadata)` pairs (filename, page number, section if available)

### 4.2 Chunking Engine (built by us)
- Recursive, paragraph/sentence-aware splitting — avoids cutting mid-sentence
- Configurable chunk size + overlap (overlap preserves context across chunk boundaries)
- Each chunk tagged with source metadata (filename, page/section) — this feeds the optional citation toggle in responses, and lets a whole document be found and removed by its source ID

### 4.3 Embedding Model
- A small local sentence-embedding model (`all-MiniLM-L6-v2` or similar), runs fine on CPU
- Converts each chunk — and each incoming query — into a fixed-length vector

### 4.4 Vector Store (built by us)
- v1: from-scratch flat index using numpy — store vectors + metadata, cosine similarity search
- v2 (optional upgrade once v1 works): swap in FAISS for speed at scale, same interface
- Persisted to disk so the knowledge base survives restarts
- Supports deletion by source document — removes all chunks tied to that file's ID, so a document can be "forgotten" cleanly

### 4.5 Reasoning Agent Loop (the core "smarts")
- Implements a ReAct-style loop: model produces a **Thought**, decides on an **Action** (e.g., "search the knowledge base for X"), receives an **Observation** (retrieved chunks), and repeats until it's ready to give a final answer
- This is what separates "agent" from "chatbot with a search bolted on" — it can decompose a question, search multiple times, and synthesize
- Starts with one tool (`search_documents`); designed so more tools can be added later (e.g. a calculator, a summarizer)
- Before finalizing an answer, the loop checks whether retrieved chunks actually support it; if retrieval comes back weak/irrelevant, the agent says so explicitly rather than guessing
- When citation mode is on, the final answer includes the source filename/page for each claim pulled from the knowledge base

### 4.6 Local LLM
- 7B-parameter instruction-tuned model, GGUF format, quantized (Q4_K_M as a starting point — good quality/speed balance)
- Served via `llama-cpp-python`, fully local inference
- Exact model to be finalized once we benchmark on your machine (Qwen2.5-7B-Instruct is the current leading candidate)

### 4.7 Memory Manager
- Rolling window of recent conversation turns kept in full
- Older turns periodically summarized (by the same local model) to stay within context limits without losing the thread

### 4.8 Interfaces
- **Phase A:** CLI — upload a file via a command, chat in the terminal, see the agent's reasoning trace (great for debugging and for interview demos)
- **Phase B:** Local web UI (browser-based, served locally, no external calls) — file upload widget, chat window, and a view into the reasoning steps

## 5. Tech stack summary

| Layer | Tool | Notes |
|---|---|---|
| LLM inference | `llama-cpp-python` | Runs GGUF models on CPU |
| Model | Qwen2.5-7B-Instruct (GGUF, Q4_K_M) | To be confirmed after benchmarking |
| Embeddings | `sentence-transformers` (MiniLM) | CPU-friendly |
| PDF parsing | `pymupdf` | |
| DOCX parsing | `python-docx` | |
| Vector index | numpy (v1) → FAISS (v2, optional) | Both CPU-only |
| Orchestration / agent loop | Custom Python, no LangChain/LlamaIndex | Keeps everything transparent and "ours" |
| CLI | Python `argparse`/`rich` for nice terminal output | |
| Web UI (phase B) | Local Flask/FastAPI server + simple frontend | Still 100% local, just served on localhost |

## 6. Build roadmap

1. **Environment setup** — install deps, pull a quantized model, confirm local inference works and measure baseline tokens/sec on your machine
2. **Ingestion + chunking** — parse a real PDF, inspect chunk quality
3. **Embeddings + vector store v1** — embed chunks, build the numpy index, verify retrieval returns sensible results for test queries
4. **Basic RAG loop** — query → retrieve → stuff into prompt → get grounded answer (no agentic reasoning yet — this is the "walking" milestone before "running")
5. **Reasoning agent loop** — upgrade to Thought/Action/Observation with the search tool, test on multi-step questions
6. **Memory manager** — add conversation buffer + summarization, test long conversations
7. **CLI polish** — clean commands for uploading docs, removing/forgetting a document, chatting, inspecting the knowledge base, and toggling citations
8. **Web UI** — local browser interface on top of the same backend
9. **Documentation pass** — one `.md` per component in `/docs`, plus an architecture overview

Each step produces something runnable and testable before moving to the next — no big-bang integration at the end.

## 7. Open items to revisit as we build
- Exact model choice, confirmed after a speed benchmark on your hardware
- Chunk size/overlap tuning, based on real document tests
- Confidence threshold for when the agent declares "not enough support in the docs" (will tune empirically once retrieval is working)
