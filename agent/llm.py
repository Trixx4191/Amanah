"""
Local LLM Layer
----------------
Thin wrapper around llama-cpp-python, loading a local GGUF model file
and exposing a simple `generate()` call. This is the only module that
talks to the model directly — the reasoning loop and memory manager
just call `generate()` with a prompt and get text back.

No API keys, no network calls: the model runs entirely from a file on
disk (see docs/setup.md for how to download one).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LLMConfig:
    model_path: str
    n_ctx: int = 8192          # context window (tokens). 8k is a safe default
                                 # for a 7B model on CPU without huge RAM use.
    n_threads: int | None = None  # None = let llama.cpp auto-detect CPU cores
    temperature: float = 0.2    # lower = more deterministic/grounded,
                                 # good default for a document-QA agent
    max_tokens: int = 1024      # cap per generation call


class LocalLLM:
    def __init__(self, config: LLMConfig):
        from llama_cpp import Llama  # imported lazily so the rest of the
                                       # codebase can be imported/tested
                                       # without llama-cpp-python installed

        self.config = config
        self._llm = Llama(
            model_path=config.model_path,
            n_ctx=config.n_ctx,
            n_threads=config.n_threads,
            verbose=False,
        )

    def generate(self, prompt: str, stop: list[str] | None = None,
                 max_tokens: int | None = None) -> str:
        """Single-shot completion. `stop` lets the reasoning loop cut
        generation off at a delimiter like 'Observation:' so the model
        doesn't hallucinate its own tool results."""
        result = self._llm(
            prompt,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=self.config.temperature,
            stop=stop or [],
        )
        return result["choices"][0]["text"]

    def chat(self, messages: list[dict], stop: list[str] | None = None,
              max_tokens: int | None = None) -> str:
        """Chat-formatted completion using the model's built-in chat
        template (llama.cpp reads this from the GGUF file's metadata),
        so we don't have to hand-roll prompt formatting per model family."""
        result = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=self.config.temperature,
            stop=stop or [],
        )
        return result["choices"][0]["message"]["content"]
