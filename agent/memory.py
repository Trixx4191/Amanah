"""
Memory Manager
--------------
Keeps conversation context across turns without blowing past the
model's context window.

Strategy:
- Keep the last `keep_recent_turns` turns verbatim (full fidelity for
  the immediate back-and-forth).
- Once older turns exceed that window, summarize them (using the same
  local model) into a running summary, so the thread isn't lost, just
  compressed.

This is deliberately simple — a single running summary rather than a
hierarchical or vector-indexed memory — because for a document-QA
agent, the *documents* are the long-term knowledge store; conversation
memory just needs to track what's been discussed so far in this session.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Turn:
    role: str  # "user" or "assistant"
    content: str


SUMMARIZE_PROMPT = """The following is a running summary of a conversation \
so far, followed by some new turns. Update the summary to incorporate the \
new turns. Keep it concise — a few sentences capturing what's been \
discussed and any conclusions reached. Do not lose important facts.

Existing summary:
{summary}

New turns:
{new_turns}

Updated summary:"""


class ConversationMemory:
    def __init__(self, keep_recent_turns: int = 6):
        self.keep_recent_turns = keep_recent_turns
        self.turns: list[Turn] = []
        self.running_summary: str = ""

    def add_turn(self, role: str, content: str) -> None:
        self.turns.append(Turn(role=role, content=content))

    def needs_summarization(self) -> bool:
        return len(self.turns) > self.keep_recent_turns * 2

    def summarize_older_turns(self, llm) -> None:
        """Call this periodically (e.g. after each turn) — it's a no-op
        unless there's enough history to warrant compressing."""
        if not self.needs_summarization():
            return

        cutoff = len(self.turns) - self.keep_recent_turns
        older = self.turns[:cutoff]
        self.turns = self.turns[cutoff:]

        new_turns_text = "\n".join(f"{t.role}: {t.content}" for t in older)
        prompt = SUMMARIZE_PROMPT.format(
            summary=self.running_summary or "(none yet)",
            new_turns=new_turns_text,
        )
        self.running_summary = llm.generate(prompt, max_tokens=300).strip()

    def as_context_string(self) -> str:
        """What gets prepended to the prompt: summary of older turns +
        the recent turns verbatim."""
        parts = []
        if self.running_summary:
            parts.append(f"[Summary of earlier conversation]\n{self.running_summary}")
        for t in self.turns:
            parts.append(f"{t.role}: {t.content}")
        return "\n\n".join(parts)
