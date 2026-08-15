"""
Reasoning Agent Loop
---------------------
This is what makes the system an *agent* rather than a single-shot RAG
chatbot: the model explicitly reasons in steps, decides when it needs
to search the knowledge base, and can search more than once before
answering.

Loop structure (ReAct-style):
    Thought: <model reasons about what it needs>
    Action: search_documents["<query>"]
    Observation: <injected by us, from the vector store>
    ... repeats as needed ...
    Final Answer: <model's grounded answer>

Confidence handling: every Observation includes the similarity scores
of what was retrieved. The system prompt instructs the model to treat
low-scoring retrievals as "not well supported" and say so explicitly
rather than guessing — this is enforced by instruction, and we also do
a programmatic check (see LOW_CONFIDENCE_THRESHOLD) as a backstop.

Citations: when enabled, we append source filename/page for the chunks
that were actually retrieved and used, rather than trusting the model
to accurately self-report citations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from agent.embeddings import embed_query
from agent.vector_store import SearchResult, VectorStore

MAX_ITERATIONS = 4
LOW_CONFIDENCE_THRESHOLD = 0.35  # cosine similarity below this = weak grounding
TOP_K = 4

SYSTEM_PROMPT = """You are a local, private research assistant with expert \
knowledge drawn from documents the user has provided. You reason step by \
step and only use the search_documents tool to look things up in those \
documents — you do not have any other source of truth.

You must follow this exact format, one step at a time:

Thought: <your reasoning about what to do next>
Action: search_documents["<search query>"]

When you have enough information, instead write:

Thought: <your reasoning>
Final Answer: <your answer to the user>

Rules:
- Only use information found via search_documents. Do not use outside knowledge \
to fill gaps.
- If the search results don't actually support an answer, say so plainly in \
your Final Answer (e.g. "The documents don't appear to cover this") rather \
than guessing.
- Search more than once if the first results aren't sufficient — try a \
different phrasing.
- Keep Thoughts brief.
"""

_ACTION_RE = re.compile(r'Action:\s*search_documents\["?(.+?)"?\]\s*$', re.MULTILINE)
_FINAL_RE = re.compile(r'Final Answer:\s*(.+)', re.DOTALL)


@dataclass
class ReasoningStep:
    thought: str
    action_query: str | None
    observation: str | None


@dataclass
class AgentResponse:
    answer: str
    steps: list[ReasoningStep] = field(default_factory=list)
    sources_used: list[SearchResult] = field(default_factory=list)
    low_confidence: bool = False


def _format_observation(results: list[SearchResult]) -> tuple[str, bool]:
    if not results:
        return "No relevant results found.", True

    best_score = max(r.score for r in results)
    low_conf = best_score < LOW_CONFIDENCE_THRESHOLD

    lines = []
    for r in results:
        loc = f"{r.source_filename}" + (f", page {r.page_number}" if r.page_number else "")
        lines.append(f"[{loc}, relevance={r.score:.2f}] {r.text}")
    obs = "\n".join(lines)
    if low_conf:
        obs += "\n(Note: relevance scores are low — this may not actually answer the query.)"
    return obs, low_conf


def run_agent(query: str, llm, store: VectorStore,
              conversation_context: str = "",
              cite_sources: bool = False) -> AgentResponse:
    """Runs the Thought/Action/Observation loop until the model produces
    a Final Answer or we hit MAX_ITERATIONS."""

    prompt = SYSTEM_PROMPT
    if conversation_context:
        prompt += f"\n\nConversation so far:\n{conversation_context}\n"
    prompt += f"\n\nUser question: {query}\n\nThought:"

    steps: list[ReasoningStep] = []
    all_sources: list[SearchResult] = []
    any_low_confidence = False

    for _ in range(MAX_ITERATIONS):
        completion = llm.generate(
            prompt,
            stop=["Observation:", "\nUser question:"],
            max_tokens=400,
        )

        final_match = _FINAL_RE.search(completion)
        if final_match:
            answer = final_match.group(1).strip()
            thought = completion[:final_match.start()].replace("Thought:", "").strip()
            steps.append(ReasoningStep(thought=thought, action_query=None, observation=None))
            return _finalize(answer, steps, all_sources, any_low_confidence, cite_sources)

        action_match = _ACTION_RE.search(completion)
        if not action_match:
            # Model didn't follow the format — treat whatever it wrote as
            # the answer rather than looping forever or crashing.
            answer = completion.strip()
            steps.append(ReasoningStep(thought=completion.strip(), action_query=None, observation=None))
            return _finalize(answer, steps, all_sources, any_low_confidence, cite_sources)

        query_text = action_match.group(1)
        thought_text = completion[:action_match.start()].replace("Thought:", "").strip()

        query_vec = embed_query(query_text)
        results = store.search(query_vec, top_k=TOP_K)
        observation, low_conf = _format_observation(results)
        any_low_confidence = any_low_confidence or low_conf
        all_sources.extend(results)

        steps.append(ReasoningStep(thought=thought_text, action_query=query_text, observation=observation))

        prompt += (
            f" {thought_text}\nAction: search_documents[\"{query_text}\"]\n"
            f"Observation: {observation}\nThought:"
        )

    # Hit max iterations without a Final Answer — ask the model to wrap up
    # using whatever it has gathered so far.
    prompt += (
        "\nYou must now give your best answer based on what you've found so far. "
        "Final Answer:"
    )
    completion = llm.generate(prompt, max_tokens=400)
    return _finalize(completion.strip(), steps, all_sources, any_low_confidence, cite_sources)


def _finalize(answer: str, steps: list[ReasoningStep], sources: list[SearchResult],
              low_confidence: bool, cite_sources: bool) -> AgentResponse:
    if cite_sources and sources:
        # de-duplicate by (filename, page), preserve order of first appearance
        seen = set()
        citations = []
        for s in sources:
            key = (s.source_filename, s.page_number)
            if key not in seen:
                seen.add(key)
                loc = s.source_filename + (f" (p.{s.page_number})" if s.page_number else "")
                citations.append(loc)
        if citations:
            answer += "\n\nSources: " + "; ".join(citations)

    return AgentResponse(
        answer=answer,
        steps=steps,
        sources_used=sources,
        low_confidence=low_confidence,
    )
