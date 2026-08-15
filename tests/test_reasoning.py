import numpy as np
import pytest

from agent.chunking import Chunk
from agent.reasoning import run_agent
from agent.vector_store import VectorStore


class FakeLLM:
    """Returns scripted completions in sequence, so we can test the
    reasoning loop's control flow without a real model."""
    def __init__(self, scripted_responses):
        self.responses = list(scripted_responses)
        self.calls = []

    def generate(self, prompt, stop=None, max_tokens=None):
        self.calls.append(prompt)
        if not self.responses:
            raise AssertionError("FakeLLM ran out of scripted responses")
        return self.responses.pop(0)


@pytest.fixture
def store_with_data():
    store = VectorStore()
    base = np.random.default_rng(0).standard_normal(384).astype(np.float32)
    base /= np.linalg.norm(base)
    chunks = [Chunk(chunk_id="d1-0", doc_id="d1", source_filename="manual.pdf",
                     page_number=3, chunk_index=0, text="The widget requires 4 AA batteries.")]
    store.add(chunks, base.reshape(1, -1))
    return store, base


def test_single_search_then_final_answer(store_with_data, monkeypatch):
    store, base = store_with_data
    # make embed_query return a vector close to our stored one, regardless of text
    monkeypatch.setattr("agent.reasoning.embed_query", lambda q: base)

    llm = FakeLLM([
        ' I need to look up battery requirements.\n'
        'Action: search_documents["battery requirements"]\n',
        ' The documents confirm it.\nFinal Answer: The widget needs 4 AA batteries.',
    ])

    response = run_agent("How many batteries does the widget need?", llm, store)

    assert "4 AA batteries" in response.answer
    assert len(response.steps) == 2
    assert response.steps[0].action_query == "battery requirements"
    assert len(response.sources_used) == 1
    assert response.low_confidence is False  # score should be high (same vector)


def test_citation_appended_when_enabled(store_with_data, monkeypatch):
    store, base = store_with_data
    monkeypatch.setattr("agent.reasoning.embed_query", lambda q: base)

    llm = FakeLLM([
        ' Looking it up.\nAction: search_documents["batteries"]\n',
        ' Found it.\nFinal Answer: It needs 4 AA batteries.',
    ])

    response = run_agent("battery question", llm, store, cite_sources=True)
    assert "Sources:" in response.answer
    assert "manual.pdf" in response.answer
    assert "p.3" in response.answer


def test_no_citation_when_disabled(store_with_data, monkeypatch):
    store, base = store_with_data
    monkeypatch.setattr("agent.reasoning.embed_query", lambda q: base)

    llm = FakeLLM([
        ' Looking it up.\nAction: search_documents["batteries"]\n',
        ' Found it.\nFinal Answer: It needs 4 AA batteries.',
    ])

    response = run_agent("battery question", llm, store, cite_sources=False)
    assert "Sources:" not in response.answer


def test_low_confidence_flagged_on_empty_store(monkeypatch):
    empty_store = VectorStore()
    monkeypatch.setattr("agent.reasoning.embed_query", lambda q: np.zeros(384, dtype=np.float32))
    llm = FakeLLM([
        ' Need to search.\nAction: search_documents["anything"]\n',
        ' Nothing found.\nFinal Answer: The documents do not appear to cover this.',
    ])
    response = run_agent("some obscure question", llm, empty_store)
    assert response.low_confidence is True
    assert "do not appear to cover" in response.answer


def test_model_skipping_format_still_returns_answer(store_with_data):
    store, _ = store_with_data
    # Model ignores the Thought/Action format entirely and just answers.
    llm = FakeLLM(["I think the answer is 42."])
    response = run_agent("what is the answer", llm, store)
    assert "42" in response.answer
    assert response.steps[0].action_query is None


def test_max_iterations_forces_wrap_up(store_with_data, monkeypatch):
    store, base = store_with_data
    monkeypatch.setattr("agent.reasoning.embed_query", lambda q: base)

    # Model keeps searching forever without ever giving a Final Answer
    infinite_search = ' Still looking.\nAction: search_documents["x"]\n'
    responses = [infinite_search] * 4 + ["Best guess: 4 AA batteries."]
    llm = FakeLLM(responses)

    response = run_agent("battery question", llm, store)
    assert "4 AA batteries" in response.answer
    assert len(llm.calls) == 5  # 4 loop iterations + 1 forced wrap-up call
