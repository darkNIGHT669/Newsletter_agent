"""
Shared pytest fixtures.

Setting CHECKPOINT_DB_PATH here, at module import time (before any `agent.*`
module is imported by a test file), ensures the process-wide checkpointer
singleton in agent/graph.py resolves to an in-memory MemorySaver for the
whole test session -- fast, and leaves no stray .sqlite file behind.

tests/test_persistence.py deliberately bypasses this: it builds its own
SqliteSaver instances against real temp files via build_graph(checkpointer=...)
to prove actual disk persistence, which is a different concern from "does
the state machine's logic work" (covered by test_graph.py) and doesn't need
a shared global to do it.
"""

import os

os.environ.setdefault("CHECKPOINT_DB_PATH", ":memory:")

import json
import uuid
from unittest.mock import MagicMock

import pytest

import agent.llm as llm_module
import agent.nodes as nodes_module


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    """save_email_tool writes to ./outputs relative to the current working
    directory. Without this, every test that runs the graph to completion
    (including the Streamlit AppTest-based ones, which don't sandbox file
    I/O on their own) would leave real generated newsletter files behind in
    the actual project's outputs/ folder on every test run."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def fake_llm(monkeypatch):
    """Provides a controllable stand-in for get_llm() so tests never need
    real API keys or make real network calls, and can deterministically
    force specific critic verdicts.

    Returns the mutable `scores` dict -- tests write to it (e.g.
    `fake_llm["tone"] = 3`) *before* invoking the graph to steer a
    specific critic's verdict for that test.
    """
    scores = {"factual": 9, "tone": 9, "structure": 9}
    feedback = {"factual": "Looks good", "tone": "Looks good", "structure": "Looks good"}

    def _factory(temperature=0.3):
        model = MagicMock()

        def _invoke(messages):
            text = messages[0].content
            resp = MagicMock()
            # Real AIMessage.usage_metadata is either a dict-like UsageMetadata
            # or None -- explicitly set it (rather than leaving it as an
            # auto-generated MagicMock attribute) so tracing code that calls
            # usage.get(...) behaves the same as it would against a real
            # provider response.
            resp.usage_metadata = {"input_tokens": 50, "output_tokens": 20, "total_tokens": 70}
            if "queries" in text and "plan" in text:
                resp.content = json.dumps({"plan": "Test plan", "queries": ["q1", "q2"]})
            elif "fact-checking editor" in text:
                resp.content = json.dumps({"score": scores["factual"], "feedback": feedback["factual"]})
            elif "copy editor" in text:
                resp.content = json.dumps({"score": scores["tone"], "feedback": feedback["tone"]})
            elif "layout/format editor" in text:
                resp.content = json.dumps({"score": scores["structure"], "feedback": feedback["structure"]})
            elif "subject line" in text:
                resp.content = "Test Subject"
            else:
                resp.content = "Test summary sentence about AI agents."
            return resp

        model.invoke.side_effect = _invoke
        return model

    monkeypatch.setattr(llm_module, "get_llm", _factory)
    monkeypatch.setattr(nodes_module, "get_llm", _factory)
    return scores


@pytest.fixture
def fresh_config():
    """A config dict with a random thread_id, so tests never collide with
    each other even though they share one process-wide cached graph."""
    return {"configurable": {"thread_id": str(uuid.uuid4())}}
