"""
App-level integration tests using Streamlit's official headless AppTest
harness -- these click real buttons and read real rendered output from an
actual (simulated) Streamlit runtime, rather than testing the graph in
isolation. Slower than tests/test_graph.py, but this is the layer that
catches wiring bugs the graph tests structurally can't see.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def _run_autonomous(at):
    at.run()
    at.sidebar.radio[0].set_value("autonomous")
    at.sidebar.button[0].click().run()  # "Run Agent"


def test_autonomous_run_streams_to_completion(fake_llm):
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    _run_autonomous(at)

    assert not at.exception
    assert any("Test Subject" in s.value for s in at.subheader)
    assert any("Sent (simulated)" in s.value for s in at.success)

    # Cost/latency trace should have rendered with real per-call data, not
    # just avoided crashing.
    assert any("Cost & latency trace" in e.label for e in at.expander)
    assert len(at.dataframe) >= 1
    trace_df = at.dataframe[0].value
    assert list(trace_df.columns) == ["Node", "Duration (ms)", "Input tokens", "Output tokens"]
    assert len(trace_df) >= 5  # planner + draft:subject + 3 critics at minimum
    assert (trace_df["Input tokens"] == 50).all()


def test_fresh_sessions_get_distinct_thread_ids(fake_llm):
    """Regression test for a real bug found during development: every fresh
    session defaulted to the literal thread_id 'session-1', so two
    different users opening the app for the first time would silently
    share (and could see each other's drafts in) the same checkpoint
    thread."""
    at_1 = AppTest.from_file(APP_PATH, default_timeout=60)
    at_1.run()
    at_2 = AppTest.from_file(APP_PATH, default_timeout=60)
    at_2.run()

    assert at_1.session_state["thread_id"] != at_2.session_state["thread_id"]


def test_hitl_pause_then_approve(fake_llm):
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value("hitl")
    at.sidebar.button[0].click().run()

    assert not at.exception
    assert any("paused for Human-in-the-Loop" in w.value for w in at.warning)

    approve_button = next(b for b in at.button if "Approve" in (b.label or ""))
    approve_button.click().run()

    assert not at.exception
    assert any("Sent (simulated)" in s.value for s in at.success)


def test_resume_by_thread_id_after_simulated_restart(fake_llm):
    # Session 1: pause mid-HITL, note the thread id, then "walk away".
    at1 = AppTest.from_file(APP_PATH, default_timeout=60)
    at1.run()
    at1.sidebar.radio[0].set_value("hitl")
    at1.sidebar.button[0].click().run()
    thread_id = at1.session_state["thread_id"]
    assert at1.session_state["pending_interrupt"] is not None

    # Session 2: a totally separate AppTest instance -- stands in for a
    # fresh browser tab / restarted app. It must start with no memory of
    # session 1 at all.
    at2 = AppTest.from_file(APP_PATH, default_timeout=60)
    at2.run()
    assert at2.session_state["thread_id"] != thread_id
    assert at2.session_state["pending_interrupt"] is None

    # Resume by pasting in the old thread id.
    resume_input = next(t for t in at2.sidebar.text_input if t.key == "resume_id_input")
    resume_input.set_value(thread_id)
    load_button = next(b for b in at2.sidebar.button if "Load" in (b.label or ""))
    load_button.click().run()

    assert not at2.exception
    assert at2.session_state["thread_id"] == thread_id
    assert at2.session_state["pending_interrupt"]["subject"] == "Test Subject"

    # And it should be genuinely resumable, not just viewable.
    approve_button = next(b for b in at2.button if "Approve" in (b.label or ""))
    approve_button.click().run()
    assert not at2.exception
    assert any("Sent (simulated)" in s.value for s in at2.success)
