"""
Graph assembly for the Newsletter Agent.

State machine:

    START -> planner -> research -> draft -> [critique_factual, critique_tone, critique_structure]
                                                              |  (fan-in)
                                                              v
                                                     critique_aggregate
                                                              |
                     +----------------------------------------+----------------------------------+
                     |                                        |                                  |
              (hitl & no decision)                  (approved OR max revisions)       (needs another
                     v                                        v                        revision pass)
              human_review                                  send                              v
                     |                                        |                             revise -> draft (loop)
          +----------+----------+                             v
          |                     |                            END
     (approve)             (revise)
          v                     v
        send                 revise -> draft (loop)

draft fans out to three parallel specialist critics (factual/relevance,
tone/clarity, structure/format) which critique_aggregate then fans back in
and merges into a single approve/revise verdict — see nodes.py for the
approval-bar rationale (average AND floor, not a flat average).

Exposes a single high-level entry point, run_newsletter_agent(goal, mode),
as required by the assignment. For interactive Human-in-the-Loop use, drive
the compiled graph directly (see app.py) so a real human can respond to the
interrupt instead of it being auto-approved.
"""

import os
import sqlite3

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from .state import NewsletterState, initial_state
from .nodes import (
    planner_node,
    research_node,
    summarize_and_draft_node,
    critique_factual_node,
    critique_tone_node,
    critique_structure_node,
    critique_aggregate_node,
    human_review_node,
    revise_node,
    send_node,
    MAX_REVISIONS,
)


_checkpointer = None
_checkpointer_conn = None  # kept alive for the lifetime of the process


def get_checkpointer():
    """Builds (once) the checkpointer that gives the graph memory across
    HITL interrupts.

    Reads CHECKPOINT_DB_PATH from the environment:
      - unset / any file path (default "checkpoints.sqlite"): a real SQLite
        file, so a paused Human-in-the-Loop run survives an app restart —
        you can close the app, come back later, and resume with the same
        thread_id.
      - ":memory:": an ephemeral in-memory MemorySaver, useful for tests/CI
        where a leftover checkpoint file would be unwanted noise.

    SqliteSaver.from_conn_string() is a @contextmanager in the underlying
    library (it opens a connection, yields it, then closes it on exit) —
    using it with `with` here would close our connection the moment
    graph-building finished, which is wrong for a long-lived Streamlit
    process. So instead we open the sqlite3.Connection ourselves
    (check_same_thread=False, since Streamlit may touch it from more than
    one thread across reruns) and keep it open for the process lifetime.
    This is exactly what from_conn_string does internally — SqliteSaver(conn)
    is a public, supported constructor, not a workaround.
    """
    global _checkpointer, _checkpointer_conn
    if _checkpointer is not None:
        return _checkpointer

    db_path = os.getenv("CHECKPOINT_DB_PATH", "checkpoints.sqlite")

    if db_path == ":memory:":
        _checkpointer = MemorySaver()
        return _checkpointer

    _checkpointer_conn = sqlite3.connect(db_path, check_same_thread=False)
    saver = SqliteSaver(_checkpointer_conn)
    saver.setup()  # idempotent; creates tables on first run only
    _checkpointer = saver
    return _checkpointer


def route_after_critique(state: NewsletterState) -> str:
    if state["mode"] == "hitl" and state.get("human_decision") is None:
        return "human_review"
    if state["approved"] or state["revision_count"] >= MAX_REVISIONS:
        return "send"
    return "revise"


def route_after_human_review(state: NewsletterState) -> str:
    if state.get("human_decision") == "approve":
        return "send"
    return "revise"


def build_graph(checkpointer=None):
    """Builds and compiles the graph. If no checkpointer is given, uses the
    process-wide cached one from get_checkpointer(). Accepting an explicit
    checkpointer (rather than always reaching for the global) is what makes
    it possible to test real SQLite persistence across two independently
    constructed connections — see tests/test_persistence.py — without
    fighting the singleton that a long-lived app process wants.
    """
    graph = StateGraph(NewsletterState)

    graph.add_node("planner", planner_node)
    graph.add_node("research", research_node)
    graph.add_node("draft", summarize_and_draft_node)
    graph.add_node("critique_factual", critique_factual_node)
    graph.add_node("critique_tone", critique_tone_node)
    graph.add_node("critique_structure", critique_structure_node)
    graph.add_node("critique_aggregate", critique_aggregate_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("revise", revise_node)
    graph.add_node("send", send_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "research")
    graph.add_edge("research", "draft")

    # Fan-out: all three specialists review the same draft in parallel.
    graph.add_edge("draft", "critique_factual")
    graph.add_edge("draft", "critique_tone")
    graph.add_edge("draft", "critique_structure")

    # Fan-in: aggregator waits for all three before running.
    graph.add_edge("critique_factual", "critique_aggregate")
    graph.add_edge("critique_tone", "critique_aggregate")
    graph.add_edge("critique_structure", "critique_aggregate")

    graph.add_conditional_edges(
        "critique_aggregate",
        route_after_critique,
        {"human_review": "human_review", "send": "send", "revise": "revise"},
    )
    graph.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {"send": "send", "revise": "revise"},
    )

    graph.add_edge("revise", "draft")
    graph.add_edge("send", END)

    checkpointer = checkpointer if checkpointer is not None else get_checkpointer()
    return graph.compile(checkpointer=checkpointer)


_compiled_graph = None


def get_graph():
    """Lazily builds and caches the compiled graph (one per process)."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def load_thread_state(thread_id: str):
    """Looks up a thread directly from the checkpointer, independent of any
    in-memory session — this is what lets the UI's "resume a paused
    session" feature work even after a full app restart, since the
    checkpoint lives in checkpoints.sqlite, not in Streamlit's session
    state.

    Returns (values, interrupt_payload):
      - values: the full state dict for that thread, or None if the
        thread_id has no saved checkpoint at all.
      - interrupt_payload: the pending human_review payload if the thread
        is currently paused waiting on a decision, else None (either the
        run finished, or it never got that far).
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    if not snapshot or not snapshot.values:
        return None, None

    interrupt_payload = None
    for task in snapshot.tasks:
        if task.interrupts:
            interrupt_payload = task.interrupts[0].value
            break

    return snapshot.values, interrupt_payload


def run_newsletter_agent(goal: str, mode: str = "autonomous", thread_id: str = "default-thread") -> NewsletterState:
    """Single entry point required by the assignment: one call runs the
    full agent end-to-end.

    NOTE on HITL: LangGraph's interrupt() genuinely pauses execution and
    hands control back to the caller. Because this function has no live
    human attached, if mode='hitl' it will auto-resume the interrupt with
    an 'approve' decision so the call still completes end-to-end. For a
    real interactive human-in-the-loop experience, use the Streamlit app
    (app.py), which pauses on the interrupt and waits for a real button
    click before resuming.
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    result = graph.invoke(initial_state(goal, mode), config=config)

    if "__interrupt__" in result:
        from langgraph.types import Command

        result = graph.invoke(Command(resume={"decision": "approve"}), config=config)

    return result
