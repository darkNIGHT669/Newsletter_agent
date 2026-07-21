"""
Proves the checkpointer is real disk persistence, not just an in-memory
convenience -- by simulating an actual restart: a paused thread is created
under one SQLite connection and compiled graph object, that connection is
closed, and then a completely independent connection + freshly-built graph
object (standing in for "the app process restarted") is used to look the
same thread back up.

This deliberately bypasses agent.graph's cached get_checkpointer() singleton
(which conftest.py forces to an in-memory MemorySaver for the rest of the
test session) by passing explicit SqliteSaver instances into
build_graph(checkpointer=...).
"""

import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from agent.graph import build_graph
from agent.state import initial_state


def _sqlite_graph(db_path: str):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return build_graph(checkpointer=saver), conn


def test_hitl_interrupt_survives_a_simulated_restart(tmp_path, fake_llm):
    db_path = str(tmp_path / "test_checkpoints.sqlite")
    config = {"configurable": {"thread_id": "persist-test-thread"}}

    # "Process 1": pause on human_review, then go away.
    graph1, conn1 = _sqlite_graph(db_path)
    result = graph1.invoke(initial_state("Create a newsletter", "hitl"), config=config)
    assert "__interrupt__" in result
    conn1.close()

    # "Process 2": nothing shared with process 1 except the file on disk.
    graph2, conn2 = _sqlite_graph(db_path)
    snapshot = graph2.get_state(config)

    assert snapshot.next == ("human_review",)
    assert len(snapshot.tasks) == 1
    assert snapshot.tasks[0].interrupts[0].value["subject"] == "Test Subject"
    assert snapshot.values["draft_markdown"]  # full state came back too, not just the interrupt

    # And it should be genuinely resumable from here, not just readable.
    result2 = graph2.invoke(Command(resume={"decision": "approve"}), config=config)
    assert result2.get("final_path")
    conn2.close()


def test_two_different_threads_in_the_same_db_stay_isolated(tmp_path, fake_llm):
    db_path = str(tmp_path / "test_checkpoints.sqlite")
    graph, conn = _sqlite_graph(db_path)

    config_a = {"configurable": {"thread_id": "thread-a"}}
    config_b = {"configurable": {"thread_id": "thread-b"}}

    result_a = graph.invoke(initial_state("Goal A", "hitl"), config=config_a)
    result_b = graph.invoke(initial_state("Goal B", "hitl"), config=config_b)

    assert graph.get_state(config_a).values["goal"] == "Goal A"
    assert graph.get_state(config_b).values["goal"] == "Goal B"
    conn.close()
