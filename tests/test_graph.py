from langgraph.types import Command

from agent.graph import build_graph
from agent.state import initial_state


def test_autonomous_run_reaches_send_when_all_critics_score_high(fake_llm, fresh_config):
    graph = build_graph()
    result = graph.invoke(initial_state("Create a newsletter", "autonomous"), config=fresh_config)

    assert result["approved"] is True
    assert result["revision_count"] == 0
    assert result["final_path"]
    assert result["critique_score"] == 9.0


def test_one_weak_dimension_blocks_approval_despite_ok_average(fake_llm, fresh_config):
    """The approval bar is avg>=8 AND floor>=6, not a flat average. Two
    critics at 9 and one at 3 gives avg=7.0 (already fails on average, but
    even if two were at 10, a floor of 3 must still block approval)."""
    fake_llm["tone"] = 3
    graph = build_graph()
    result = graph.invoke(initial_state("Create a newsletter", "autonomous"), config=fresh_config)

    assert result["revision_count"] >= 1, "a genuinely weak dimension must trigger at least one revision"


def test_max_revisions_cap_prevents_an_infinite_loop(fake_llm, fresh_config):
    """If a critic never improves, the graph must still terminate rather
    than looping forever."""
    fake_llm["tone"] = 2
    graph = build_graph()
    result = graph.invoke(initial_state("Create a newsletter", "autonomous"), config=fresh_config)

    assert result["revision_count"] == 2  # MAX_REVISIONS
    assert result["final_path"], "must still send eventually rather than hang"


def test_hitl_pauses_with_full_critic_breakdown(fake_llm, fresh_config):
    graph = build_graph()
    result = graph.invoke(initial_state("Create a newsletter", "hitl"), config=fresh_config)

    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["subject"] == "Test Subject"
    assert payload["critique_breakdown"]["factual"]["score"] == 9
    assert payload["critique_breakdown"]["tone"]["score"] == 9
    assert payload["critique_breakdown"]["structure"]["score"] == 9


def test_hitl_re_reviews_after_every_revision_rather_than_auto_sending(fake_llm, fresh_config):
    """Regression test for a real bug caught during development: once a
    human had given feedback once, the graph would skip human_review on
    subsequent passes and auto-send as soon as the critic score recovered.
    Every revised draft must go back through a human, not just the first
    one."""
    fake_llm["tone"] = 3
    graph = build_graph()

    result = graph.invoke(initial_state("Create a newsletter", "hitl"), config=fresh_config)
    assert "__interrupt__" in result

    result2 = graph.invoke(
        Command(resume={"decision": "revise", "feedback": "Fix the tone"}), config=fresh_config
    )
    assert "__interrupt__" in result2, "the revised draft must pause for a fresh human review"

    result3 = graph.invoke(Command(resume={"decision": "approve"}), config=fresh_config)
    assert result3.get("final_path"), "must complete and send after explicit human approval"
