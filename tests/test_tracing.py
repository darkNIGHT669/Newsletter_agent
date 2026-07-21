from agent.graph import build_graph
from agent.state import initial_state


def test_trace_has_one_entry_per_llm_call(fake_llm, fresh_config):
    """planner: 1 call. draft: 1 call per article + 1 subject-line call.
    3 critics: 1 call each. With the mock search corpus returning up to 5
    articles per query across up to 3 queries (deduped), the exact article
    count varies, so we check the *shape* rather than an exact total."""
    graph = build_graph()
    result = graph.invoke(initial_state("Create a newsletter", "autonomous"), config=fresh_config)

    nodes_seen = {entry["node"] for entry in result["trace"]}
    assert "planner" in nodes_seen
    assert "draft:summarize" in nodes_seen
    assert "draft:subject" in nodes_seen
    assert "critique_factual" in nodes_seen
    assert "critique_tone" in nodes_seen
    assert "critique_structure" in nodes_seen

    # planner and draft:subject each fire exactly once per draft pass;
    # each critic fires exactly once per critique pass.
    counts = {n: sum(1 for e in result["trace"] if e["node"] == n) for n in nodes_seen}
    assert counts["planner"] == 1
    assert counts["critique_factual"] == 1
    assert counts["critique_tone"] == 1
    assert counts["critique_structure"] == 1


def test_trace_records_are_additive_across_the_parallel_critics(fake_llm, fresh_config):
    """The three critics run in the same superstep and must not clobber
    each other's trace entries -- this is the same operator.add concern
    that logs needed."""
    graph = build_graph()
    result = graph.invoke(initial_state("Create a newsletter", "autonomous"), config=fresh_config)

    critic_entries = [e for e in result["trace"] if e["node"].startswith("critique_")]
    assert len(critic_entries) == 3, "all three critics' trace entries must survive the fan-in"


def test_trace_entries_have_duration_and_token_usage(fake_llm, fresh_config):
    graph = build_graph()
    result = graph.invoke(initial_state("Create a newsletter", "autonomous"), config=fresh_config)

    for entry in result["trace"]:
        assert entry["duration_ms"] >= 0
        # The fake LLM fixture always sets usage_metadata, so these should
        # never be None here (a real provider without usage_metadata support
        # would leave them None -- see _invoke_with_trace in nodes.py).
        assert entry["input_tokens"] == 50
        assert entry["output_tokens"] == 20


def test_every_revision_pass_adds_new_trace_entries_not_a_replacement(fake_llm, fresh_config):
    """A critic that never improves (tone=3 forever) should hit the
    MAX_REVISIONS=2 cap: that's 2 *revisions*, i.e. 3 total draft passes
    (initial + 2 revisions) -- matching test_graph.py's
    test_max_revisions_cap_prevents_an_infinite_loop. Each pass must add a
    fresh trace entry, not overwrite the previous one."""
    fake_llm["tone"] = 3
    graph = build_graph()
    result = graph.invoke(initial_state("Create a newsletter", "autonomous"), config=fresh_config)

    assert result["revision_count"] == 2
    draft_entries = [e for e in result["trace"] if e["node"] == "draft:subject"]
    assert len(draft_entries) == 3, "initial draft + 2 revisions = 3 subject-line calls"
