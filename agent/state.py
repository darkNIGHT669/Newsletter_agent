"""
State schema for the Newsletter Agent.

This is the single shared object that flows through every node in the
LangGraph state graph. Each node reads what it needs from it and returns a
partial dict of updates, which LangGraph merges back into the state.
"""

import operator
from typing import TypedDict, List, Dict, Optional, Literal, Annotated


class Article(TypedDict):
    title: str
    url: str
    snippet: str
    summary: Optional[str]


class CritiqueResult(TypedDict):
    score: int
    feedback: str


class TraceRecord(TypedDict):
    """One LLM call's cost/latency footprint, recorded by _invoke_with_trace
    in nodes.py. input_tokens/output_tokens are None when the provider
    doesn't report usage_metadata (this varies by provider/model)."""
    node: str
    duration_ms: float
    input_tokens: Optional[int]
    output_tokens: Optional[int]


class NewsletterState(TypedDict):
    # --- input ---
    goal: str
    mode: Literal["autonomous", "hitl"]

    # --- planning ---
    plan: str
    search_queries: List[str]

    # --- research ---
    articles: List[Article]

    # --- drafting ---
    subject: str
    draft_markdown: str
    draft_html: str

    # --- multi-critic self-reflection loop ---
    # Three specialists run in parallel (fan-out from "draft"), each scoring
    # a different quality dimension. critique_aggregate_node then fans them
    # back in and derives the two fields below that the rest of the graph
    # (human_review, revise, routing) actually consumes.
    critique_factual: Optional[CritiqueResult]
    critique_tone: Optional[CritiqueResult]
    critique_structure: Optional[CritiqueResult]
    critique_feedback: str
    critique_score: float
    approved: bool
    revision_count: int

    # --- human-in-the-loop ---
    human_decision: Optional[str]   # "approve" | "revise"
    human_feedback: Optional[str]

    # --- observability ---
    # Annotated with operator.add: when multiple nodes run in the same
    # superstep (the three parallel critics), each one's returned "logs"
    # list is *appended*, not overwritten. Without this, LangGraph raises
    # InvalidUpdateError because the default channel only accepts one write
    # per step. Every node must therefore return only its OWN new log
    # line(s) here, e.g. {"logs": [line]} — never the full accumulated list.
    logs: Annotated[List[str], operator.add]

    # Same additive-reducer reasoning as logs above: the three parallel
    # critics each append their own LLM call's trace record in the same
    # superstep, so this must be additive too, not last-write-wins.
    trace: Annotated[List[TraceRecord], operator.add]

    # --- execution / simulated send ---
    final_path: Optional[str]


def initial_state(goal: str, mode: str = "autonomous") -> NewsletterState:
    """Builds a fresh state dict for a new run."""
    return {
        "goal": goal,
        "mode": mode,
        "plan": "",
        "search_queries": [],
        "articles": [],
        "subject": "",
        "draft_markdown": "",
        "draft_html": "",
        "critique_factual": None,
        "critique_tone": None,
        "critique_structure": None,
        "critique_feedback": "",
        "critique_score": 0,
        "approved": False,
        "revision_count": 0,
        "human_decision": None,
        "human_feedback": "",
        "logs": [],
        "trace": [],
        "final_path": None,
    }
