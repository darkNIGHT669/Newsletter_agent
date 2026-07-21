"""
Node functions for the Newsletter Agent's LangGraph state graph.

Each node is a plain function: (state) -> partial state update dict.
This mirrors the assignment's required multi-step reasoning pipeline:

    planning -> research -> writing -> review (critique + optional human) -> output
"""

import json
import time
from langchain_core.messages import HumanMessage
from langgraph.types import interrupt

from .llm import get_llm
from .state import NewsletterState
from .tools import web_search_tool, html_newsletter_tool, save_email_tool

MAX_REVISIONS = 2


def _invoke_with_trace(llm, messages, node_name: str, trace: list):
    """Wraps a single llm.invoke() call, timing it and recording token usage
    when the provider reports it. usage_metadata is None for providers/models
    that don't populate it, in which case token counts are just omitted
    rather than guessed at.

    Appends one TraceRecord to the caller's local `trace` list (nodes that
    make several calls — e.g. summarize_and_draft_node summarizing each
    article — accumulate multiple entries before returning them all in one
    "trace" key, since state.trace uses the same additive reducer as logs).
    """
    start = time.perf_counter()
    response = llm.invoke(messages)
    duration_ms = (time.perf_counter() - start) * 1000

    # If the response content is a list of blocks (e.g. Gemini 3.5 output),
    # extract and concatenate the text elements to ensure it is a string.
    if isinstance(response.content, list):
        text_parts = []
        for block in response.content:
            if isinstance(block, dict):
                text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        response.content = "".join(text_parts)

    usage = getattr(response, "usage_metadata", None)
    trace.append(
        {
            "node": node_name,
            "duration_ms": round(duration_ms, 1),
            "input_tokens": usage.get("input_tokens") if usage else None,
            "output_tokens": usage.get("output_tokens") if usage else None,
        }
    )
    return response


def _safe_json(text: str, default: dict) -> dict:
    """LLMs sometimes wrap JSON in markdown fences or add stray text.
    This defensively extracts a JSON object from the raw response."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.split("json", 1)[-1] if cleaned.lower().startswith("json") else cleaned
    try:
        return json.loads(cleaned)
    except Exception:
        try:
            start = cleaned.index("{")
            end = cleaned.rindex("}") + 1
            return json.loads(cleaned[start:end])
        except Exception:
            return default


# ---------------------------------------------------------------------------
# 1. PLANNING
# ---------------------------------------------------------------------------
def planner_node(state: NewsletterState) -> dict:
    llm = get_llm(temperature=0.2)
    prompt = f"""You are the planning module of an autonomous Newsletter Agent.

Goal: "{state['goal']}"

Break this down into:
1. A one-paragraph plan describing the steps you will take.
2. Exactly 3 focused web search queries that will surface the latest,
   most relevant AI agent news.

Respond ONLY as compact JSON, no markdown fences, no preamble:
{{"plan": "...", "queries": ["q1", "q2", "q3"]}}"""

    trace = []
    response = _invoke_with_trace(llm, [HumanMessage(content=prompt)], "planner", trace)
    data = _safe_json(
        response.content,
        default={
            "plan": "Search for recent AI agent news, summarize the top stories, "
                     "draft a newsletter, critique it, and send it.",
            "queries": [
                "latest AI agent news",
                "AI agent frameworks update",
                "autonomous AI agents industry news",
            ],
        },
    )

    log = f"[PLANNER] {data['plan']} | queries={data['queries']}"
    return {
        "plan": data["plan"],
        "search_queries": data["queries"],
        "logs": [log],
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# 2. RESEARCH  (tool use #1: web_search_tool)
# ---------------------------------------------------------------------------
def research_node(state: NewsletterState) -> dict:
    all_articles = []
    seen_urls = set()

    for query in state["search_queries"]:
        raw = web_search_tool.invoke({"query": query})
        results = json.loads(raw)
        for r in results:
            url = r.get("url", "")
            if url and url in seen_urls:
                continue
            seen_urls.add(url)
            all_articles.append(
                {"title": r["title"], "url": url, "snippet": r.get("snippet", ""), "summary": None}
            )

    top_articles = all_articles[:7]
    if len(top_articles) < 5 and all_articles:
        top_articles = all_articles  # take whatever we found rather than under-deliver silently

    log = f"[RESEARCH] Collected {len(all_articles)} unique articles across {len(state['search_queries'])} queries; keeping top {len(top_articles)}."
    return {"articles": top_articles, "logs": [log]}


# ---------------------------------------------------------------------------
# 3. SUMMARIZATION + DRAFTING  (tool use #2: html_newsletter_tool)
# ---------------------------------------------------------------------------
def summarize_and_draft_node(state: NewsletterState) -> dict:
    llm = get_llm(temperature=0.4)

    feedback_note = ""
    if state.get("critique_feedback") and state["revision_count"] > 0:
        feedback_note += f"\nEditorial critique to address: {state['critique_feedback']}"
    if state.get("human_feedback"):
        feedback_note += f"\nHuman reviewer feedback to address: {state['human_feedback']}"

    trace = []
    summarized = []
    for article in state["articles"]:
        prompt = f"""Summarize this AI news item in 2-3 crisp sentences for a
technical newsletter audience. Be specific and avoid generic filler.

Title: {article['title']}
Snippet: {article['snippet']}
{feedback_note}

Respond with only the summary text, no preamble, no quotation marks."""
        resp = _invoke_with_trace(llm, [HumanMessage(content=prompt)], "draft:summarize", trace)
        summarized.append({**article, "summary": resp.content.strip()})

    subject_prompt = f"""Write one punchy, specific email subject line (under 65
characters) for a weekly AI agent newsletter covering these stories:
{[a['title'] for a in summarized]}
{feedback_note}

Respond with only the subject line, no quotation marks."""
    subject_resp = _invoke_with_trace(llm, [HumanMessage(content=subject_prompt)], "draft:subject", trace)
    subject = subject_resp.content.strip().strip('"')

    md_lines = [f"# {subject}", "", "_Your weekly briefing on AI agent news_", ""]
    for a in summarized:
        md_lines += [f"### {a['title']}", a["summary"], f"[Read more]({a['url']})", ""]
    draft_markdown = "\n".join(md_lines)

    draft_html = html_newsletter_tool.invoke(
        {"subject": subject, "articles_json": json.dumps(summarized)}
    )

    tag = " (revision)" if state["revision_count"] > 0 else ""
    log = f"[DRAFT{tag}] '{subject}' with {len(summarized)} stories."
    return {
        "articles": summarized,
        "subject": subject,
        "draft_markdown": draft_markdown,
        "draft_html": draft_html,
        "logs": [log],
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# 4. SELF-REFLECTION: multi-critic ensemble
#
# Instead of one LLM call judging everything at once (which tends to blur
# together "is this true" with "does this read well" and let one bad story
# hide behind decent prose), three specialists each look at ONE dimension.
# They run in parallel (fan-out from "draft") and are joined by
# critique_aggregate_node (fan-in) before routing decisions are made.
# ---------------------------------------------------------------------------
def _run_critic(state: NewsletterState, criteria_prompt: str, node_name: str):
    llm = get_llm(temperature=0.1)
    prompt = f"""{criteria_prompt}

Draft to review:
{state['draft_markdown']}

Respond ONLY as compact JSON, no markdown fences:
{{"score": <integer 0-10>, "feedback": "<one specific, actionable sentence, or 'Looks good' if score >= 8>"}}"""
    trace = []
    resp = _invoke_with_trace(llm, [HumanMessage(content=prompt)], node_name, trace)
    data = _safe_json(resp.content, default={"score": 7, "feedback": "Looks good"})
    return data, trace


def critique_factual_node(state: NewsletterState) -> dict:
    """Specialist #1: relevance & factual hygiene. Catches off-topic stories,
    duplicated stories, unsupported claims, and leftover placeholder text —
    the things a tone or formatting check would never notice."""
    data, trace = _run_critic(
        state,
        "You are a fact-checking editor. Judge ONLY relevance and factual "
        "hygiene: every story must genuinely be about AI agents (not generic "
        "AI), no two stories should cover the same news, no claim should read "
        "as invented or unsupported, and there must be no placeholder text "
        "(e.g. 'TODO', 'lorem ipsum', empty links).",
        "critique_factual",
    )
    log = f"[CRITIC:factual] score={data['score']}/10 — {data['feedback']}"
    return {"critique_factual": data, "logs": [log], "trace": trace}


def critique_tone_node(state: NewsletterState) -> dict:
    """Specialist #2: tone & clarity. Judges how it reads, independent of
    whether the facts are right or the formatting is consistent."""
    data, trace = _run_critic(
        state,
        "You are a copy editor. Judge ONLY tone and clarity: is the subject "
        "line specific and engaging (not generic), are summaries plain and "
        "non-redundant, is the register professional and consistent across "
        "stories, and is anything needlessly wordy or jargon-heavy for a "
        "reader who is technical but time-constrained?",
        "critique_tone",
    )
    log = f"[CRITIC:tone] score={data['score']}/10 — {data['feedback']}"
    return {"critique_tone": data, "logs": [log], "trace": trace}


def critique_structure_node(state: NewsletterState) -> dict:
    """Specialist #3: structure & formatting. Judges the shape of the
    document, independent of content quality."""
    data, trace = _run_critic(
        state,
        "You are a layout/format editor. Judge ONLY structure: are there "
        "5-7 distinct stories, does every story consistently have a title, "
        "summary, and link, is the Markdown well-formed with no broken "
        "syntax, and is each summary a reasonable length (not one line, not "
        "a wall of text)?",
        "critique_structure",
    )
    log = f"[CRITIC:structure] score={data['score']}/10 — {data['feedback']}"
    return {"critique_structure": data, "logs": [log], "trace": trace}


def critique_aggregate_node(state: NewsletterState) -> dict:
    """Fan-in node: merges the three specialist verdicts into the single
    critique_score / critique_feedback / approved fields the rest of the
    graph (human_review, revise, routing) consumes.

    Approval bar is deliberately NOT a flat average: average >= 8 AND the
    weakest individual score >= 6. A flat average lets one bad dimension
    (e.g. a factually shaky story) hide behind two lenient scores elsewhere;
    requiring a decent floor on every dimension models how a real editorial
    board works — one strong dissent blocks publication even if the other
    reviewers are happy.
    """
    factual = state["critique_factual"]
    tone = state["critique_tone"]
    structure = state["critique_structure"]

    scores = [factual["score"], tone["score"], structure["score"]]
    avg_score = round(sum(scores) / len(scores), 1)
    min_score = min(scores)
    approved = avg_score >= 8 and min_score >= 6

    issues = []
    for label, result in [
        ("Factual/Relevance", factual),
        ("Tone/Clarity", tone),
        ("Structure/Format", structure),
    ]:
        if result["score"] < 8:
            issues.append(f"{label} ({result['score']}/10): {result['feedback']}")
    combined_feedback = " | ".join(issues) if issues else "Looks good across all dimensions."

    log = (
        f"[CRITIQUE-AGGREGATE] avg={avg_score}/10 (weakest={min_score}) "
        f"approved={approved} — factual={factual['score']}, tone={tone['score']}, "
        f"structure={structure['score']}"
    )
    return {
        "critique_score": avg_score,
        "critique_feedback": combined_feedback,
        "approved": approved,
        "logs": [log],
    }


def revise_node(state: NewsletterState) -> dict:
    log = f"[REVISE] Sending draft back for revision (attempt {state['revision_count'] + 1})."
    return {
        "revision_count": state["revision_count"] + 1,
        # Reset the human decision so that, in HITL mode, the *revised* draft
        # is routed back through human_review for a fresh approval rather
        # than silently auto-sending on the next critique pass.
        "human_decision": None,
        "logs": [log],
    }


# ---------------------------------------------------------------------------
# 5. HUMAN-IN-THE-LOOP GATE
# ---------------------------------------------------------------------------
def human_review_node(state: NewsletterState) -> dict:
    """Pauses the graph using LangGraph's interrupt(). The value passed here
    is surfaced to the caller (see app.py) so it can render the draft and
    ask the human to approve or request a revision. When the graph is
    resumed with Command(resume=<decision dict>), `decision` below becomes
    that dict."""
    payload = {
        "subject": state["subject"],
        "draft_markdown": state["draft_markdown"],
        "critique_score": state["critique_score"],
        "critique_feedback": state["critique_feedback"],
        "critique_breakdown": {
            "factual": state["critique_factual"],
            "tone": state["critique_tone"],
            "structure": state["critique_structure"],
        },
    }
    decision = interrupt(payload)

    log = f"[HUMAN REVIEW] decision={decision.get('decision')}"
    return {
        "human_decision": decision.get("decision"),
        "human_feedback": decision.get("feedback", ""),
        "logs": [log],
    }


# ---------------------------------------------------------------------------
# 6. EXECUTION / SIMULATED SEND  (tool use #3: save_email_tool)
# ---------------------------------------------------------------------------
def send_node(state: NewsletterState) -> dict:
    path = save_email_tool.invoke(
        {
            "subject": state["subject"],
            "html_content": state["draft_html"],
            "markdown_content": state["draft_markdown"],
        }
    )
    log = f"[SEND] Newsletter simulated-sent and saved to {path}"
    return {"final_path": path, "logs": [log]}
