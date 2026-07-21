"""
Tools available to the Newsletter Agent.

1. web_search_tool     - real search via Tavily if TAVILY_API_KEY is set,
                          otherwise falls back to a small offline mock corpus
                          so the whole graph is runnable with zero search keys.
2. html_newsletter_tool - turns a subject + list of summarized articles into
                          a clean, responsive HTML email.
3. save_email_tool      - "sends" the newsletter by saving it to disk and
                          printing the subject/body, per the assignment spec.
"""

import os
import json
import random
from datetime import datetime

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Offline fallback corpus used when no TAVILY_API_KEY is configured, so the
# agent is fully runnable/demoable without any search API key.
# ---------------------------------------------------------------------------
MOCK_ARTICLES = [
    {
        "title": "LangGraph adds native human-in-the-loop interrupts",
        "url": "https://example.com/langgraph-hitl",
        "snippet": "The LangGraph team shipped a first-class interrupt() primitive that lets any node pause a running graph and resume later with injected human input, simplifying approval workflows for agentic apps.",
    },
    {
        "title": "Multi-agent orchestration frameworks see rapid adoption",
        "url": "https://example.com/multi-agent-orchestration",
        "snippet": "Enterprises are increasingly wiring together planner, researcher, and critic agents into supervised graphs rather than single monolithic prompts, citing better reliability and easier debugging.",
    },
    {
        "title": "New benchmark measures agent self-correction ability",
        "url": "https://example.com/agent-self-correction-benchmark",
        "snippet": "A new open benchmark scores agents on whether a critique step actually improves output quality across successive revisions, rather than just producing longer responses.",
    },
    {
        "title": "Tool-use reliability remains the top blocker for production agents",
        "url": "https://example.com/tool-use-reliability",
        "snippet": "A survey of engineering teams building autonomous agents found malformed tool calls and silent tool failures as the leading cause of production incidents, ahead of hallucination.",
    },
    {
        "title": "Open-source agent frameworks converge on graph-based state machines",
        "url": "https://example.com/agent-frameworks-converge",
        "snippet": "Rather than pure chain-of-thought loops, most popular frameworks now model agents as explicit state graphs with typed transitions, making long-running workflows easier to checkpoint and resume.",
    },
    {
        "title": "Autonomous research agents shorten weekly reporting cycles",
        "url": "https://example.com/autonomous-research-agents",
        "snippet": "Teams piloting research-and-summarize agents for internal newsletters report cutting manual curation time from hours to minutes, while keeping a human approval gate before anything goes out.",
    },
    {
        "title": "Guardrails for agentic email sending gain traction",
        "url": "https://example.com/agentic-email-guardrails",
        "snippet": "As more agents gain the ability to draft and send communications, teams are standardizing on a simulate-then-approve pattern before any real send action is enabled.",
    },
    {
        "title": "Vector-memory agents show gains on long-horizon tasks",
        "url": "https://example.com/vector-memory-agents",
        "snippet": "Agents equipped with persistent vector memory outperformed stateless baselines on tasks spanning multiple sessions, particularly for personalization and long-running project tracking.",
    },
]


@tool
def web_search_tool(query: str) -> str:
    """Search the web for the latest AI agent news matching the given query.
    Returns a JSON string: a list of objects with 'title', 'url', 'snippet'.
    Uses Tavily if TAVILY_API_KEY is set in the environment, otherwise falls
    back to a small offline mock corpus so the agent stays fully runnable.
    """
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults

            search = TavilySearchResults(max_results=5, tavily_api_key=tavily_key)
            results = search.invoke({"query": query})
            normalized = [
                {
                    "title": r.get("title", "Untitled"),
                    "url": r.get("url", ""),
                    "snippet": (r.get("content", "") or "")[:500],
                }
                for r in results
            ]
            if normalized:
                return json.dumps(normalized)
        except Exception:
            # Fall through to mock data on any search/API failure so the
            # graph never hard-fails just because search is unavailable.
            pass

    sample_size = min(5, len(MOCK_ARTICLES))
    sample = random.sample(MOCK_ARTICLES, k=sample_size)
    return json.dumps(sample)


@tool
def html_newsletter_tool(subject: str, articles_json: str) -> str:
    """Generate a clean, responsive HTML newsletter document.
    'articles_json' is a JSON string: a list of objects with 'title', 'url',
    'summary'. Returns the full HTML document as a string.
    """
    articles = json.loads(articles_json)

    story_blocks = ""
    for a in articles:
        story_blocks += f"""
        <tr>
          <td style="padding:20px 0;border-bottom:1px solid #e5e7eb;">
            <h2 style="margin:0 0 8px 0;font-size:18px;color:#111827;font-family:Arial,sans-serif;">
              {a.get("title", "")}
            </h2>
            <p style="margin:0 0 10px 0;font-size:14px;line-height:1.5;color:#374151;font-family:Arial,sans-serif;">
              {a.get("summary", "")}
            </p>
            <a href="{a.get("url", "#")}" style="font-size:13px;color:#2563eb;text-decoration:none;font-family:Arial,sans-serif;">
              Read more &rarr;
            </a>
          </td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>{subject}</title></head>
<body style="margin:0;padding:0;background-color:#f3f4f6;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#f3f4f6;padding:24px 0;">
    <tr>
      <td align="center">
        <table width="600" cellpadding="0" cellspacing="0" style="background-color:#ffffff;border-radius:8px;overflow:hidden;">
          <tr>
            <td style="background-color:#111827;padding:24px 32px;">
              <h1 style="margin:0;font-size:20px;color:#ffffff;font-family:Arial,sans-serif;">
                {subject}
              </h1>
              <p style="margin:4px 0 0 0;font-size:12px;color:#9ca3af;font-family:Arial,sans-serif;">
                Your weekly briefing on AI agent news
              </p>
            </td>
          </tr>
          <tr>
            <td style="padding:0 32px;">
              <table width="100%" cellpadding="0" cellspacing="0">
                {story_blocks}
              </table>
            </td>
          </tr>
          <tr>
            <td style="padding:20px 32px;background-color:#f9fafb;">
              <p style="margin:0;font-size:11px;color:#9ca3af;font-family:Arial,sans-serif;">
                Sent by your Autonomous Newsletter Agent. You are receiving this
                because you subscribed to AI agent news updates.
              </p>
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>"""
    return html


@tool
def save_email_tool(subject: str, html_content: str, markdown_content: str) -> str:
    """Simulate sending the newsletter: saves the HTML and Markdown versions
    to the local 'outputs/' folder and prints the subject + body to the
    console. Returns the path to the saved HTML file.
    """
    out_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = os.path.join(out_dir, f"newsletter_{timestamp}.html")
    md_path = os.path.join(out_dir, f"newsletter_{timestamp}.md")

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    print("=" * 60)
    print(f"SUBJECT: {subject}")
    print("=" * 60)
    print(markdown_content)
    print("=" * 60)
    print(f"[SIMULATED SEND] Newsletter saved to: {html_path}")

    return html_path
