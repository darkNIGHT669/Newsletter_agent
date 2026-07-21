import json
from pathlib import Path

from agent.tools import web_search_tool, html_newsletter_tool, save_email_tool


def test_web_search_falls_back_to_mock_without_tavily_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    raw = web_search_tool.invoke({"query": "latest AI agent news"})
    results = json.loads(raw)

    assert 1 <= len(results) <= 5
    for r in results:
        assert {"title", "url", "snippet"} <= r.keys()
        assert r["title"]  # non-empty


def test_html_newsletter_tool_embeds_subject_and_article_content():
    articles = [
        {"title": "Story One", "url": "https://example.com/1", "summary": "Summary one."},
        {"title": "Story Two", "url": "https://example.com/2", "summary": "Summary two."},
    ]
    html = html_newsletter_tool.invoke(
        {"subject": "My Weekly Subject", "articles_json": json.dumps(articles)}
    )

    assert "My Weekly Subject" in html
    assert "Story One" in html and "Story Two" in html
    assert "Summary one." in html and "Summary two." in html
    assert "https://example.com/1" in html
    assert html.strip().startswith("<!DOCTYPE html>")


def test_save_email_tool_writes_both_html_and_markdown(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # save_email_tool writes to ./outputs relative to cwd

    html_path = save_email_tool.invoke(
        {
            "subject": "Test Subject",
            "html_content": "<html><body>hi</body></html>",
            "markdown_content": "# hi",
        }
    )

    saved_html = Path(html_path)
    assert saved_html.exists()
    assert saved_html.read_text(encoding="utf-8") == "<html><body>hi</body></html>"

    saved_md = saved_html.with_suffix(".md")
    assert saved_md.exists()
    assert saved_md.read_text(encoding="utf-8") == "# hi"
