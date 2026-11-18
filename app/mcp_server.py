"""ActLens as an MCP server (Model Context Protocol), at /mcp on the same website.

MCP is a standard way to give AI apps (Claude, Cursor, VS Code, ChatGPT connectors...) new tools. Anyone can add
https://actlens.mohitshekhawat.com/mcp to their AI app, and their assistant can then search and read the AI Act itself.

The search and read tools are free (no model call on our side). ask_actlens runs our full agent, so it counts against
the same daily budget as the website.
"""

from __future__ import annotations

from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_request

from . import agent, budget
from .search import get_index

mcp = FastMCP(
    "ActLens",
    instructions="Search and read the official text of the EU AI Act (Regulation (EU) 2024/1689). "
                 "Cite the labels you get back, such as Art. 5(1) or Annex III.",
)


@mcp.tool
def search_ai_act(query: str, k: int = 6) -> list[dict]:
    """Hybrid keyword + semantic search over the EU AI Act. Returns passages with citation labels."""
    hits = get_index().search(query, k=max(1, min(k, 15)))
    return [{"label": h.label, "title": h.title, "kind": h.kind, "text": h.text} for h in hits]


@mcp.tool
def read_ai_act(ref: str) -> list[dict]:
    """Full text of an article, paragraph, annex or recital, e.g. 'Art. 5', 'Art. 99(3)', 'Annex III', 'Recital 27'."""
    return [{"label": c["label"], "title": c.get("title", ""), "text": c["text"]} for c in get_index().article(ref)]


@mcp.tool
def ask_actlens(question: str) -> dict:
    """Ask the ActLens agent a question; it searches the Act itself and answers with verified citations."""
    try:
        request = get_http_request()
        ip = request.headers.get("cf-connecting-ip") or (request.client.host if request.client else "mcp")
    except RuntimeError:
        ip = "mcp-local"
    if not budget.status(ip)["open"]:
        return {"error": "Daily demo limit reached. search_ai_act and read_ai_act still work."}
    result = agent.answer(question[:400])
    budget.record(ip, result.get("cost_usd", 0))
    return {k: result[k] for k in ("answer", "check", "steps", "cost_usd")}
