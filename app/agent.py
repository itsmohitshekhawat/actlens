"""The ActLens agent: an agentic RAG loop with tool calling and a citation check.

Plain RAG = search once, stuff the results into the prompt, answer. That fails when the question needs two lookups
("is X high-risk, and what must I do then?") or when the first search uses the wrong words.
Agentic RAG = the model decides what to search, reads what it found, searches again if needed, and only then answers.

Loop (at most MAX_STEPS model calls):
  1. The model gets the question + two tools: search_law(query) and read(ref).
  2. It calls tools; we run them and send the results back.
  3. When it stops calling tools, its message is the answer, with citations like [Art. 5(1)].
  4. Citation check (no extra model call): every citation must point to a passage the agent actually saw.
     Citations it never retrieved are flagged as unverified, which catches made-up article numbers.

`run()` yields events (step, tool call, tool result, answer, check) so the web page can show the trace live.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator

from . import llm
from .search import get_index

MAX_STEPS = 6
SNIPPET_CHARS = 700
READ_CHARS = 7000

SYSTEM = """You are ActLens, an assistant that answers questions about the EU Artificial Intelligence Act \
(Regulation (EU) 2024/1689) using ONLY the official text you retrieve with your tools.

How to work:
- Always search before answering. Use several short searches with different wording if the first results do not
  contain the answer. Use read(ref) to see a whole article or annex when a snippet is not enough.
- Prefer articles and annexes (the binding law) over recitals (explanations).
- Answer in plain English, at most about 180 words, for a smart reader who is not a lawyer. Start with a direct
  answer (yes / no / it depends) when the question allows it.
- Put a citation in square brackets after every claim, using the exact labels from the tool results, for example
  [Art. 5(1)], [Annex III], [Recital 27]. Never cite anything you did not retrieve.
- If the retrieved text does not answer the question, say that the AI Act text you found does not cover it.
- Do not give legal advice; you explain what the text says."""

TOOLS = [
    {"type": "function", "function": {
        "name": "search_law",
        "description": "Hybrid keyword + semantic search over the AI Act (articles, annexes, recitals). Returns the best passages with citation labels.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Short search query, e.g. 'emotion recognition workplace'"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "read",
        "description": "Read the full text of an article, annex or recital, e.g. 'Art. 5', 'Art. 99(3)', 'Annex III', 'Recital 27'.",
        "parameters": {"type": "object", "properties": {"ref": {"type": "string"}}, "required": ["ref"]}}},
]

CITATION = re.compile(r"\[((?:Art\.|Article|Annex|Recital)[^\]]{0,40})\]")


def tool_search(query: str, seen: dict[str, str]) -> tuple[str, list[dict]]:
    hits = get_index().search(query, k=6)
    rows = []
    for h in hits:
        seen[h.label] = h.text
        rows.append({"label": h.label, "title": h.title, "text": h.text[:SNIPPET_CHARS]})
    return json.dumps(rows, ensure_ascii=False), [{"label": r["label"], "title": r["title"]} for r in rows]


def tool_read(ref: str, seen: dict[str, str]) -> tuple[str, list[dict]]:
    chunks = get_index().article(ref)
    if not chunks:
        return json.dumps({"error": f"nothing found for {ref!r}"}), []
    text, used = "", []
    for c in chunks:
        if len(text) > READ_CHARS:
            break
        seen[c["label"]] = c["text"]
        text += f"[{c['label']}] {c['text']}\n"
        used.append({"label": c["label"], "title": c.get("title", "")})
    return text[:READ_CHARS], used


def citation_base(label: str) -> str:
    """'Art. 5(1)(f)' -> 'Art. 5'; 'Annex III (part 2)' -> 'Annex III'; 'Article 50' -> 'Art. 50'."""
    label = label.strip().replace("Article", "Art.")
    m = re.match(r"(Art\.)\s*(\d+)", label)
    if m:
        return f"Art. {m.group(2)}"
    m = re.match(r"(Annex [IVXLC]+|Recital \d+)", label)
    return m.group(1) if m else label


def best_source(citation: str, seen: dict[str, str]) -> str | None:
    """The retrieved passage that best matches a citation: 'Art. 5(1)(f)' -> 'Art. 5(1)' rather than 'Art. 5(8)'."""
    cite = citation.strip().replace("Article", "Art.").replace(" ", "")
    exact = [lbl for lbl in seen if cite.startswith(lbl.replace(" ", "").split("(part")[0])]
    if exact:
        return max(exact, key=len)
    base = citation_base(citation)
    return next((lbl for lbl in seen if citation_base(lbl) == base), None)


def check_citations(answer: str, seen: dict[str, str]) -> dict:
    cited = []
    for raw in CITATION.findall(answer):
        for part in re.split(r";\s*", raw):  # "[Art. 5(1); Art. 99(3)]"
            if part and part not in cited:
                cited.append(part.strip())
    seen_bases = {citation_base(s) for s in seen}
    results = [{"citation": c, "verified": citation_base(c) in seen_bases} for c in cited]
    ok = sum(r["verified"] for r in results)
    return {"citations": results, "verified": ok, "total": len(results),
            "grounded": bool(results) and ok == len(results)}


def run(question: str) -> Iterator[dict]:
    start = time.perf_counter()
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    seen: dict[str, str] = {}
    cost = tokens_in = tokens_out = 0
    answer = ""
    for step in range(1, MAX_STEPS + 1):
        last = step == MAX_STEPS
        msg, usage = llm.chat(messages, tools=None if last else TOOLS)
        cost += usage["cost"]
        tokens_in += usage["in"]
        tokens_out += usage["out"]
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            answer = msg["content"].strip()
            break
        if msg["content"].strip():
            yield {"type": "thought", "step": step, "text": msg["content"].strip()}
        for call in calls:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            yield {"type": "tool_call", "step": step, "tool": name, "args": args}
            if name == "search_law":
                result, found = tool_search(str(args.get("query", question)), seen)
            elif name == "read":
                result, found = tool_read(str(args.get("ref", "")), seen)
            else:
                result, found = json.dumps({"error": f"unknown tool {name}"}), []
            yield {"type": "tool_result", "step": step, "tool": name, "found": found}
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
    if not answer:
        answer = "Sorry, I could not finish the answer in the allowed number of steps."
    check = check_citations(answer, seen)
    sources = {}
    for c in check["citations"]:
        match = best_source(c["citation"], seen)
        if match:
            sources[c["citation"]] = {"label": match, "text": seen[match][:1200]}
    yield {"type": "answer", "text": answer, "check": check, "sources": sources}
    yield {"type": "done", "steps": step, "seconds": round(time.perf_counter() - start, 2),
           "tokens_in": tokens_in, "tokens_out": tokens_out, "cost_usd": round(cost, 6)}


def answer(question: str) -> dict:
    """Run the agent to the end and return everything (used by the MCP server, the eval script and tests)."""
    trace, final, done = [], {}, {}
    for event in run(question):
        if event["type"] == "answer":
            final = event
        elif event["type"] == "done":
            done = event
        else:
            trace.append(event)
    return {"question": question, "answer": final.get("text", ""), "check": final.get("check", {}),
            "sources": final.get("sources", {}), "trace": trace, **done}
