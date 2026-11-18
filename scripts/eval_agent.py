"""Evaluate the full agent on the 50-question eval set (costs about $0.03 per run with DeepSeek V4.1 Flash).

For each question we record:
- cited_gold: did the answer cite at least one of the correct articles/annexes?
- grounded:   were ALL citations verified (pointing to passages the agent actually retrieved)?
- steps, seconds, tokens and cost.
Compare: "one-shot search" = is a correct article in the top 6 of a single search with the raw question (what plain
RAG would put in the prompt). The agent should beat it, because it can search again with better words.

Usage: LLM_API_KEY=... python scripts/eval_agent.py  -> writes data/eval_agent.json
"""

from __future__ import annotations

import json
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import agent  # noqa: E402
from app.search import DATA, get_index, ref_of  # noqa: E402


def ref_of_citation(c: str) -> str:
    base = agent.citation_base(c)  # 'Art. 5' / 'Annex III' / 'Recital 4'
    return base.replace("Art. ", "Art ")


def one(item: dict) -> dict:
    try:
        r = agent.answer(item["q"])
    except Exception as exc:  # count API failures instead of crashing the whole run
        return {"q": item["q"], "error": type(exc).__name__}
    cited = {ref_of_citation(c["citation"]) for c in r["check"].get("citations", [])}
    one_shot = {ref_of(h) for h in get_index().search(item["q"], k=6)}
    return {"q": item["q"], "gold": item["gold"], "cited": sorted(cited),
            "cited_gold": bool(cited & set(item["gold"])), "one_shot_hit": bool(one_shot & set(item["gold"])),
            "grounded": r["check"].get("grounded", False), "citations": r["check"].get("total", 0),
            "steps": r["steps"], "seconds": r["seconds"], "cost_usd": r["cost_usd"],
            "tokens": r["tokens_in"] + r["tokens_out"]}


def main() -> None:
    questions = json.loads((DATA / "evalset.json").read_text(encoding="utf-8"))
    get_index()
    with ThreadPoolExecutor(max_workers=5) as pool:
        rows = list(pool.map(one, questions))
    ok = [r for r in rows if "error" not in r]
    n = len(ok)
    summary = {
        "questions": len(rows), "errors": len(rows) - n,
        "agent_cites_correct_article": round(sum(r["cited_gold"] for r in ok) / n, 3),
        "one_shot_search_top6": round(sum(r["one_shot_hit"] for r in ok) / n, 3),
        "all_citations_verified": round(sum(r["grounded"] for r in ok) / n, 3),
        "avg_citations": round(statistics.mean(r["citations"] for r in ok), 1),
        "avg_steps": round(statistics.mean(r["steps"] for r in ok), 2),
        "median_seconds": round(statistics.median(r["seconds"] for r in ok), 1),
        "avg_cost_usd": round(statistics.mean(r["cost_usd"] for r in ok), 6),
        "total_cost_usd": round(sum(r["cost_usd"] for r in ok), 4),
        "avg_tokens": round(statistics.mean(r["tokens"] for r in ok)),
        "model": agent.llm.MODEL,
    }
    (DATA / "eval_agent.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False))
    for k, v in summary.items():
        print(f"{k:32} {v}")
    for r in ok:
        if not r["cited_gold"]:
            print("MISS", r["q"], r["gold"], r["cited"])


if __name__ == "__main__":
    main()
