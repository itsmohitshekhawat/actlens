"""Measure search quality on the hand-written eval set (data/evalset.json, 50 questions with the correct articles).

Metrics, all at article/annex level:
- hit@k: share of questions where at least one correct article is in the top k results (often called recall@k).
- MRR: mean of 1/rank of the first correct result (1.0 = always first).

Usage: python scripts/eval_retrieval.py  -> prints a table and writes data/eval_retrieval.json (shown on /eval)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.search import DATA, get_index, ref_of  # noqa: E402

KS = (1, 3, 5, 10)


def evaluate(mode: str, questions: list[dict]) -> dict:
    ix = get_index()
    hits = {k: 0 for k in KS}
    rr, misses = 0.0, []
    start = time.perf_counter()
    for item in questions:
        refs = [ref_of(h) for h in ix.search(item["q"], k=max(KS), mode=mode)]
        rank = next((i + 1 for i, r in enumerate(refs) if r in item["gold"]), None)
        for k in KS:
            hits[k] += bool(rank and rank <= k)
        rr += 1 / rank if rank else 0
        if not rank or rank > 5:
            misses.append({"q": item["q"], "gold": item["gold"], "got": refs[:5]})
    n = len(questions)
    return {"mode": mode, **{f"hit@{k}": round(hits[k] / n, 3) for k in KS}, "mrr": round(rr / n, 3),
            "ms_per_query": round((time.perf_counter() - start) * 1000 / n, 2), "misses_at_5": misses}


def main() -> None:
    questions = json.loads((DATA / "evalset.json").read_text(encoding="utf-8"))
    results = [evaluate(m, questions) for m in ("keyword", "semantic", "hybrid")]
    print(f"{'mode':10}" + "".join(f"{'hit@' + str(k):>8}" for k in KS) + f"{'MRR':>8}{'ms':>8}")
    for r in results:
        print(f"{r['mode']:10}" + "".join(f"{r['hit@' + str(k)]:>8}" for k in KS) + f"{r['mrr']:>8}{r['ms_per_query']:>8}")
    out = DATA / "eval_retrieval.json"
    old = json.loads(out.read_text()) if out.exists() else {}
    out.write_text(json.dumps({**old, "questions": len(questions), "retrieval": results}, indent=1))


if __name__ == "__main__":
    main()
