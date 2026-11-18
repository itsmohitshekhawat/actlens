"""Hybrid search over the AI Act: keyword search (BM25) + meaning search (static embeddings), merged with
Reciprocal Rank Fusion (RRF).

Why both: BM25 is great at exact words ("Article 99", "biometric"), embeddings are great at meaning ("fines" finds
"administrative penalties"). RRF merges the two ranked lists without needing their scores to be comparable:
score(doc) = sum over lists of 1 / (k + rank).

Everything here runs on a small CPU: BM25 is a sparse matrix, and the embedding model is a static Model2Vec model
(about 30 MB, no neural network at query time, just a lookup and an average), so a search takes a few milliseconds.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import bm25s
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
INDEX = DATA / "index"
EMBED_MODEL = "minishlab/potion-base-8M"
RRF_K = 60
# Recitals explain the law but are not the law itself, so they rank a bit lower than articles and annexes.
KIND_WEIGHT = {"article": 1.0, "annex": 1.0, "recital": 0.7}


@dataclass
class Hit:
    id: str
    label: str
    title: str
    text: str
    score: float
    kind: str


def doc_text(chunk: dict) -> str:
    """What we index: the citation label and titles help both search methods."""
    return " ".join(x for x in (chunk["label"], chunk.get("title", ""), chunk.get("section", ""), chunk["text"]) if x)


def tokenize(texts: list[str]):
    return bm25s.tokenize(texts, stopwords="en", show_progress=False)


def load_embedder():
    from model2vec import StaticModel

    local = INDEX / "model"
    return StaticModel.from_pretrained(str(local) if local.exists() else EMBED_MODEL)


class SearchIndex:
    def __init__(self) -> None:
        corpus = json.loads((DATA / "corpus.json").read_text(encoding="utf-8"))
        self.source = corpus["source"]
        self.chunks: list[dict] = corpus["chunks"]
        self.by_id = {c["id"]: c for c in self.chunks}
        self.bm25 = bm25s.BM25.load(str(INDEX / "bm25"))
        self.vectors = np.load(INDEX / "embeddings.npy")
        self.embedder = load_embedder()

    def keyword(self, query: str, k: int) -> list[tuple[int, float]]:
        docs, scores = self.bm25.retrieve(tokenize([query]), k=min(k, len(self.chunks)), show_progress=False)
        return [(int(d), float(s)) for d, s in zip(docs[0], scores[0]) if s > 0]

    def semantic(self, query: str, k: int) -> list[tuple[int, float]]:
        q = self.embedder.encode([query])[0]
        q = q / (np.linalg.norm(q) + 1e-9)
        sims = self.vectors @ q
        top = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in top]

    def search(self, query: str, k: int = 6, mode: str = "hybrid", kinds: list[str] | None = None,
               max_per_ref: int = 2) -> list[Hit]:
        pool = max(k * 5, 30)
        lists = []
        if mode in ("hybrid", "keyword"):
            lists.append(self.keyword(query, pool))
        if mode in ("hybrid", "semantic"):
            lists.append(self.semantic(query, pool))
        fused: dict[int, float] = {}
        for ranked in lists:
            for rank, (doc, _) in enumerate(ranked):
                fused[doc] = fused.get(doc, 0.0) + 1.0 / (RRF_K + rank + 1)
        # A query that names an article directly ("Article 50") gets that article's chunks first.
        for number in re.findall(r"\b(?:art(?:icle)?\.?\s*)(\d{1,3})\b", query, flags=re.I):
            for i, c in enumerate(self.chunks):
                if c.get("article") == int(number):
                    fused[i] = fused.get(i, 0.0) + 1.0
        for doc in fused:
            fused[doc] *= KIND_WEIGHT[self.chunks[doc]["kind"]]
        order = sorted(fused, key=fused.get, reverse=True)
        hits, per_ref = [], {}
        for i in order:
            c = self.chunks[i]
            if kinds and c["kind"] not in kinds:
                continue
            # Diversity: at most max_per_ref chunks from one article/annex, so one long article can't fill every slot.
            ref = ref_of(c)
            if max_per_ref and per_ref.get(ref, 0) >= max_per_ref:
                continue
            per_ref[ref] = per_ref.get(ref, 0) + 1
            hits.append(Hit(c["id"], c["label"], c.get("title", ""), c["text"], round(fused[i], 4), c["kind"]))
            if len(hits) == k:
                break
        return hits

    def article(self, ref: str) -> list[dict]:
        """All chunks for a reference like 'Art. 5', '5', 'Annex III', 'Recital 27' or a chunk id."""
        ref = ref.strip()
        if ref in self.by_id:
            return [self.by_id[ref]]
        m = re.fullmatch(r"(?:art(?:icle)?\.?\s*)?(\d{1,3})(?:\s*\((\d+)\))?", ref, flags=re.I)
        if m:
            number, para = int(m.group(1)), m.group(2)
            out = [c for c in self.chunks if c.get("article") == number]
            if para:
                out = [c for c in out if c.get("paragraph") == int(para)] or out
            return out
        m = re.fullmatch(r"annex\s+([IVXLC]+)", ref, flags=re.I)
        if m:
            return [c for c in self.chunks if c.get("annex") == m.group(1).upper()]
        m = re.fullmatch(r"recital\s+(\d+)", ref, flags=re.I)
        if m:
            return [c for c in self.chunks if c.get("recital") == int(m.group(1))]
        return []


@lru_cache(maxsize=1)
def get_index() -> SearchIndex:
    return SearchIndex()


def ref_of(chunk_or_hit) -> str:
    """Article/annex-level reference used for evaluation: 'Art 5', 'Annex III' or 'Recital 27'."""
    label = chunk_or_hit["label"] if isinstance(chunk_or_hit, dict) else chunk_or_hit.label
    m = re.match(r"Art\. (\d+)", label)
    if m:
        return f"Art {m.group(1)}"
    m = re.match(r"(Annex [IVXLC]+)", label)
    return m.group(1) if m else label
