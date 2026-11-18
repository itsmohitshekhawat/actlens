"""Build the search index once, offline: BM25 (keyword) + embeddings (meaning). The web app only loads these files.

Usage: python scripts/build_index.py   (reads data/corpus.json, writes data/index/)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bm25s
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.search import EMBED_MODEL, INDEX, DATA, doc_text, tokenize  # noqa: E402


def main() -> None:
    from model2vec import StaticModel

    chunks = json.loads((DATA / "corpus.json").read_text(encoding="utf-8"))["chunks"]
    texts = [doc_text(c) for c in chunks]
    INDEX.mkdir(parents=True, exist_ok=True)

    retriever = bm25s.BM25()
    retriever.index(tokenize(texts), show_progress=False)
    retriever.save(str(INDEX / "bm25"))

    model = StaticModel.from_pretrained(EMBED_MODEL)
    model.save_pretrained(str(INDEX / "model"))  # shipped with the app, so it never downloads at runtime
    vectors = model.encode(texts)
    vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)
    np.save(INDEX / "embeddings.npy", vectors.astype(np.float32))
    print(f"indexed {len(chunks)} chunks: bm25 + embeddings {vectors.shape}")


if __name__ == "__main__":
    main()
