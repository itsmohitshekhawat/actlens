# ActLens

**An AI agent that answers questions about the EU AI Act from the official text, cites every claim, and checks its own citations.**

Live demo: **https://actlens.mohitshekhawat.com** · Evaluation: [/eval](https://actlens.mohitshekhawat.com/eval) · MCP server: `https://actlens.mohitshekhawat.com/mcp`

Plain RAG searches once and hopes the right passage is in the results. ActLens is *agentic*: the model decides what to search, reads whole articles when a snippet is not enough, searches again with new words, and only then answers. A deterministic checker then matches every citation against the passages the agent actually retrieved.

## Results (50 hand-written questions, `data/evalset.json`)

| | Result |
|---|---|
| Agent cites a correct article | **100%** (50/50) |
| One search with the raw question (plain RAG) has a correct article in its top 6 | 78% |
| Answers where every citation is verified | 98% (the checker flagged one citation the agent never read) |
| Average cost / median time per answer | $0.0008 / 6 s (DeepSeek V4.1 Flash, 3.6 model calls) |

Search alone (no model), article-level:

| Method | hit@1 | hit@5 | MRR |
|---|---|---|---|
| Keyword (BM25) | 0.56 | **0.84** | 0.655 |
| Meaning (static embeddings) | 0.54 | 0.82 | 0.655 |
| Hybrid (RRF, recitals ×0.7) | **0.66** | 0.80 | **0.712** |

Hybrid puts the right article first most often, which matters most for an agent that searches several times. The recital weight was tuned on the same 50 questions, so treat these numbers as optimistic.

## How it works

```
official XHTML ──► ingest.py ──► 800 passages (article paragraphs, single definitions, annex parts, recitals)
                                     │
                     build_index.py  ├─► BM25 keyword index
                                     └─► Model2Vec static embeddings (30 MB, no PyTorch)
question ──► agent loop (DeepSeek, tool calling, max 6 rounds)
               ├─ search_law(query)  ─► hybrid search, Reciprocal Rank Fusion
               └─ read(ref)          ─► full article / annex / recital
           ──► answer with [Art. 5(1)] citations ──► citation check ──► streamed to the page (SSE)
```

- **Data:** Regulation (EU) 2024/1689 from the EU Publications Office API, split by legal structure so every passage has an exact citation label.
- **Search:** `bm25s` + `model2vec`, merged with RRF (`score = Σ 1/(60 + rank)`). Under 1 ms per query on a CPU.
- **Agent:** OpenAI-compatible tool calling (`app/agent.py`). Every step is streamed to the browser as Server-Sent Events, so you watch it work.
- **Citation check:** no extra model call. A citation counts as verified only if the agent retrieved that article, annex or recital.
- **MCP:** the same tools are exposed through `fastmcp` at `/mcp` (`search_ai_act`, `read_ai_act`, `ask_actlens`).
- **Cost control:** a daily USD budget and a per-visitor daily limit in SQLite (only hashed IPs are stored). Search and the eval page stay free when the budget is used up.
- **Footprint:** about 150 MB RAM in Docker, so it runs on a small VPS.

## Run it

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
python scripts/build_index.py              # builds data/index/
cp .env.example .env                       # add LLM_API_KEY
export $(grep -v '^#' .env | xargs)
uvicorn app.main:app --reload              # http://localhost:8000
pytest -q                                  # tests, no API key needed
python scripts/eval_retrieval.py           # search metrics
python scripts/eval_agent.py               # agent metrics (about $0.04 per run)
```

Or with Docker: `docker build -t actlens . && docker run -p 8000:8000 -e LLM_API_KEY=... actlens`.

Use it from an MCP client:

```json
{ "mcpServers": { "actlens": { "type": "http", "url": "https://actlens.mohitshekhawat.com/mcp" } } }
```

## Project layout

| Path | What it does |
|---|---|
| `scripts/ingest.py` | Official XHTML → `data/corpus.json` (800 labelled passages) |
| `scripts/build_index.py` | BM25 + embedding index |
| `app/search.py` | Hybrid search, RRF, reference lookup |
| `app/agent.py` | Agent loop, tools, citation checker |
| `app/llm.py` | Model client (any OpenAI-compatible API), cost tracking |
| `app/budget.py` | Daily budget + per-visitor limits |
| `app/mcp_server.py` | MCP tools |
| `app/main.py` | FastAPI routes, SSE streaming |
| `scripts/eval_*.py`, `data/evalset.json` | Evaluation |

## Limits and next steps

- The eval set is small (50 questions) and was written by the author. A larger, independently written set would be fairer.
- Next: a cross-encoder reranker; guidelines and harmonised standards as extra sources; an LLM judge for faithfulness, checked against human labels.

ActLens explains the text; it is not legal advice. Law text © European Union, reused under Commission Decision 2011/833/EU. Code: MIT licence.
