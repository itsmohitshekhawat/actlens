"""ActLens web app (FastAPI).

  /                 the demo page (ask a question, watch the agent work, click citations)
  /eval             how well search and the agent perform, measured on 50 questions
  /api/ask?q=...    the agent, streamed as Server-Sent Events (one event per step)
  /api/search?q=... search only (free, no model call); mode=hybrid|keyword|semantic
  /api/eval         the saved evaluation results
  /api/examples     saved example answers (work even when the daily budget is used up)
  /mcp              the MCP server (tools for other AI apps)
  /health           for uptime monitoring
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import agent, budget
from .mcp_server import mcp
from .search import DATA, get_index

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
MAX_QUESTION = 400

mcp_app = mcp.http_app(path="/")
app = FastAPI(title="ActLens", lifespan=mcp_app.lifespan, docs_url=None, redoc_url=None)
app.mount("/mcp", mcp_app)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["X-Frame-Options"] = "DENY"
    return response


def client_ip(request: Request) -> str:
    # Behind a CDN proxy the real visitor address is in the CF-Connecting-IP header.
    return request.headers.get("cf-connecting-ip") or (request.client.host if request.client else "unknown")


@app.on_event("startup")
def warm() -> None:
    get_index()  # load the index once, so the first visitor does not wait


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/eval", include_in_schema=False)
def eval_page() -> FileResponse:
    return FileResponse(STATIC / "eval.html")


@app.get("/health")
def health() -> dict:
    return {"ok": True, "chunks": len(get_index().chunks)}


@app.get("/api/status")
def api_status(request: Request) -> dict:
    return budget.status(client_ip(request))


@app.get("/api/search")
def api_search(q: str = Query(..., min_length=2, max_length=MAX_QUESTION), mode: str = "hybrid", k: int = 8) -> dict:
    if mode not in ("hybrid", "keyword", "semantic"):
        raise HTTPException(400, "mode must be hybrid, keyword or semantic")
    hits = get_index().search(q, k=max(1, min(k, 20)), mode=mode)
    return {"query": q, "mode": mode, "results": [h.__dict__ for h in hits]}


@app.get("/api/ask")
def api_ask(request: Request, q: str = Query(..., min_length=3, max_length=MAX_QUESTION)) -> StreamingResponse:
    ip = client_ip(request)

    def stream():
        state = budget.status(ip)
        if not state["open"]:
            yield f"event: limit\ndata: {json.dumps(state)}\n\n"
            return
        cost = 0.0
        try:
            for event in agent.run(q.strip()):
                if event["type"] == "done":
                    cost = event["cost_usd"]
                yield f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:  # the model API can time out; tell the page instead of hanging
            yield f"event: failed\ndata: {json.dumps({'error': type(exc).__name__})}\n\n"
        finally:
            budget.record(ip, cost)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def read_json(name: str) -> dict | list:
    path = DATA / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@app.get("/api/eval")
def api_eval() -> JSONResponse:
    return JSONResponse({"retrieval": read_json("eval_retrieval.json"), "agent": read_json("eval_agent.json")})


@app.get("/api/examples")
def api_examples() -> JSONResponse:
    return JSONResponse(read_json("examples.json") or [])

