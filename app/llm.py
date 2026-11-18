"""The language model: DeepSeek V4.1 Flash through an OpenAI-compatible API.

The OpenAI SDK speaks the same chat + tool-calling format, so only the base URL, key and model name change.
Every call reports its token use and cost, so the app can show cost per question and enforce a daily budget.
"""

from __future__ import annotations

import os

from openai import OpenAI

MODEL = os.environ.get("LLM_MODEL", "deepseek-v4.1-flash")
BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1")
# Rough prices (USD per 1M tokens) used only when the API does not report the cost itself.
PRICE_IN, PRICE_OUT = 0.15, 0.60

_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        key = os.environ.get("LLM_API_KEY")
        if not key:
            raise RuntimeError("LLM_API_KEY is not set")
        _client = OpenAI(api_key=key, base_url=BASE_URL, timeout=60, max_retries=2)
    return _client


def chat(messages: list[dict], tools: list[dict] | None = None, max_tokens: int = 1200) -> tuple[dict, dict]:
    """One model call. Returns (assistant message as a dict, usage with tokens + cost in USD)."""
    kwargs = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.2}
    if tools:
        kwargs["tools"] = tools
    # Some OpenAI-compatible gateways report the exact cost when asked; others ignore this field.
    resp = client().chat.completions.create(**kwargs, extra_body={"usage": {"include": True}})
    msg = resp.choices[0].message
    out = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        out["tool_calls"] = [{"id": t.id, "type": "function",
                              "function": {"name": t.function.name, "arguments": t.function.arguments}}
                             for t in msg.tool_calls]
    u = resp.usage
    cost = getattr(u, "cost", None)
    if cost is None:
        cost = (u.prompt_tokens * PRICE_IN + u.completion_tokens * PRICE_OUT) / 1e6
    return out, {"in": u.prompt_tokens, "out": u.completion_tokens, "cost": float(cost)}
