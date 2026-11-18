# ActLens: small CPU-only image (no PyTorch). The search index is built during the image build.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HUB_DISABLE_TELEMETRY=1
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app ./app
COPY scripts ./scripts
COPY data/corpus.json data/evalset.json data/eval_retrieval.json data/eval_agent.json ./data/
RUN python scripts/build_index.py && chmod -R a+rX /app/data && useradd -r -u 10001 actlens && mkdir -p /var/lib/actlens && chown actlens /var/lib/actlens
USER actlens
ENV BUDGET_DB=/var/lib/actlens/budget.sqlite
EXPOSE 8000
HEALTHCHECK --interval=60s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
