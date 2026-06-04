# Brave Search API — value comparison demo

A LangFuse-instrumented comparison of three ways to answer finance questions about recent events:

| Pipeline | Approach |
| --- | --- |
| `baseline` | LLM only, no retrieval |
| `diy_rag` | Manual RAG — search → fetch → extract → chunk → embed → index → retrieve → LLM |
| `brave-search-api` | Brave **LLM Context** endpoint → LLM with citations |

It exists to show two things side by side:

- **Quality** — does grounding (`brave-search-api`) reduce hallucinations versus the ungrounded `baseline`?
- **Infrastructure** — how much code, latency, and dependency overhead does hand-rolling RAG (`diy_rag`) add over Brave's single LLM Context call?

The whole demo lives in `walkthrough.ipynb`, scored by a self-hosted LangFuse.

## Getting started

Prerequisites: 
* [`uv`](https://docs.astral.sh/uv/)
* Docker + Docker Compose
* A Brave Search API key with the **LLM Context** endpoint
* An Anthropic API key.

Execute the following to get started:

```bash
make local
```

Open `walkthrough.ipynb` and set your Brave Search API key and Anthropic API key. The `LANGFUSE` values are what the compose file seeds on first boot, just use the defaults:

```bash
export BRAVE_API_KEY=...
export ANTHROPIC_API_KEY=...
export LANGFUSE_HOST=http://localhost:3005
export LANGFUSE_PUBLIC_KEY=pk-lf-local-brave-demo
export LANGFUSE_SECRET_KEY=sk-lf-local-brave-demo
```

Run all cells. Traces and scores appear at **http://localhost:3005** (log in with `admin@admin.com` / `password`).

## File tree

```
walkthrough.ipynb        # the demo — narration + code + outputs interleaved
evaluation.py            # LLM-as-judge evaluators + pipeline task wrappers
pipelines/
    _llm.py              # shared Claude client, citation prompt, generate() helper
    baseline.py          # LLM only, no retrieval
    diy_rag.py           # manual RAG (the eight steps above)
    brave_search_api.py  # Brave LLM Context → LLM with citations
docker-compose.yml       # self-hosted LangFuse (Postgres, ClickHouse, Redis, MinIO)
Makefile                 # make local | remote | down | format
pyproject.toml           # dependencies (uv) + ruff config
```
