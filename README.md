# Brave Search API — value comparison demo

A LangFuse-instrumented comparison of three pipelines answering the same finance questions:

| Run | Pipeline | What it shows |
| --- | --- | --- |
| `baseline-v1` | LLM only, no retrieval | Hallucination baseline |
| `diy-rag-v1` | Manual RAG (Search + fetch + extract + chunk + embed + index + retrieve) → LLM | Infrastructure overhead of doing it yourself |
| `brave-search-api-v1` | Brave **LLM Context** endpoint → LLM with citation enforcement | Quality lift from grounding |

Three evaluators score every output:
- `citation_rate` (0–1) — LLM-as-judge
- `factuality` (1–5) — LLM-as-judge
- `latency_ms` — measured during execution

## 2-minute live demo

The primary demo surface is the Jupyter notebook **`walkthrough.ipynb`** — it interleaves markdown narration, code execution, and outputs, so you can run cells and talk over them as you walk through.

**Before the demo (one-time):** make sure the required env vars (see [Setup](#setup)) are present in your environment, then:
```bash
jupyter lab walkthrough.ipynb # or jupyter notebook
```

**During the demo:**

```
T+0:00–0:30   Cells 1–7: intro + setup + show pipeline files
              → audience sees the file sizes contrast immediately

T+0:30–1:00   Cells 8–13: single question, three pipelines in parallel
              → ~25s wall-clock, visible latency gap
              → audience reads the actual answers side-by-side

T+1:00–1:45   Cells 14–25: dataset upload, experiments, LangFuse link
              → switch to LangFuse comparison view for the scores

T+1:45–2:00   Cells 26–30: code contrast + takeaways
              → diy_rag vs brave-search-api source side-by-side via inspect.getsource()
```

The notebook also works as a self-paced exploration — collaborators can clone the repo, set the env vars, and walk through it themselves.

## The two comparisons this demo demonstrates

**1. Quality** — `baseline` vs `brave-search-api`. Does retrieval reduce hallucinations on questions about recent events?

**2. Infrastructure** — `diy-rag` vs `brave-search-api`. If you build RAG from scratch instead of using Brave's LLM Context endpoint, how much extra code, latency, and dependency surface do you take on — and do you get better answers for the trouble?

## Infrastructure overhead (the `diy-rag` vs `brave-search-api` story)

Both pipelines start from the same place (Brave's search index) and end at the same place (Claude with a citation-forcing prompt). The only thing that changes between them is **what handles the middle**:

| Step | `diy-rag` (from scratch) | `brave-search-api` (LLM Context) |
| --- | --- | --- |
| 1. Search | Brave Web Search API | ✓ included |
| 2. Fetch HTML | `requests` with timeouts, redirects, UA | ✓ included |
| 3. Extract main content | `trafilatura` | ✓ included |
| 4. Chunk text | Custom paragraph-aware splitter | ✓ included |
| 5. Embed chunks | `sentence-transformers` (local model) | ✓ included |
| 6. Vector index | `faiss-cpu` | ✓ included |
| 7. Top-k retrieval | Inner-product search | ✓ included |
| 8. Generate with citations | Same | Same |

Concretely:

|  | `diy-rag` | `brave-search-api` |
| --- | --- | --- |
| Pipeline file size | ~200 lines | ~30 lines |
| External services | 1 (Brave Web Search) + local embedder + local vector store | 1 (Brave LLM Context) |
| Python deps | `requests`, `trafilatura`, `sentence-transformers`, `faiss-cpu`, `numpy` | `requests` |
| First-run setup | API key + ~80MB model download | API key |
| Expected p50 latency | ~5–10s | ~1–2s |
| Failure modes | Brave is down, page fetches fail, extraction fails, embedding fails, FAISS errors, chunking edge cases | Brave is down |

A production DIY setup would add managed vector DB (Pinecone/Weaviate), hosted embeddings (OpenAI/Voyage), and a reranker (Cohere). Each of those is more code, more services, more cost, more failure modes.

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) installed (`brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- A Brave Search API key on a plan that includes the **LLM Context** endpoint (verify at https://api-dashboard.search.brave.com/). Web Search is on most plans.
- An Anthropic API key
- [Docker](https://docs.docker.com/get-docker/) + Docker Compose — runs a **self-hosted LangFuse** locally, no cloud account needed (the official stack is vendored as `docker-compose.yml`)
- ~500MB disk for the sentence-transformers model + faiss, plus a few GB for the LangFuse containers (Postgres, ClickHouse, Redis, MinIO)

Python version is pinned in `.python-version`; uv installs the right interpreter automatically if you don't have it.

## Setup

```bash
git clone <this repo> && cd brave-langfuse-demo
uv sync                              # creates .venv/, installs all deps, writes uv.lock
```

### Self-hosted LangFuse (local, zero clicks)

This repo vendors LangFuse's official `docker-compose.yml` and turns on **headless initialization**, so the whole observability stack runs locally — no cloud account, no UI setup.

```bash
docker compose up -d                 # langfuse-web + worker, Postgres, ClickHouse, Redis, MinIO
```

First boot takes ~2–3 minutes (watch with `docker compose logs -f langfuse-web` until it logs `Ready`). On that first boot the compose file auto-creates an org, a project, an admin user, **and a fixed API key pair** — so you don't click through any setup. Just point the demo at it with the seeded keys (see below).

To browse traces in the UI, open **http://localhost:3005** and log in with `admin@example.com` / `langfuse-local`.

> The compose file ships with local-dev credentials (marked `# CHANGEME`, including the seeded keys/login) and binds everything except the web UI to `127.0.0.1`. Fine for local use — change them before exposing this stack anywhere. Stop it with `docker compose down` (add `-v` to also wipe stored traces; note the seed only runs against a fresh database).

### Environment variables

Make these available in the environment where you launch Jupyter — export them in your shell, source a `.env` you manage yourself, use direnv, whatever you prefer. The three `LANGFUSE_*` values below match what the compose file seeds, so you can paste them as-is:

```bash
export LANGFUSE_HOST=http://localhost:3005
export LANGFUSE_PUBLIC_KEY=pk-lf-local-brave-demo
export LANGFUSE_SECRET_KEY=sk-lf-local-brave-demo
```

**Required:**

| Var | Purpose |
| --- | --- |
| `BRAVE_API_KEY` | Brave Search API (LLM Context + Web Search) |
| `ANTHROPIC_API_KEY` | Claude generation + judges |
| `LANGFUSE_HOST` | Local LangFuse — `http://localhost:3005` |
| `LANGFUSE_PUBLIC_KEY` | Seeded by compose — `pk-lf-local-brave-demo` |
| `LANGFUSE_SECRET_KEY` | Seeded by compose — `sk-lf-local-brave-demo` |

**Optional (defaults shown):**

| Var | Default |
| --- | --- |
| `ANSWER_MODEL` | `claude-opus-4-7` |
| `JUDGE_MODEL` | `claude-opus-4-7` |

## Run

Make sure the local LangFuse stack is up (`docker compose up -d`) and your env vars are set, then:

```bash
uv run jupyter lab walkthrough.ipynb
```

(`uv run` executes inside the project's venv without needing an explicit `source .venv/bin/activate`.)

Then run cells top-to-bottom. First run downloads the sentence-transformers model (~80MB). After that, the full eval (5 questions × 3 pipelines × judges) completes in ~45–60 seconds with concurrent execution. Small enough to live-demo.

## What to look at in LangFuse

After running the experiment cells, the notebook prints a LangFuse URL. Open it and:

1. **Datasets → brave-finance-eval → Runs** — select all three (`baseline-v1`, `diy-rag-v1`, `brave-search-api-v1`) and click "Compare." Three columns, three scores per row.
2. **Trace view** — click any item to see the full chain. For `diy-rag` traces, check the metadata for the `pipeline` and `latency_ms` fields.
3. **Aggregate scores per run** — these are your headline numbers for the write-up.

## Tuning knobs

While iterating, change `RUN_TAG = "v2"` in the notebook so each variation shows up as a separate column in the LangFuse compare view.

In `pipelines/brave_search_api.py`:
- `CITATION_SYSTEM` — tighten or loosen citation rules
- `fetch_brave_context` params — try `count=20`, `context_threshold_mode="lenient"`, or `freshness="pw"`

In `pipelines/diy_rag.py`:
- `chunk_text` size/overlap — affects retrieval granularity
- `retrieve_top_k` value of `k` — affects how much context the LLM sees
- `EMBEDDING_MODEL_NAME` — swap to `BAAI/bge-small-en-v1.5` or others
- `fetch_all` `max_workers` — concurrency for the fetch stage

In the notebook itself:
- `QUESTIONS` list (cell 15) — swap the slant (crypto, macro only, regulatory only, etc.)
- Judge prompts (`evaluation.py`'s `JUDGE_CITATION` / `JUDGE_FACTUALITY`) — change what "good" means

## File layout

```
walkthrough.ipynb     # primary demo surface — narration + code + outputs interleaved
evaluation.py         # library: judges + task wrappers imported by the notebook

pipelines/
    __init__.py          # exposes the three pipeline functions
    baseline.py          #  32 lines — LLM only, no retrieval
    diy_rag.py           # 280 lines — manual RAG (search + fetch + extract + chunk + embed + index + retrieve)
    brave_search_api.py  # 110 lines — Brave LLM Context endpoint -> LLM with citations

pyproject.toml        # project metadata + dependencies (grouped by pipeline)
.python-version       # Python version pin for uv
uv.lock               # generated by `uv sync` — pinned dependency tree (commit this)
```

The three pipelines are siblings in `pipelines/`. The size difference between `diy_rag.py` and `brave_search_api.py` *is the point* — both produce cited answers under the same contract; only the retrieval implementation differs.

## Cost notes

For one full sweep (5 questions × 3 runs):
- Brave: ~5 LLM Context calls + ~5 Web Search calls + ~40 page fetches
- Anthropic: ~15 generation calls + ~30 judge calls
- Sentence-transformers: free (runs locally on CPU)

In dollar terms this is well under $1 per sweep. Run as many variations as you want while iterating.

## Caveats

- The refusal-test question in the dataset (forward-looking) is intentionally unanswerable. Both retrieval pipelines (`diy-rag` and `brave-search-api`) should refuse cleanly; baseline often won't. That contrast is part of the demo.
- The DIY pipeline uses a small local embedder (`all-MiniLM-L6-v2`) and in-memory FAISS for portability. A production setup with hosted embeddings + managed vector DB would have *more* dependencies and infrastructure, not fewer.
- Judge scoring is non-deterministic. For tighter eval, set `JUDGE_MODEL` to your strongest available model and run each experiment 2–3 times to estimate variance.
