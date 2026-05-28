# Demo script — 2-minute live walkthrough (notebook version)

The primary demo surface is `walkthrough.ipynb`. This script gives you what to say at each cell. Time markers assume you start running cells from the top.

## Pre-flight (30 seconds before going live)

- [ ] `walkthrough.ipynb` open in JupyterLab, kernel active, env vars loaded
- [ ] Notebook scrolled to the top, all previous outputs cleared (Kernel → Restart & Clear Outputs)
- [ ] One smoke-test run already done today (so the embedder is cached locally)
- [ ] Browser tab 1: LangFuse → Datasets → `brave-finance-eval` → Runs view, signed in
- [ ] Phone or stopwatch ready

## Opening line (before running the first cell)

> "I'm going to compare three ways of answering finance questions about recent events: just an LLM, that same LLM grounded with Brave's Search API, and a DIY RAG pipeline I built from scratch. LangFuse scores all three. The whole thing is a Jupyter notebook so you can see exactly what's happening at each step."

Run Cell 1 (the title markdown) — audience reads it while you talk.

---

## Phase 1 — Setup and single-question demo (cells 1–13, ~30s)

### Cells 1–4 (intro + env + clients)

> "Quick setup: verifying I have keys for Brave, Anthropic, and LangFuse, then initializing clients. The dataset will get uploaded later in the notebook."

Run cells 3 and 4 — they should complete in under a second each.

### Cells 5–7 (three pipelines + line counts)

This is your first hero moment. Run cell 7 (the line counts).

> "Three pipelines, same input/output contract. Baseline is the LLM with no retrieval. Grounded uses Brave's LLM Context endpoint and adds a citation prompt. DIY RAG is a manual implementation: search, fetch, extract, chunk, embed, index, retrieve, generate."

Point at the line counts as they print:

> "Thirty-two lines, a hundred-ten lines, two hundred-eighty lines. That number ratio is the whole story before we even run anything."

### Cells 8–10 (demo question + pre-warm)

> "Now let's actually run one question through all three pipelines, in parallel. The pre-warm cell is loading the local embedding model off the clock — that's setup, not part of the timed comparison."

Run cells 9 and 10. The pre-warm finishes in 5-10 seconds the first time, near-instant if cached.

### Cell 11 (the parallel run)

This is the live moment. Run cell 11.

> "Three pipelines launching in parallel. The wall-clock time is whoever finishes last, not the sum."

As outputs appear:

> "Baseline came back in two seconds. Grounded in four. DIY took fifteen because of all those sequential steps."

### Cell 12 (display answers)

Run cell 12. Three answer blocks appear.

> "Read these top-to-bottom. Baseline is hedged, vague, no specifics. Grounded has inline citation markers like `[1]` and `[2]`. DIY also has citations but the chunking is messier."

### Cell 13 (markdown observations)

Pause to let the audience read it. ~5 seconds.

> "That's the qualitative reveal. But three answers isn't an eval — let's quantify across a small dataset."

---

## Phase 2 — Eval setup and LangFuse experiments (cells 14–25, ~60s)

### Cells 14–15 (dataset upload)

Run cell 15.

> "Five hand-picked finance questions across categories: central bank, earnings, macro data, corporate, plus one deliberately unanswerable forward-looking one. Small enough to live-run, varied enough to be meaningful."

### Cells 16–18 (eval policy)

Run cells 17 and 18.

> "Three evaluators: citation rate, factuality, latency. Two are LLM-as-judge — another model scoring the answers — and one is just measured execution time. The judge prompt for citation rate is right there: it counts factual claims and checks how many have citation markers."

Point at the judge prompt.

> "I'm showing you this because the eval policy matters. The headline numbers are only as credible as what we're asking the judge to measure."

### Cells 19–23 (run the three experiments)

Run cell 20.

> "Loading the dataset from LangFuse. Now three experiments — one per pipeline."

Run cell 21 (baseline). It finishes in ~8 seconds.

> "Baseline experiment: five questions, all in parallel, generation plus two judges per item. Eight seconds total."

Run cell 22 (grounded). Finishes in ~10 seconds.

> "Grounded experiment. Slightly slower because of the Brave API call, but still fast."

Run cell 23 (DIY). Finishes in ~20-25 seconds.

> "DIY experiment. This is the long one — eight pipeline steps per question. Notice the wall-clock difference."

### Cells 24–25 (LangFuse link)

Run cell 25. The URL prints.

> "Now to LangFuse for the comparison view."

**Switch to LangFuse tab.** Navigate to Datasets → brave-finance-eval → Runs → select all three → Compare.

Point at the aggregate row:

> "Citation rate: baseline near zero, grounded near ninety percent, DIY around seventy. Factuality: baseline around two, grounded four-plus, DIY around three-and-a-half. Latency: grounded around three seconds, DIY around fifteen."

Click into one trace (Nvidia earnings or whichever has the most dramatic split):

> "Here's the full chain for the grounded run — Brave call, chunks returned, LLM call with the citation prompt, final output. Now compare to the baseline trace — same question, generic answer, no sources."

(Optional, 5s) The refusal test row:

> "And the forward-looking question — the model literally can't know — baseline speculates, grounded refuses cleanly. Citation enforcement gives you refusal behavior almost for free."

---

## Phase 3 — Code contrast and close (cells 26–30, ~30s)

**Switch back to the notebook.**

### Cells 26–27 (dependency split)

Run cell 27.

> "Here's the dependency declaration from pyproject.toml. The shared section is three packages — anthropic, langfuse, httpx. The DIY-only section adds four heavy libraries: trafilatura for extraction, sentence-transformers for embeddings, FAISS for the vector store, numpy. Every one of those is more code to write, more to maintain, more to fail."

### Cell 28 (grounded source via inspect)

Run cell 28. The grounded_pipeline source prints inline.

> "This is the entire grounded pipeline. Roughly thirty lines. One call to Brave, one call to Claude. That's it."

### Cell 29 (DIY source via inspect)

Run cell 29. The much longer source prints.

> "And this is the DIY pipeline. Eight numbered steps. Search, fetch concurrently, extract main content, chunk by paragraph, embed locally, build a FAISS index, retrieve top-k, generate. Two hundred-eighty lines. Same final answer contract. Worse scores in the eval we just ran. More to maintain forever."

### Cell 30 (takeaways markdown)

Scroll to it. Audience reads.

> "That's the demo. Brave's LLM Context endpoint collapses an entire retrieval stack into one API call, the eval shows simpler is also better here, and the whole thing is reproducible and instrumented end-to-end."

---

## If something goes wrong

| Failure | What to say | What to do |
| --- | --- | --- |
| Brave returns no results for one question | "Clean 'cannot verify' rather than hallucinating — that's the right behavior." | Keep going — it's a feature |
| A page fetch fails in DIY | "DIY had a page fetch fail. The kind of edge case you'd handle with retries in production. More code." | Use as ammunition for the infra point |
| Rate limit error from Anthropic | "Hit a rate limit. Let me lower concurrency and re-run that cell." | Change `max_concurrency` to 2 in the failing cell and re-run |
| Kernel crash or stuck cell | "Let me restart the kernel and we'll pick up from the eval." | Kernel → Restart → run from cell 17 (imports), then skip ahead |
| LangFuse API timeout | "Let me re-run that experiment cell." | Cells are idempotent — just rerun |
| Score gap smaller than expected | "Judge scoring is non-deterministic. The structural points still hold: latency, code size, dependency surface." | Pivot to Phase 3 sooner |

## Likely audience questions

- **"Why not OpenAI / Cohere / Voyage embeddings for DIY?"** → "Wanted DIY self-contained and free to run. Switching to hosted embeddings *adds* a fourth service and a per-query cost. Makes the contrast bigger, not smaller."
- **"Couldn't you optimize the DIY pipeline?"** → "Yes, and that's part of the point. Reranker, better chunker, hosted embeddings — all more code, more services, more cost. The eval is stock DIY vs. one Brave call."
- **"How robust is the LLM-as-judge?"** → "Non-deterministic. Real benchmark would run each experiment 2-3 times and average. For a demo, the gap is large enough that single-run noise doesn't change the conclusion."
- **"What about cost per query?"** → "Brave LLM Context is one call. DIY is Brave Web Search plus embeddings (free locally) plus the LLM call. Production DIY with hosted embeddings adds a recurring per-token cost — a real ongoing tax."
- **"Can I run this myself?"** → "Repo's public, README has setup. Five minutes from clone to first run if you already have API keys."

## After the demo (the 10-second close, if asked)

> "Everything's in the notebook, reproducible, instrumented with LangFuse. Pull it, swap your own dataset in the cell, and you have your own benchmark running in five minutes."
