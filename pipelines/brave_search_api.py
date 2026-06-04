"""
Brave Search API pipeline — Brave LLM Context endpoint -> LLM with citation enforcement.

Two API calls total:
  1. Brave LLM Context  (handles search + fetch + extract + chunk + rank)
  2. Claude generation  (with citation-forcing system prompt)

That's the whole pipeline. Compare to diy_rag.py for what those two
calls replace.
"""

from __future__ import annotations

import os

import requests

from ._llm import CITATION_SYSTEM, format_sources, generate

BRAVE_API_KEY = os.environ["BRAVE_API_KEY"]
BRAVE_LLM_CONTEXT_URL = "https://api.search.brave.com/res/v1/llm/context"


def fetch_brave_context(query: str, freshness: str = "pm") -> list[dict]:
    """Single API call. Brave handles search + fetch + extract + chunk + rank
    and returns pre-extracted snippets ready for LLM consumption.

    Args:
        query: The search query to send to Brave.
        freshness: Recency filter — ``pd`` past day, ``pw`` past week,
            ``pm`` past month, ``py`` past year.

    Returns:
        List of chunk dicts with keys ``n``, ``url``, ``title``, ``snippet``.
    """
    resp = requests.get(
        BRAVE_LLM_CONTEXT_URL,
        headers={
            "X-Subscription-Token": BRAVE_API_KEY,
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
        },
        params={
            "q": query,
            "count": 10,
            "maximum_number_of_tokens": 8192,
            "freshness": freshness,
            "context_threshold_mode": "strict",  # precision over recall
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    chunks: list[dict] = []
    for page in data.get("grounding", {}).get("generic", []):
        for snippet in page.get("snippets", []):
            chunks.append(
                {
                    "n": len(chunks) + 1,
                    "url": page.get("url", ""),
                    "title": page.get("title", ""),
                    "snippet": snippet,
                }
            )
    return chunks


def brave_search_api_pipeline(question: str) -> dict:
    """Brave LLM Context -> LLM with citation enforcement."""
    chunks = fetch_brave_context(question)
    if not chunks:
        return {
            "answer": "I cannot verify this from the provided sources.",
            "sources": [],
            "chunks_returned": 0,
        }

    sources_text = format_sources(chunks)
    answer = generate(CITATION_SYSTEM.format(sources=sources_text), question)
    return {
        "answer": answer,
        "sources": [{"n": c["n"], "url": c["url"], "title": c["title"]} for c in chunks],
        "sources_text": sources_text,
        "chunks_returned": len(chunks),
    }
