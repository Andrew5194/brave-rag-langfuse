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
import anthropic
import requests

BRAVE_API_KEY = os.environ["BRAVE_API_KEY"]
BRAVE_LLM_CONTEXT_URL = "https://api.search.brave.com/res/v1/llm/context"
ANSWER_MODEL = os.environ.get("ANSWER_MODEL", "claude-opus-4-7")

_claude = anthropic.Anthropic()


# Citation contract: also imported by diy_rag.py so both retrieval pipelines
# produce answers under the same rules. The only thing that varies is HOW
# the sources got retrieved.
CITATION_SYSTEM = """You are a financial research assistant. Answer the user's question using ONLY the sources provided below.

STRICT RULES:
- Every factual claim (numbers, dates, names, events, decisions) MUST be followed by a citation marker like [1], [2], etc., matching the source numbers below.
- If the sources do not contain enough information to answer, respond exactly: "I cannot verify this from the provided sources."
- Do not use prior knowledge that is not supported by the sources.
- Keep the answer to 3-6 sentences.

SOURCES:
{sources}
"""


def fetch_brave_context(query: str, freshness: str = "pm") -> list[dict]:
    """
    Single API call. Brave handles search + fetch + extract + chunk + rank
    and returns pre-extracted snippets ready for LLM consumption.

    freshness: 'pd' past day, 'pw' past week, 'pm' past month, 'py' past year.
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


def format_sources(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{c['n']}] {c['title']} ({c['url']})\n{c['snippet']}"
        for c in chunks
    )


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
    system_prompt = CITATION_SYSTEM.format(sources=sources_text)
    msg = _claude.messages.create(
        model=ANSWER_MODEL,
        max_tokens=600,
        system=system_prompt,
        messages=[{"role": "user", "content": question}],
    )
    return {
        "answer": msg.content[0].text,
        "sources": [{"n": c["n"], "url": c["url"], "title": c["title"]} for c in chunks],
        "sources_text": sources_text,
        "chunks_returned": len(chunks),
    }
