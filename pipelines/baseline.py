"""
Baseline pipeline — LLM only, no retrieval.

This is the hallucination baseline. No external knowledge,
no citations, no grounding. Whatever the model knows from training.
"""

from __future__ import annotations

import os

import anthropic

ANSWER_MODEL = os.environ.get("ANSWER_MODEL", "claude-opus-4-5")

_claude = anthropic.Anthropic()

BASELINE_SYSTEM = (
    "You are a financial research assistant. Answer the user's question "
    "concisely and factually. Do not invent figures or dates you are not sure about."
)


def baseline_pipeline(question: str) -> dict:
    """One LLM call. No tools, no retrieval, no citations."""
    msg = _claude.messages.create(
        model=ANSWER_MODEL,
        max_tokens=600,
        system=BASELINE_SYSTEM,
        messages=[{"role": "user", "content": question}],
    )
    return {"answer": msg.content[0].text, "sources": []}
