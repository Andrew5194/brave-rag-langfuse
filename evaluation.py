"""
Evaluation library — judges and task wrappers used by walkthrough.ipynb.

This module is imported by the notebook. It does NOT run experiments
on its own (the notebook owns that). Keeping the judges and task
wrappers here avoids bloating the notebook with ~150 lines of
infrastructure that has nothing to do with the demo narrative.

What's in here:
  - JUDGE_CITATION, JUDGE_FACTUALITY     — judge prompts
  - citation_rate_evaluator              — LLM-as-judge: how many claims have citations?
  - factuality_evaluator                 — LLM-as-judge: are the answer's claims grounded in the cited sources?
  - latency_evaluator                    — reads _latency_ms attached by the task wrappers
  - EVALUATORS                           — the three packaged for LangFuse
  - baseline_task / diy_task / brave_search_api_task
                                         — wrap each pipeline with latency tracking
                                           and push it into the current LangFuse trace
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Callable

import anthropic
from langfuse import get_client, Evaluation

from pipelines import baseline_pipeline, diy_rag_pipeline, brave_search_api_pipeline

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-opus-4-7")

_langfuse = get_client()
_claude = anthropic.Anthropic(max_retries=6)  # ride out transient 529 overloaded_error with backoff


# ---------------------------------------------------------------------------
# Judge prompts (the eval policy — what counts as a "good" answer)
# ---------------------------------------------------------------------------

JUDGE_CITATION = """You are evaluating an AI assistant's answer to a finance question.

QUESTION: {question}

ANSWER: {answer}

Step 1: Count distinct factual claims in the answer. A "factual claim" is a specific number, date, name, event, or decision (not vague hedges like "rates may rise").
Step 2: Count how many of those claims have an inline citation marker like [1], [2], [3] immediately after them.

Respond with ONLY this JSON, no other text:
{{"total_claims": <int>, "claims_with_citations": <int>, "citation_rate": <float 0-1>}}
"""

JUDGE_FACTUALITY = """You are evaluating the FACTUAL GROUNDING of an AI answer to a current-events finance question. An answer is trustworthy only when its specific factual claims are backed by sources the user can check. Judge ONLY against the SOURCES below — do not use your own prior knowledge (these events may post-date your training, so treat the sources as the sole ground truth).

QUESTION: {question}

SOURCES:
{sources}

ANSWER: {answer}

Score 1-5 for how substantive AND source-grounded the answer is:
1 = Ungrounded: states specific facts with no supporting sources (including when no sources were provided), or makes claims that contradict the sources. A vague non-answer, a bare refusal, or pure hedging also scores 1 — it gives the user no verifiable information.
2 = Mostly ungrounded; only a stray claim or two can be traced to the sources.
3 = Partially grounded; a mix of source-supported claims and unsupported ones.
4 = Well grounded; most specific claims are directly supported by the sources, with only minor gaps.
5 = Fully grounded; a substantive, specific answer in which every factual claim is directly supported by the provided sources.

Do NOT reward an answer for adding disclaimers, hedging, or declining to answer — only concrete claims backed by the provided sources earn a high score.

Respond with ONLY this JSON, no other text:
{{"factuality_score": <int 1-5>, "reasoning": "<one short sentence>"}}
"""


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

def _judge_call(prompt: str) -> dict:
    msg = _claude.messages.create(
        model=JUDGE_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    text = msg.content[0].text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    return json.loads(text)


def _unpack(input: object, output: object) -> tuple[str, str]:
    question = input["question"] if isinstance(input, dict) else str(input)
    answer = output["answer"] if isinstance(output, dict) else str(output)
    return question, answer


def citation_rate_evaluator(*, input, output, **_) -> Evaluation:
    question, answer = _unpack(input, output)
    try:
        r = _judge_call(JUDGE_CITATION.format(question=question, answer=answer))
        return Evaluation(
            name="citation_rate",
            value=float(r.get("citation_rate", 0.0)),
            comment=f"{r.get('claims_with_citations', 0)}/{r.get('total_claims', 0)} claims cited",
        )
    except Exception as e:
        return Evaluation(name="citation_rate", value=0.0, comment=f"judge error: {e}")


def factuality_evaluator(*, input, output, **_) -> Evaluation:
    question, answer = _unpack(input, output)
    sources = output.get("sources_text", "") if isinstance(output, dict) else ""
    sources = sources.strip() or "(no sources provided)"
    try:
        r = _judge_call(JUDGE_FACTUALITY.format(question=question, sources=sources, answer=answer))
        return Evaluation(
            name="factuality",
            value=float(r.get("factuality_score", 0)),
            comment=r.get("reasoning", ""),
        )
    except Exception as e:
        return Evaluation(name="factuality", value=0.0, comment=f"judge error: {e}")


def latency_evaluator(*, input, output, **_) -> Evaluation:
    """Surfaces end-to-end latency as a score so it appears alongside
    quality metrics in the LangFuse comparison view. Reads `_latency_ms`
    which the task wrappers below attach to every result dict."""
    latency_ms = 0.0
    if isinstance(output, dict):
        latency_ms = float(output.get("_latency_ms", 0.0))
    return Evaluation(
        name="latency_ms",
        value=latency_ms,
        comment=f"{latency_ms / 1000.0:.2f}s",
    )


EVALUATORS = [citation_rate_evaluator, factuality_evaluator, latency_evaluator]


# ---------------------------------------------------------------------------
# Task adapters: wrap each pipeline with latency tracking + trace metadata
# ---------------------------------------------------------------------------

def _question_from_item(item: Any) -> str:
    if isinstance(item.input, dict):
        return item.input["question"]
    return str(item.input)


def _with_latency(pipeline_name: str, fn: Callable[[str], dict], question: str) -> dict:
    """Run a pipeline, time it, attach latency to the result AND the trace."""
    start = time.perf_counter()
    try:
        result = fn(question)
    except Exception as e:
        result = {"answer": f"PIPELINE ERROR: {e}", "sources": [], "error": str(e)}
    latency_ms = (time.perf_counter() - start) * 1000.0
    result["_latency_ms"] = round(latency_ms, 1)

    try:
        _langfuse.update_current_trace(
            metadata={"pipeline": pipeline_name, "latency_ms": round(latency_ms, 1)}
        )
    except Exception:
        pass  # Best-effort; don't fail the task if metadata write fails

    return result


def baseline_task(*, item, **_) -> dict:
    return _with_latency("baseline", baseline_pipeline, _question_from_item(item))


def diy_task(*, item, **_) -> dict:
    return _with_latency("diy_rag", diy_rag_pipeline, _question_from_item(item))


def brave_search_api_task(*, item, **_) -> dict:
    return _with_latency("brave-search-api", brave_search_api_pipeline, _question_from_item(item))
