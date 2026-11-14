"""CLI Trace Viewer for Legal RAG Chatbot Observability.

Inspect traces, retrieved chunks, latency breakdowns, and the golden dataset review list.

Usage:
    python view_trace.py <trace_id>
    python view_trace.py --latest
    python view_trace.py --list
    python view_trace.py --review-list
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from observability import tracer


def print_trace_details(trace: dict[str, Any]) -> None:
    """Print a readable, formatted tree representation of an LLM RAG trace."""
    print("=" * 80)
    print("                      RAG OBSERVABILITY TRACE INSPECTOR                      ")
    print("=" * 80)
    print(f"Trace ID     : {trace.get('trace_id')}")
    print(f"Timestamp    : {trace.get('timestamp')}")
    print(f"User / Role  : @{trace.get('user', 'unknown')} ({trace.get('role', 'general')})")
    print(f"Trace URL    : {trace.get('trace_url')}")
    print(f"Total Time   : {trace.get('total_latency_seconds')} seconds")
    print(f"Status       : {trace.get('status')} | Blocked: {trace.get('is_blocked')} | Not Found: {trace.get('not_found')}")

    feedback = trace.get("feedback")
    if feedback:
        fb_icon = "👍 Thumbs Up" if feedback.get("type") == "thumbs_up" else "👎 Thumbs Down"
        comment = f" - \"{feedback.get('comment')}\"" if feedback.get("comment") else ""
        print(f"Feedback     : {fb_icon}{comment}")
    else:
        print(f"Feedback     : No feedback recorded yet")

    print("-" * 80)
    print(f"QUESTION:\n{trace.get('question')}\n")
    print("-" * 80)

    # Retrieval Span
    spans = trace.get("spans", {})
    retrieval = spans.get("retrieval")
    if retrieval:
        print(f"🔍 RETRIEVAL SPAN (Mode: {retrieval.get('retrieval_mode', 'hybrid').upper()}, Latency: {retrieval.get('latency_seconds')}s):")
        print(f"   Search Query: \"{retrieval.get('search_query')}\"")
        chunks = retrieval.get("chunks", [])
        print(f"   Chunks Retrieved: {len(chunks)}")
        for ch in chunks:
            sim = f"sim={ch.get('similarity'):.4f}" if ch.get('similarity') is not None else ""
            rerank = f", rerank={ch.get('rerank_score'):.4f}" if ch.get('rerank_score') is not None else ""
            print(f"     #{ch.get('rank')} {ch.get('citation')} ({sim}{rerank}, id={ch.get('chunk_id')})")
            print(f"        \"{ch.get('text_preview')}\"")
        print()

    # Guardrails Spans
    guardrails = spans.get("guardrails", [])
    if guardrails:
        print(f"🛡️ GUARDRAILS EVALUATIONS ({len(guardrails)} checks executed):")
        for g in guardrails:
            status = "ALLOWED" if g.get("allowed") else f"BLOCKED ({g.get('action')})"
            print(f"   - {g.get('rail_type').upper()} Rail: {status} ({g.get('latency_seconds')}s, model={g.get('model_used')})")
        print()

    # Generation Span
    gen = spans.get("generation")
    if gen:
        print(f"🤖 GENERATION SPAN (Model: {gen.get('model')}, Latency: {gen.get('latency_seconds')}s):")
        if gen.get("tokens_in"):
            print(f"   Tokens: {gen.get('tokens_in')} in / {gen.get('tokens_out')} out")
        print()

    print("-" * 80)
    print("FINAL BOT ANSWER:")
    print(trace.get("answer"))
    print("=" * 80)
    print()


def list_recent_traces(limit: int = 10) -> None:
    traces = tracer.get_latest_traces(limit=limit)
    if not traces:
        print("No traces found in logs/traces.jsonl.")
        return

    print("=" * 80)
    print(f"RECENT RAG TRACES (Last {len(traces)} questions)")
    print("=" * 80)
    print(f"{'Trace ID':<38} | {'User':<12} | {'Time':<7} | {'Feedback':<11} | {'Question Preview':<25}")
    print("-" * 105)
    for t in traces:
        tid = t.get("trace_id", "")
        usr = f"@{t.get('user', 'anon')}"
        lat = f"{t.get('total_latency_seconds', 0):.2f}s"
        fb_data = t.get("feedback")
        fb_str = "👍 Up" if (fb_data and fb_data.get("type") == "thumbs_up") else ("👎 Down" if (fb_data and fb_data.get("type") == "thumbs_down") else "None")
        q_prev = (t.get("question") or "")[:25] + "..."
        print(f"{tid:<38} | {usr:<12} | {lat:<7} | {fb_str:<11} | {q_prev:<25}")
    print("=" * 105)
    print(f"\nTip: Run `python view_trace.py <trace_id>` to inspect the full trace tree.")


def show_review_list() -> None:
    items = tracer.get_review_items()
    if not items:
        print("No thumbs-down questions found in golden_review_list.json.")
        return

    print("=" * 80)
    print(f"GOLDEN DATASET REVIEW LIST ({len(items)} Downvoted Questions)")
    print("=" * 80)
    for idx, item in enumerate(items, start=1):
        print(f"[{idx}/{len(items)}] ID: {item.get('id')} | Trace: {item.get('trace_id')}")
        print(f"Timestamp   : {item.get('timestamp')}")
        print(f"User / Role : @{item.get('user')} ({item.get('role')})")
        print(f"Reason      : {item.get('reason')}")
        print(f"Question    :\n\"{item.get('question')}\"")
        print(f"Bot Answer  :\n{item.get('bot_answer')[:250]}...")
        sources = item.get("retrieved_sources", [])
        if sources:
            print("Retrieved Sources:")
            for s in sources:
                print(f"  - {s.get('citation')} (sim={s.get('similarity')})")
        print(f"Curation Status: {item.get('curation_status')}")
        print("-" * 80)
    print("\nTip: Review these questions and add ground truth answers to your golden dataset!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect RAG chatbot traces and review list.")
    parser.add_argument("trace_id", nargs="?", default=None, help="Trace ID to inspect.")
    parser.add_argument("--latest", action="store_true", help="View the most recent question trace.")
    parser.add_argument("--list", action="store_true", help="List recent traces.")
    parser.add_argument("--review-list", action="store_true", help="Display all thumbs-down questions saved for review.")
    args = parser.parse_args()

    if args.review_list:
        show_review_list()
        return

    if args.list:
        list_recent_traces()
        return

    if args.latest:
        traces = tracer.get_latest_traces(limit=1)
        if not traces:
            print("No traces found. Ask a question through app.py or answer.py first!")
            return
        print_trace_details(traces[0])
        return

    if args.trace_id:
        t = tracer.get_trace(args.trace_id)
        if not t:
            print(f"Trace '{args.trace_id}' not found in logs/traces.jsonl.")
            sys.exit(1)
        print_trace_details(t)
        return

    # Default action if no arguments: list recent
    list_recent_traces()


if __name__ == "__main__":
    main()

