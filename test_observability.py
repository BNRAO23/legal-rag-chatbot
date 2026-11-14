"""End-to-end test suite for RAG Observability, Feedback, and Trace Inspection.

Verifies:
1. Trace lifecycle: start_trace, log_retrieval, end_trace.
2. Local trace persistence to logs/traces.jsonl.
3. User feedback recording (thumbs up & thumbs down).
4. Automatic population of golden_review_list.json on thumbs down.
5. CLI inspection tool view_trace.py with various flags.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from observability import tracer


def test_observability_pipeline() -> None:
    print("=" * 70)
    print("STEP 1: Testing trace creation and logging")
    print("=" * 70)

    test_q = "What is the threshold for mandatory CSR spending under Section 135?"
    test_user = "test_lawyer"
    test_role = "corporate"

    # Start trace
    trace_id = tracer.start_trace(question=test_q, user=test_user, role=test_role)
    print(f"[*] Started trace ID: {trace_id}")
    assert trace_id, "trace_id should not be empty"

    # Log retrieval
    sample_chunks = [
        {
            "rank": 1,
            "chunk_id": "chunk_sec135_p93",
            "citation": "Companies Act, 2013 | Section 135 | p.93",
            "similarity": 0.884,
            "rerank_score": 0.941,
            "text": "Every company having net worth of rupees five hundred crore or more, or turnover of rupees one thousand crore or more or a net profit of rupees five crore or more during immediately preceding financial year shall constitute a Corporate Social Responsibility Committee...",
            "metadata": {
                "document_name": "Companies Act, 2013",
                "unit_number": "Section 135",
                "page": 93,
            },
        }
    ]
    tracer.log_retrieval(
        trace_id=trace_id,
        search_query=test_q,
        retrieved_chunks=sample_chunks,
        latency_seconds=0.142,
        retrieval_mode="hybrid",
    )
    print("[*] Logged retrieval span")

    # End trace
    test_ans = "Under Section 135(1) of the Companies Act, 2013, CSR applies to companies with net worth >= 500 cr, turnover >= 1000 cr, or net profit >= 5 cr [Companies Act, 2013 | Section 135 | p.93]."
    trace_record = tracer.end_trace(
        trace_id=trace_id,
        answer=test_ans,
        retrieved_chunks=sample_chunks,
        latency_seconds=0.485,
    )
    assert trace_record["status"] == "completed"
    assert trace_record["total_latency_seconds"] == 0.485
    print(f"[*] Ended trace. Recorded latency: {trace_record['total_latency_seconds']}s")

    # Verify persistence in logs/traces.jsonl
    log_path = Path("logs/traces.jsonl")
    assert log_path.exists(), "logs/traces.jsonl must exist"
    with log_path.open("r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    matching = [l for l in lines if l.get("trace_id") == trace_id]
    assert len(matching) >= 1, f"Trace {trace_id} must be in logs/traces.jsonl"
    print(f"[OK] Trace successfully persisted in {log_path}")

    # STEP 2: Thumbs Up Feedback Test
    print("\n" + "=" * 70)
    print("STEP 2: Testing Thumbs-Up Feedback")
    print("=" * 70)
    fb_res = tracer.record_feedback(
        trace_id=trace_id,
        feedback="thumbs_up",
        question=test_q,
        answer=test_ans,
        chunks=sample_chunks,
        user=test_user,
        role=test_role,
        comment="Accurate statutory citation",
    )
    assert fb_res["score"] == 1.0
    assert fb_res["added_to_review_list"] is False
    print("[OK] Thumbs up successfully registered and scored 1.0 (not added to review list)")

    # STEP 3: Thumbs Down & Golden Review List Test
    print("\n" + "=" * 70)
    print("STEP 3: Testing Thumbs-Down Feedback & Golden Review List Generation")
    print("=" * 70)
    down_q = "What are the rules for filing Form MGT-7 under the Companies Act, 2013?"
    down_trace_id = tracer.start_trace(question=down_q, user="tax_associate", role="tax")
    down_chunks = [
        {
            "rank": 1,
            "chunk_id": "chunk_sec92_p65",
            "citation": "Companies Act, 2013 | Section 92 | p.65",
            "similarity": 0.725,
            "text": "Every company shall prepare a return (hereinafter referred to as the annual return) in the prescribed form...",
            "metadata": {"document_name": "Companies Act, 2013", "unit_number": "Section 92", "page": 65},
        }
    ]
    tracer.log_retrieval(down_trace_id, down_q, down_chunks, 0.12, "hybrid")
    bot_ans_imperfect = "Companies must prepare an annual return in prescribed form [Companies Act, 2013 | Section 92 | p.65]."
    tracer.end_trace(down_trace_id, bot_ans_imperfect, down_chunks, 0.42)

    down_res = tracer.record_feedback(
        trace_id=down_trace_id,
        feedback="thumbs_down",
        question=down_q,
        answer=bot_ans_imperfect,
        chunks=down_chunks,
        user="tax_associate",
        role="tax",
        comment="Answer missed the 60-day deadline from AGM and specific e-form particulars",
    )
    assert down_res["score"] == 0.0
    assert down_res["added_to_review_list"] is True

    review_file = Path("golden_review_list.json")
    assert review_file.exists(), "golden_review_list.json must exist"
    with review_file.open("r", encoding="utf-8") as f:
        review_data = json.load(f)

    flagged_item = next((item for item in review_data if item.get("trace_id") == down_trace_id), None)
    assert flagged_item is not None, "Downvoted question must appear in golden_review_list.json"
    assert flagged_item["question"] == down_q
    assert flagged_item["user"] == "tax_associate"
    assert flagged_item["role"] == "tax"
    assert "must_cite" in flagged_item
    assert "expected_output" in flagged_item
    print(f"[OK] Question added to {review_file} with ID: {flagged_item.get('id')}")
    print(f"     Must-cite candidates: {flagged_item.get('must_cite')}")

    # STEP 4: CLI Trace Viewer Tests
    print("\n" + "=" * 70)
    print("STEP 4: Testing CLI Tool (view_trace.py)")
    print("=" * 70)

    # Test 1: view_trace.py --latest
    cmd1 = [sys.executable, "view_trace.py", "--latest"]
    out1 = subprocess.check_output(cmd1, text=True)
    assert "RAG OBSERVABILITY TRACE INSPECTOR" in out1
    assert down_trace_id in out1
    print("[OK] `python view_trace.py --latest` successfully displayed most recent trace")

    # Test 2: view_trace.py <trace_id>
    cmd2 = [sys.executable, "view_trace.py", trace_id]
    out2 = subprocess.check_output(cmd2, text=True)
    assert trace_id in out2
    assert "👍 Thumbs Up" in out2
    assert "Accurate statutory citation" in out2
    print(f"[OK] `python view_trace.py {trace_id}` successfully displayed trace details and thumbs-up feedback")

    # Test 3: view_trace.py --list
    cmd3 = [sys.executable, "view_trace.py", "--list"]
    out3 = subprocess.check_output(cmd3, text=True)
    assert "RECENT RAG TRACES" in out3
    print("[OK] `python view_trace.py --list` successfully displayed trace summary table")

    # Test 4: view_trace.py --review-list
    cmd4 = [sys.executable, "view_trace.py", "--review-list"]
    out4 = subprocess.check_output(cmd4, text=True)
    assert "GOLDEN DATASET REVIEW LIST" in out4
    assert down_q in out4
    print("[OK] `python view_trace.py --review-list` successfully displayed review queue")

    print("\n" + "=" * 70)
    print("ALL OBSERVABILITY AND FEEDBACK TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    test_observability_pipeline()

