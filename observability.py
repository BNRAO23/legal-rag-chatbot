"""Observability, Tracing, and Feedback Management for Legal RAG Chatbot.

Provides end-to-end tracing across the full RAG pipeline:
1. Tracing with Langfuse:
   - If LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set, syncs directly to
     Langfuse Cloud (free 50k traces/mo) or a local self-hosted instance.
   - If keys are not set, runs in local offline mode, recording full traces to
     `logs/traces.jsonl` with an in-app viewer and CLI inspection tool.
2. Comprehensive Span Logging:
   - Input Question and User Identity (username, role)
   - Chunks Retrieved (metadata, citations, similarity, re-rank scores, text)
   - Answer Generation (prompt, model, output tokens, input tokens, latency)
   - Guardrails Decisions (jailbreak, topic, safety, grounding)
3. User Feedback & Golden Dataset Review List:
   - Thumbs up (+1) and Thumbs down (0) scoring linked to trace_id.
   - Every thumbs-down question is automatically saved to `golden_review_list.json`
     with full retrieval context, ready for human curation and golden dataset expansion.
"""

from __future__ import annotations

import datetime
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from rag_config import get_secret

TRACES_LOG_PATH = Path("logs/traces.jsonl")
REVIEW_LIST_PATH = Path("golden_review_list.json")


class ObservabilityManager:
    """Manages LLM traces, feedback scoring, and golden review list storage."""

    def __init__(self, log_path: Path = TRACES_LOG_PATH, review_path: Path = REVIEW_LIST_PATH) -> None:
        self.log_path = log_path
        self.review_path = review_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.langfuse_client = None
        self.public_key = get_secret("LANGFUSE_PUBLIC_KEY") or ""
        self.secret_key = get_secret("LANGFUSE_SECRET_KEY") or ""
        self.host = get_secret("LANGFUSE_HOST") or "https://cloud.langfuse.com"

        if self.public_key and self.secret_key:
            try:
                from langfuse import Langfuse
                self.langfuse_client = Langfuse(
                    public_key=self.public_key,
                    secret_key=self.secret_key,
                    host=self.host,
                )
            except Exception:
                self.langfuse_client = None

        self._in_memory_traces: dict[str, dict[str, Any]] = {}

    @property
    def is_cloud_enabled(self) -> bool:
        return self.langfuse_client is not None

    def start_trace(
        self,
        question: str,
        user: str = "anonymous",
        role: str = "general",
        session_id: str | None = None,
    ) -> str:
        """Create a new trace and return its unique trace_id."""
        trace_id = str(uuid.uuid4())
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        record = {
            "trace_id": trace_id,
            "timestamp": now_iso,
            "user": user,
            "role": role,
            "session_id": session_id or trace_id,
            "question": question,
            "start_time": time.perf_counter(),
            "spans": {},
            "feedback": None,
            "status": "in_progress",
        }
        self._in_memory_traces[trace_id] = record
        return trace_id

    def log_guardrail_check(
        self,
        trace_id: str,
        rail_type: str,
        decision: Any,
    ) -> None:
        """Record input or output guardrail evaluation in the trace."""
        record = self._in_memory_traces.get(trace_id)
        if not record:
            return

        rail_data = {
            "rail_type": rail_type,
            "allowed": getattr(decision, "allowed", True),
            "action": getattr(decision, "action", "allow"),
            "model_used": getattr(decision, "model_used", ""),
            "latency_seconds": getattr(decision, "latency_seconds", 0.0),
            "message": getattr(decision, "message", ""),
        }
        record["spans"].setdefault("guardrails", []).append(rail_data)

    def log_retrieval(
        self,
        trace_id: str,
        search_query: str,
        retrieved_chunks: list[dict[str, Any]],
        latency_seconds: float,
        retrieval_mode: str = "hybrid",
    ) -> None:
        """Record retrieval candidate chunks, scores, citations, and latency."""
        record = self._in_memory_traces.get(trace_id)
        if not record:
            return

        serialized_chunks = []
        for c in retrieved_chunks:
            meta = c.get("metadata") or {}
            serialized_chunks.append(
                {
                    "rank": c.get("rank"),
                    "chunk_id": c.get("chunk_id"),
                    "citation": c.get("citation"),
                    "similarity": c.get("similarity"),
                    "rerank_score": c.get("rerank_score"),
                    "document_name": meta.get("document_name"),
                    "section": meta.get("unit_number"),
                    "page": meta.get("page"),
                    "text_preview": (c.get("text") or "")[:250] + "...",
                }
            )

        record["spans"]["retrieval"] = {
            "search_query": search_query,
            "retrieval_mode": retrieval_mode,
            "chunks_retrieved_count": len(retrieved_chunks),
            "latency_seconds": round(latency_seconds, 4),
            "chunks": serialized_chunks,
        }

    def log_generation(
        self,
        trace_id: str,
        answer: str,
        latency_seconds: float,
        chat_model: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
    ) -> None:
        """Record LLM generation output and token metrics."""
        record = self._in_memory_traces.get(trace_id)
        if not record:
            return

        record["spans"]["generation"] = {
            "model": chat_model,
            "latency_seconds": round(latency_seconds, 4),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "answer_preview": (answer or "")[:300],
        }

    def end_trace(
        self,
        trace_id: str,
        answer: str,
        retrieved_chunks: list[dict[str, Any]],
        latency_seconds: float | None = None,
        is_blocked: bool = False,
        not_found: bool = False,
    ) -> dict[str, Any]:
        """Finalize the trace, persist locally to traces.jsonl, and sync to Langfuse if active."""
        record = self._in_memory_traces.get(trace_id)
        if not record:
            return {}

        start_time = record.pop("start_time", time.perf_counter())
        total_latency = latency_seconds if latency_seconds is not None else (time.perf_counter() - start_time)

        record["total_latency_seconds"] = round(total_latency, 3)
        record["answer"] = answer
        record["is_blocked"] = is_blocked
        record["not_found"] = not_found
        record["status"] = "completed"

        # Construct trace URL (Langfuse Cloud or local inspector)
        if self.is_cloud_enabled:
            record["trace_url"] = f"{self.host.rstrip('/')}/trace/{trace_id}"
        else:
            record["trace_url"] = f"local://traces/{trace_id}"

        # 1. Append to local JSONL log
        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

        # 2. Sync to Langfuse if credentials are provided
        if self.langfuse_client:
            try:
                # In Langfuse v5+, create span and generation observations
                span = self.langfuse_client.start_observation(
                    name="legal-rag-query",
                    as_type="chain",
                    input={"question": record["question"], "user": record["user"], "role": record["role"]},
                    output={"answer": answer, "chunks_count": len(retrieved_chunks)},
                    metadata={
                        "trace_id": trace_id,
                        "latency_seconds": total_latency,
                        "is_blocked": is_blocked,
                        "not_found": not_found,
                    },
                )
                span.end()
                self.langfuse_client.flush()
            except Exception:
                pass

        return record

    def record_feedback(
        self,
        trace_id: str,
        feedback: str,  # "thumbs_up" or "thumbs_down"
        question: str = "",
        answer: str = "",
        chunks: list[dict[str, Any]] | None = None,
        user: str = "user",
        role: str = "general",
        comment: str = "",
    ) -> dict[str, Any]:
        """Record user thumbs up/down, update trace, and add thumbs-down to golden review list."""
        score_value = 1.0 if feedback == "thumbs_up" else 0.0
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        fb_data = {
            "score": score_value,
            "type": feedback,
            "comment": comment,
            "timestamp": now_iso,
        }

        # 1. Update in-memory record
        if trace_id in self._in_memory_traces:
            self._in_memory_traces[trace_id]["feedback"] = fb_data

        # 2. Update entry in local JSONL log file
        if self.log_path.exists():
            try:
                updated_lines = []
                with self.log_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        t_obj = json.loads(line)
                        if t_obj.get("trace_id") == trace_id:
                            t_obj["feedback"] = fb_data
                        updated_lines.append(t_obj)
                with self.log_path.open("w", encoding="utf-8") as f:
                    for t_obj in updated_lines:
                        f.write(json.dumps(t_obj, ensure_ascii=False) + "\n")
            except Exception:
                pass

        # 3. Sync feedback score to Langfuse if active
        if self.langfuse_client:
            try:
                self.langfuse_client.create_score(
                    name="user_feedback",
                    value=score_value,
                    trace_id=trace_id,
                    comment=comment or feedback,
                )
                self.langfuse_client.flush()
            except Exception:
                pass

        # 3. IF THUMBS DOWN: Save directly to Golden Dataset Review List
        review_entry = None
        if feedback == "thumbs_down":
            review_entry = self.save_to_review_list(
                trace_id=trace_id,
                question=question,
                bot_answer=answer,
                chunks=chunks or [],
                user=user,
                role=role,
                reason=comment or "User downvoted response via thumbs down",
            )

        return {
            "trace_id": trace_id,
            "feedback": feedback,
            "score": score_value,
            "added_to_review_list": feedback == "thumbs_down",
            "review_entry": review_entry,
        }

    def save_to_review_list(
        self,
        trace_id: str,
        question: str,
        bot_answer: str,
        chunks: list[dict[str, Any]],
        user: str,
        role: str,
        reason: str = "",
    ) -> dict[str, Any]:
        """Save a downvoted question to golden_review_list.json with full context for dataset curation."""
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        review_items: list[dict[str, Any]] = []

        if self.review_path.exists():
            try:
                with self.review_path.open("r", encoding="utf-8") as f:
                    review_items = json.load(f)
            except Exception:
                review_items = []

        # Check for duplicates of the same question or trace
        existing = next((r for r in review_items if r.get("trace_id") == trace_id), None)
        if existing:
            return existing

        review_id = f"REV-{len(review_items) + 1:03d}"

        # Format sources into standard golden dataset must_cite format
        must_cite_candidates = []
        for c in chunks:
            cit = c.get("citation") or ""
            if cit:
                must_cite_candidates.append(cit)

        entry = {
            "id": review_id,
            "trace_id": trace_id,
            "timestamp": now_iso,
            "user": user,
            "role": role,
            "question": question,
            "bot_answer": bot_answer,
            "reason": reason,
            "retrieved_sources": [
                {
                    "citation": c.get("citation"),
                    "chunk_id": c.get("chunk_id"),
                    "similarity": c.get("similarity"),
                    "rerank_score": c.get("rerank_score"),
                }
                for c in chunks[:5]
            ],
            # Golden dataset import templates (for human curation)
            "expected_output": "",  # To be filled by legal expert
            "must_cite": must_cite_candidates,
            "curation_status": "pending_curation",
        }

        review_items.append(entry)

        try:
            with self.review_path.open("w", encoding="utf-8") as f:
                json.dump(review_items, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

        return entry

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        """Find a trace by ID from memory or local log."""
        if trace_id in self._in_memory_traces:
            return self._in_memory_traces[trace_id]

        if self.log_path.exists():
            try:
                with self.log_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            t = json.loads(line)
                            if t.get("trace_id") == trace_id:
                                return t
            except Exception:
                pass
        return None

    def get_latest_traces(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return the most recent traces recorded."""
        traces = []
        if self.log_path.exists():
            try:
                with self.log_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            traces.append(json.loads(line))
            except Exception:
                pass
        return list(reversed(traces[-limit:]))

    def get_review_items(self) -> list[dict[str, Any]]:
        """Return all questions saved to the review list."""
        if not self.review_path.exists():
            return []
        try:
            with self.review_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []


# Global singleton instance
tracer = ObservabilityManager()
