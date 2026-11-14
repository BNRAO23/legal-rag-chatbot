"""DeepEval evaluation suite for the legal RAG chatbot with Hybrid vs Vector comparison.

Evaluates:
1. Retrieval quality: Contextual Recall, Contextual Precision, and Expected Source Retrieval.
2. Answer quality: Faithfulness, Answer Relevancy, and Citation Compliance.
3. Safety & Grounding: Refusal verification for NOT_FOUND questions.
4. Latency: Timed end-to-end for every question.
5. Search modes: Vector-only, Hybrid (Vector + BM25 with Reciprocal Rank Fusion), or Both side-by-side.

Saves detailed per-question results to `eval_results_{mode}.json` (and `.csv`)
and prints a comparative side-by-side scorecard.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"
os.environ["CONFIDENT_AI_API_KEY"] = ""
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

from answer import NOT_FOUND_RESPONSE, answer
from rag_config import RagConfig, load_rag_config


@dataclass
class EvalCase:
    id: str
    question: str
    expected: str
    must_cite: list[str]
    subject: str = ""
    topic: str = ""
    is_unanswerable: bool = False


@dataclass
class MetricResult:
    score: float | None
    reason: str = ""
    error: str | None = None


@dataclass
class CaseEvalResult:
    id: str
    question: str
    expected: str
    actual: str
    is_unanswerable: bool
    latency_seconds: float
    cites_expected_source: bool | str
    retrieved_expected_source: bool | str
    expected_doc: str
    expected_sections: list[str]
    not_found_refused: bool | str
    contextual_recall: MetricResult
    contextual_precision: MetricResult
    faithfulness: MetricResult
    answer_relevancy: MetricResult
    retrieved_sources: list[dict[str, Any]]
    retrieval_mode: str = "vector"
    rerank_enabled: bool = False
    filter_document_enabled: bool = False
    rewrite_query_enabled: bool = False
    parent_child_enabled: bool = False
    rewritten_query: str = ""
    detected_document: str = ""


def extract_expected_sections(case: EvalCase) -> tuple[str, list[str]]:
    """Extract expected Act keyword and section numbers from `must_cite`."""
    subject = (case.subject or "").lower()
    if "companies act" in subject:
        expected_doc = "Companies Act, 2013"
    elif "fema" in subject:
        expected_doc = "Foreign Exchange Management Act, 1999"
    elif "income-tax" in subject or "income tax" in subject:
        expected_doc = "Income-tax Act, 2025"
    else:
        expected_doc = ""

    sections: list[str] = []
    for cite in case.must_cite:
        cite_str = str(cite)
        if any(k in cite_str.lower() for k in ("companies act", "fema", "income-tax", "income tax", "section", "sec")):
            for num in re.findall(r"\b(\d+[A-Z]*)\b", cite_str):
                if num not in {"2013", "1999", "2025", "2026", "2014", "2019", "2022", "2000"}:
                    sections.append(num)
    # Fallback to question or topic if must_cite was generic
    if not sections:
        for text in (case.topic, case.question):
            for num in re.findall(r"\b(?:Section|Sec\.?|s\.)\s*(\d+[A-Z]*)\b", text, re.IGNORECASE):
                sections.append(num)

    deduped = list(dict.fromkeys(sections))
    return expected_doc, deduped


def load_evaluation_dataset(
    dataset_path: Path,
    unanswerable_path: Path | None = None,
    include_unanswerable: bool = True,
) -> list[EvalCase]:
    """Load evaluation cases from text/json files."""
    cases: list[EvalCase] = []

    # 1. Load primary dataset
    if dataset_path.exists():
        raw = dataset_path.read_text(encoding="utf-8")
        stripped = raw.strip()

        if stripped.startswith("["):
            json_data = json.loads(stripped)
            for item in json_data:
                cid = item.get("id") or item.get("additional_metadata", {}).get("id") or f"CASE-{len(cases)+1:03d}"
                q = item.get("question") or item.get("input") or ""
                exp = item.get("expected") or item.get("expected_output") or ""
                if isinstance(exp, list):
                    exp = "\n".join(exp)
                must_cite = item.get("must_cite") or item.get("additional_metadata", {}).get("sources", [])
                if isinstance(must_cite, list) and must_cite and isinstance(must_cite[0], dict):
                    must_cite = [f"{s.get('document_name')} Sec {s.get('unit_number')}" for s in must_cite]
                is_un = (
                    exp.strip() == "NOT_FOUND"
                    or item.get("additional_metadata", {}).get("question_type") == "unanswerable"
                )
                cases.append(
                    EvalCase(
                        id=cid,
                        question=q,
                        expected=exp,
                        must_cite=must_cite,
                        subject=item.get("subject", ""),
                        topic=item.get("topic", ""),
                        is_unanswerable=is_un,
                    )
                )
        else:
            for line in raw.splitlines():
                line_s = line.strip()
                if line_s.startswith("{") and line_s.endswith("}"):
                    try:
                        item = json.loads(line_s)
                        cid = item.get("id") or f"CASE-{len(cases)+1:03d}"
                        q = item.get("question") or item.get("input") or ""
                        exp = item.get("expected") or item.get("expected_output") or ""
                        if isinstance(exp, list):
                            exp = "\n".join(exp)
                        must_cite = item.get("must_cite", [])
                        is_un = exp.strip() == "NOT_FOUND"
                        cases.append(
                            EvalCase(
                                id=cid,
                                question=q,
                                expected=exp,
                                must_cite=must_cite,
                                subject=item.get("subject", ""),
                                topic=item.get("topic", ""),
                                is_unanswerable=is_un,
                            )
                        )
                    except json.JSONDecodeError:
                        continue

    # 2. Optionally load unanswerable questions (if not already included)
    if include_unanswerable and unanswerable_path and unanswerable_path.exists():
        existing_ids = {c.id for c in cases}
        try:
            un_data = json.loads(unanswerable_path.read_text(encoding="utf-8"))
            for item in un_data:
                cid = item.get("additional_metadata", {}).get("id") or item.get("id") or f"UN-{len(cases)+1:03d}"
                if cid in existing_ids:
                    continue
                q = item.get("input") or item.get("question") or ""
                exp = item.get("expected_output") or item.get("expected") or "NOT_FOUND"
                cases.append(
                    EvalCase(
                        id=cid,
                        question=q,
                        expected=exp,
                        must_cite=[],
                        subject="Unanswerable / Out of Corpus",
                        topic="Refusal Check",
                        is_unanswerable=True,
                    )
                )
        except Exception as exc:
            print(f"Warning: could not load unanswerable dataset from {unanswerable_path}: {exc}")

    return cases


def _init_eval_model(config: RagConfig, model_name: str | None = None) -> Any:
    """Initialize a DeepEval LLM judge model."""
    provider = config.chat_provider
    target_model = model_name or config.chat_model

    if provider == "google":
        from deepeval.models import GeminiModel

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY required for DeepEval GeminiModel.")
        return GeminiModel(model=target_model or "gemini-3.5-flash-lite", api_key=api_key)

    if provider == "openai":
        from deepeval.models import OpenAIModel

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY required for DeepEval OpenAIModel.")
        return OpenAIModel(model=target_model, api_key=api_key)

    if provider == "ollama":
        from deepeval.models import OllamaModel

        return OllamaModel(model=target_model, base_url=config.ollama_base_url)

    from deepeval.models import GeminiModel

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return GeminiModel(model=target_model or "gemini-3.5-flash-lite", api_key=api_key)


def _safe_measure_metric(metric: Any, test_case: Any) -> MetricResult:
    """Measure a DeepEval metric safely without aborting on failure."""
    for attempt in range(4):
        try:
            metric.measure(test_case)
            score = float(metric.score) if metric.score is not None else None
            reason = str(getattr(metric, "reason", "") or "")
            return MetricResult(score=score, reason=reason, error=None)
        except Exception as exc:
            msg = str(exc)
            if ("503" in msg or "429" in msg or "UNAVAILABLE" in msg or "timeout" in msg.lower()) and attempt < 3:
                time.sleep(5.0 * (attempt + 1))
                continue
            return MetricResult(score=None, reason="", error=str(exc))
    return MetricResult(score=None, reason="", error="Max retries exceeded")


def evaluate_single_case(
    case: EvalCase,
    top_k: int,
    config: RagConfig,
    metrics: dict[str, Any] | None,
    mode: str = "vector",
    rerank: bool = False,
    rerank_candidates: int | None = None,
    filter_document: bool = False,
    rewrite_query: bool = False,
    parent_child: bool = False,
) -> CaseEvalResult:
    """Run chatbot answer function on a case, perform rule checks, and compute DeepEval metrics."""
    # 1. Timed execution of answer function with chosen mode, reranking, doc filter, query rewrite, parent-child
    t0 = time.perf_counter()
    res = answer(
        case.question,
        top_k=top_k,
        config=config,
        mode=mode,
        rerank=rerank,
        rerank_candidates=rerank_candidates,
        filter_document=filter_document,
        rewrite_query=rewrite_query,
        parent_child=parent_child,
    )
    latency = round(time.perf_counter() - t0, 3)

    actual_output = res["answer"]
    retrieved_chunks = res.get("chunks", [])
    is_not_found = bool(res.get("not_found")) or (actual_output.strip() == NOT_FOUND_RESPONSE)

    # 2. Check 1: Refusal of NOT_FOUND questions
    if case.is_unanswerable:
        not_found_refused: bool | str = is_not_found
    else:
        not_found_refused = "N/A"

    # 3. Check 2: Expected source citation & retrieval
    expected_doc, expected_secs = extract_expected_sections(case)

    retrieved_sources_summary: list[dict[str, Any]] = []
    matched_retrievals: list[str] = []

    for ch in retrieved_chunks:
        m = ch.get("metadata", {})
        doc = str(m.get("document_name") or "")
        unit_num = str(m.get("unit_number") or "")
        retrieved_sources_summary.append(
            {
                "rank": ch.get("rank"),
                "chunk_id": ch.get("chunk_id"),
                "citation": ch.get("citation"),
                "similarity": ch.get("similarity"),
                "document": doc,
                "section": unit_num,
                "page": m.get("page"),
            }
        )
        doc_matches = (not expected_doc) or (doc == expected_doc)
        if doc_matches and unit_num in expected_secs:
            matched_retrievals.append(unit_num)

    if case.is_unanswerable:
        cites_expected_source: bool | str = "N/A"
        retrieved_expected_source: bool | str = "N/A"
    else:
        retrieved_expected_source = len(matched_retrievals) > 0
        # Check whether any expected section is cited in the actual answer
        matched_citations: list[str] = []
        for sec in expected_secs:
            pattern = rf"(?:Section|Sec\.?|s\.)\s*{re.escape(sec)}\b"
            if re.search(pattern, actual_output, re.IGNORECASE):
                matched_citations.append(sec)
        cites_expected_source = len(matched_citations) > 0

    # 4. DeepEval LLMTestCase & Metrics
    retrieval_context = [ch.get("text", "") for ch in retrieved_chunks]
    expected_text = case.expected if case.expected else (NOT_FOUND_RESPONSE if case.is_unanswerable else case.question)

    from deepeval.test_case import LLMTestCase

    test_case = LLMTestCase(
        input=case.question,
        actual_output=actual_output,
        expected_output=expected_text,
        retrieval_context=retrieval_context,
    )

    if metrics:
        if case.is_unanswerable:
            if is_not_found:
                recall_res = MetricResult(score=1.0, reason="Out-of-corpus question correctly refused; no context expected.")
                precision_res = MetricResult(score=1.0, reason="Out-of-corpus question correctly refused.")
                faith_res = MetricResult(score=1.0, reason="Model refrained from hallucinating ungrounded claims.")
                relevancy_res = MetricResult(score=1.0, reason="Refusal strictly relevant for out-of-corpus question.")
            else:
                recall_res = MetricResult(score=0.0, reason="Model failed to refuse out-of-corpus question.")
                precision_res = MetricResult(score=0.0, reason="Model hallucinated on out-of-corpus question.")
                faith_res = MetricResult(score=0.0, reason="Ungrounded claims made for out-of-corpus question.")
                relevancy_res = MetricResult(score=0.0, reason="Hallucinated answer for out-of-corpus question.")
        else:
            recall_res = _safe_measure_metric(metrics["recall"], test_case)
            precision_res = _safe_measure_metric(metrics["precision"], test_case)
            faith_res = _safe_measure_metric(metrics["faithfulness"], test_case)
            relevancy_res = _safe_measure_metric(metrics["relevancy"], test_case)
    else:
        recall_res = MetricResult(score=None)
        precision_res = MetricResult(score=None)
        faith_res = MetricResult(score=None)
        relevancy_res = MetricResult(score=None)

    return CaseEvalResult(
        id=case.id,
        question=case.question,
        expected=case.expected,
        actual=actual_output,
        is_unanswerable=case.is_unanswerable,
        latency_seconds=latency,
        cites_expected_source=cites_expected_source,
        retrieved_expected_source=retrieved_expected_source,
        expected_doc=expected_doc,
        expected_sections=expected_secs,
        not_found_refused=not_found_refused,
        contextual_recall=recall_res,
        contextual_precision=precision_res,
        faithfulness=faith_res,
        answer_relevancy=relevancy_res,
        retrieved_sources=retrieved_sources_summary,
        retrieval_mode=mode,
        rerank_enabled=rerank,
        filter_document_enabled=filter_document,
        rewrite_query_enabled=rewrite_query,
        parent_child_enabled=parent_child,
        rewritten_query=str(res.get("rewritten_query") or ""),
        detected_document=str(res.get("detected_document") or ""),
    )


def save_results(
    results: list[CaseEvalResult],
    output_json_path: Path,
    output_csv_path: Path | None,
    cfg: RagConfig,
    mode: str = "vector",
    rerank: bool = False,
) -> None:
    """Save evaluation results to JSON and optional CSV."""
    serializable_results: list[dict[str, Any]] = []
    for r in results:
        d = asdict(r)
        serializable_results.append(d)

    summary_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "retrieval_mode": mode,
        "rerank_enabled": rerank,
        "rerank_model": cfg.rerank_model if rerank else None,
        "total_cases": len(results),
        "chat_model": f"{cfg.chat_provider}/{cfg.chat_model}",
        "embedding_model": f"{cfg.embedding_provider}/{cfg.embedding_model}",
        "results": serializable_results,
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(summary_data, indent=2, ensure_ascii=False), encoding="utf-8")

    if output_csv_path:
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "ID",
                    "Mode",
                    "Rerank",
                    "Type",
                    "Latency (s)",
                    "Retrieved Expected",
                    "Cites Expected",
                    "NOT_FOUND Refused",
                    "Contextual Recall",
                    "Contextual Precision",
                    "Faithfulness",
                    "Answer Relevancy",
                    "Question",
                ]
            )
            for r in results:
                qtype = "Unanswerable" if r.is_unanswerable else "Answerable"
                writer.writerow(
                    [
                        r.id,
                        r.retrieval_mode,
                        "YES" if r.rerank_enabled else "NO",
                        qtype,
                        f"{r.latency_seconds:.2f}",
                        str(r.retrieved_expected_source),
                        str(r.cites_expected_source),
                        str(r.not_found_refused),
                        f"{r.contextual_recall.score:.2f}" if r.contextual_recall.score is not None else "N/A",
                        f"{r.contextual_precision.score:.2f}" if r.contextual_precision.score is not None else "N/A",
                        f"{r.faithfulness.score:.2f}" if r.faithfulness.score is not None else "N/A",
                        f"{r.answer_relevancy.score:.2f}" if r.answer_relevancy.score is not None else "N/A",
                        r.question.replace("\n", " "),
                    ]
                )


def _compute_summary_stats(results: list[CaseEvalResult]) -> dict[str, Any]:
    """Compute aggregate statistics for a list of evaluation results."""
    total = len(results)
    if total == 0:
        return {}

    answerable = [r for r in results if not r.is_unanswerable]
    unanswerable = [r for r in results if r.is_unanswerable]

    latencies = [r.latency_seconds for r in results]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    min_latency = min(latencies) if latencies else 0.0
    max_latency = max(latencies) if latencies else 0.0

    retrieved_hits = sum(1 for r in answerable if r.retrieved_expected_source is True)
    cited_hits = sum(1 for r in answerable if r.cites_expected_source is True)
    retrieved_rate = (retrieved_hits / len(answerable) * 100.0) if answerable else 0.0
    cited_rate = (cited_hits / len(answerable) * 100.0) if answerable else 0.0

    refusal_hits = sum(1 for r in unanswerable if r.not_found_refused is True)
    refusal_rate = (refusal_hits / len(unanswerable) * 100.0) if unanswerable else 0.0

    def _mean_metric(scores: list[float | None]) -> tuple[float | None, int]:
        valid = [s for s in scores if s is not None]
        return (sum(valid) / len(valid) if valid else None, len(valid))

    avg_recall, n_recall = _mean_metric([r.contextual_recall.score for r in answerable])
    avg_precision, n_precision = _mean_metric([r.contextual_precision.score for r in answerable])
    avg_faith, n_faith = _mean_metric([r.faithfulness.score for r in results])
    avg_relevancy, n_relevancy = _mean_metric([r.answer_relevancy.score for r in results])

    return {
        "total": total,
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "avg_latency": avg_latency,
        "min_latency": min_latency,
        "max_latency": max_latency,
        "retrieved_hits": retrieved_hits,
        "retrieved_rate": retrieved_rate,
        "cited_hits": cited_hits,
        "cited_rate": cited_rate,
        "refusal_hits": refusal_hits,
        "refusal_rate": refusal_rate,
        "avg_recall": avg_recall,
        "n_recall": n_recall,
        "avg_precision": avg_precision,
        "n_precision": n_precision,
        "avg_faith": avg_faith,
        "n_faith": n_faith,
        "avg_relevancy": avg_relevancy,
        "n_relevancy": n_relevancy,
    }


def print_scorecard(results: list[CaseEvalResult], cfg: RagConfig, elapsed_total: float, mode: str = "vector") -> None:
    """Print a clean summary scorecard for a single retrieval mode."""
    stats = _compute_summary_stats(results)
    if not stats:
        print("No test cases evaluated.")
        return

    w = 78
    print("\n" + "=" * w)
    print(f"       LEGAL RAG EVALUATION SCORECARD ({mode.upper()} RETRIEVAL)       ")
    print("=" * w)
    print(f"Retrieval Mode     : {mode.upper()}")
    print(f"Chat Model         : {cfg.chat_provider} / {cfg.chat_model}")
    print(f"Embedding Model    : {cfg.embedding_provider} / {cfg.embedding_model}")
    print(f"Total Test Cases   : {stats['total']} ({stats['n_answerable']} answerable, {stats['n_unanswerable']} unanswerable)")
    print(f"Total Time         : {elapsed_total:.2f}s (avg latency: {stats['avg_latency']:.2f}s/q, min: {stats['min_latency']:.2f}s, max: {stats['max_latency']:.2f}s)")
    print("-" * w)
    print("1. RETRIEVAL METRICS (Top Chunks)")
    rec_str = f"{stats['avg_recall']:.4f}" if stats['avg_recall'] is not None else "N/A"
    prec_str = f"{stats['avg_precision']:.4f}" if stats['avg_precision'] is not None else "N/A"
    print(f"  * Contextual Recall     : {rec_str:>8} (evaluated on {stats['n_recall']} cases)")
    print(f"  * Contextual Precision  : {prec_str:>8} (evaluated on {stats['n_precision']} cases)")
    print(f"  * Expected Source In Top: {stats['retrieved_hits']}/{stats['n_answerable']} ({stats['retrieved_rate']:.1f}%)")
    print("-" * w)
    print("2. ANSWER QUALITY & GROUNDING METRICS")
    faith_str = f"{stats['avg_faith']:.4f}" if stats['avg_faith'] is not None else "N/A"
    rel_str = f"{stats['avg_relevancy']:.4f}" if stats['avg_relevancy'] is not None else "N/A"
    print(f"  * Faithfulness          : {faith_str:>8} (hallucination-free adherence, n={stats['n_faith']})")
    print(f"  * Answer Relevancy      : {rel_str:>8} (addressing user inquiry, n={stats['n_relevancy']})")
    print(f"  * Citation Compliance   : {stats['cited_hits']}/{stats['n_answerable']} ({stats['cited_rate']:.1f}% cited expected section)")
    print("-" * w)
    print("3. SAFETY & REFUSAL (NOT_FOUND)")
    print(f"  * NOT_FOUND Refusal Rate: {stats['refusal_hits']}/{stats['n_unanswerable']} ({stats['refusal_rate']:.1f}% properly refused)")
    print("=" * w)

    # Per-question mini table
    print("\nPER-QUESTION RESULTS TABLE:")
    header = f"{'ID':<12} | {'Type':<8} | {'Latency':<8} | {'RetrSrc':<8} | {'CiteSrc':<8} | {'Refused':<8} | {'Recall':<6} | {'Prec':<6} | {'Faith':<6} | {'Relev':<6}"
    print(header)
    print("-" * len(header))
    for r in results:
        qtype = "Unans" if r.is_unanswerable else "Ans"
        lat = f"{r.latency_seconds:.2f}s"
        r_src = "YES" if r.retrieved_expected_source is True else ("NO" if r.retrieved_expected_source is False else "N/A")
        c_src = "YES" if r.cites_expected_source is True else ("NO" if r.cites_expected_source is False else "N/A")
        ref = "PASS" if r.not_found_refused is True else ("FAIL" if r.not_found_refused is False else "N/A")

        rec = f"{r.contextual_recall.score:.2f}" if r.contextual_recall.score is not None else "-"
        prec = f"{r.contextual_precision.score:.2f}" if r.contextual_precision.score is not None else "-"
        fth = f"{r.faithfulness.score:.2f}" if r.faithfulness.score is not None else "-"
        rlv = f"{r.answer_relevancy.score:.2f}" if r.answer_relevancy.score is not None else "-"

        print(f"{r.id:<12} | {qtype:<8} | {lat:<8} | {r_src:<8} | {c_src:<8} | {ref:<8} | {rec:<6} | {prec:<6} | {fth:<6} | {rlv:<6}")
    print("-" * len(header) + "\n")


def print_side_by_side_scorecard(
    v_results: list[CaseEvalResult],
    h_results: list[CaseEvalResult],
    cfg: RagConfig,
    t_v: float,
    t_h: float,
) -> None:
    """Print two scorecards side-by-side comparing Vector-only vs Hybrid retrieval."""
    s_v = _compute_summary_stats(v_results)
    s_h = _compute_summary_stats(h_results)

    def _fmt_score(val: float | None) -> str:
        return f"{val:.4f}" if val is not None else "N/A"

    def _fmt_pct(val: float) -> str:
        return f"{val:.1f}%"

    def _fmt_delta(v_val: float | None, h_val: float | None, is_pct: bool = False) -> str:
        if v_val is None or h_val is None:
            return "N/A"
        diff = h_val - v_val
        prefix = "+" if diff > 0 else ""
        return f"{prefix}{diff:.1f}%" if is_pct else f"{prefix}{diff:.4f}"

    w = 88
    print("\n" + "=" * w)
    print("      LEGAL RAG CHATBOT: VECTOR vs HYBRID RETRIEVAL SIDE-BY-SIDE SCORECARD      ")
    print("=" * w)
    print(f"Chat Model         : {cfg.chat_provider} / {cfg.chat_model}")
    print(f"Embedding Model    : {cfg.embedding_provider} / {cfg.embedding_model}")
    print(f"Evaluation Cases   : {s_v.get('total', 0)} ({s_v.get('n_answerable', 0)} answerable, {s_v.get('n_unanswerable', 0)} unanswerable)")
    print("-" * w)

    col_w_metric = 36
    col_w_val = 15
    header = f"{'Metric / Dimension':<{col_w_metric}} | {'Vector-Only':^{col_w_val}} | {'Hybrid (RRF)':^{col_w_val}} | {'Delta (H - V)':^{col_w_val}}"
    print(header)
    print("-" * w)

    # 1. Retrieval Quality
    print("1. RETRIEVAL QUALITY (Top Chunks)")
    rec_v = s_v.get("avg_recall")
    rec_h = s_h.get("avg_recall")
    print(f"  * Contextual Recall                | { _fmt_score(rec_v):^{col_w_val}} | { _fmt_score(rec_h):^{col_w_val}} | { _fmt_delta(rec_v, rec_h):^{col_w_val}}")

    prec_v = s_v.get("avg_precision")
    prec_h = s_h.get("avg_precision")
    print(f"  * Contextual Precision             | { _fmt_score(prec_v):^{col_w_val}} | { _fmt_score(prec_h):^{col_w_val}} | { _fmt_delta(prec_v, prec_h):^{col_w_val}}")

    retr_v = s_v.get("retrieved_rate", 0.0)
    retr_h = s_h.get("retrieved_rate", 0.0)
    print(f"  * Expected Source In Top-k         | { _fmt_pct(retr_v):^{col_w_val}} | { _fmt_pct(retr_h):^{col_w_val}} | { _fmt_delta(retr_v, retr_h, is_pct=True):^{col_w_val}}")
    print("-" * w)

    # 2. Answer Quality & Grounding
    print("2. ANSWER QUALITY & GROUNDING")
    faith_v = s_v.get("avg_faith")
    faith_h = s_h.get("avg_faith")
    print(f"  * Faithfulness (Hallucination-free)| { _fmt_score(faith_v):^{col_w_val}} | { _fmt_score(faith_h):^{col_w_val}} | { _fmt_delta(faith_v, faith_h):^{col_w_val}}")

    rel_v = s_v.get("avg_relevancy")
    rel_h = s_h.get("avg_relevancy")
    print(f"  * Answer Relevancy                 | { _fmt_score(rel_v):^{col_w_val}} | { _fmt_score(rel_h):^{col_w_val}} | { _fmt_delta(rel_v, rel_h):^{col_w_val}}")

    cite_v = s_v.get("cited_rate", 0.0)
    cite_h = s_h.get("cited_rate", 0.0)
    print(f"  * Citation Compliance              | { _fmt_pct(cite_v):^{col_w_val}} | { _fmt_pct(cite_h):^{col_w_val}} | { _fmt_delta(cite_v, cite_h, is_pct=True):^{col_w_val}}")
    print("-" * w)

    # 3. Safety & Refusal
    print("3. SAFETY & REFUSAL (NOT_FOUND)")
    ref_v = s_v.get("refusal_rate", 0.0)
    ref_h = s_h.get("refusal_rate", 0.0)
    print(f"  * NOT_FOUND Refusal Accuracy       | { _fmt_pct(ref_v):^{col_w_val}} | { _fmt_pct(ref_h):^{col_w_val}} | { _fmt_delta(ref_v, ref_h, is_pct=True):^{col_w_val}}")
    print("-" * w)

    # 4. Latency
    print("4. PERFORMANCE & LATENCY")
    lat_v = s_v.get("avg_latency", 0.0)
    lat_h = s_h.get("avg_latency", 0.0)
    lat_diff = lat_h - lat_v
    lat_diff_str = f"{'+' if lat_diff > 0 else ''}{lat_diff:.2f}s"
    print(f"  * Average Question Latency         | {f'{lat_v:.2f}s':^{col_w_val}} | {f'{lat_h:.2f}s':^{col_w_val}} | {lat_diff_str:^{col_w_val}}")
    print(f"  * Total Evaluation Time            | {f'{t_v:.2f}s':^{col_w_val}} | {f'{t_h:.2f}s':^{col_w_val}} | {f'{t_h - t_v:+.2f}s':^{col_w_val}}")
    print("=" * w)

    # Per-Question Comparison Table
    print("\nPER-QUESTION COMPARISON (Vector vs Hybrid):")
    sub_header = f"{'ID':<11} | {'Type':<7} | {'Vec Retr':<8} | {'Hyb Retr':<8} | {'Vec Cite':<8} | {'Hyb Cite':<8} | {'Vec Rec':<7} | {'Hyb Rec':<7} | {'Vec Prec':<8} | {'Hyb Prec':<8}"
    print(sub_header)
    print("-" * len(sub_header))

    h_map = {r.id: r for r in h_results}
    for rv in v_results:
        rh = h_map.get(rv.id)
        if not rh:
            continue
        qtype = "Unans" if rv.is_unanswerable else "Ans"
        vr = "YES" if rv.retrieved_expected_source is True else ("NO" if rv.retrieved_expected_source is False else "N/A")
        hr = "YES" if rh.retrieved_expected_source is True else ("NO" if rh.retrieved_expected_source is False else "N/A")
        vc = "YES" if rv.cites_expected_source is True else ("NO" if rv.cites_expected_source is False else "N/A")
        hc = "YES" if rh.cites_expected_source is True else ("NO" if rh.cites_expected_source is False else "N/A")

        v_rec = f"{rv.contextual_recall.score:.2f}" if rv.contextual_recall.score is not None else "-"
        h_rec = f"{rh.contextual_recall.score:.2f}" if rh.contextual_recall.score is not None else "-"
        v_pre = f"{rv.contextual_precision.score:.2f}" if rv.contextual_precision.score is not None else "-"
        h_pre = f"{rh.contextual_precision.score:.2f}" if rh.contextual_precision.score is not None else "-"

        print(f"{rv.id:<11} | {qtype:<7} | {vr:<8} | {hr:<8} | {vc:<8} | {hc:<8} | {v_rec:<7} | {h_rec:<7} | {v_pre:<8} | {h_pre:<8}")
    print("-" * len(sub_header) + "\n")


def print_rerank_comparison_scorecard(
    base_results: list[CaseEvalResult],
    rerank_results: list[CaseEvalResult],
    cfg: RagConfig,
    t_base: float,
    t_rerank: float,
    mode: str = "hybrid",
) -> None:
    """Print two scorecards side-by-side comparing retrieval with and without cross-encoder re-ranking."""
    s_b = _compute_summary_stats(base_results)
    s_r = _compute_summary_stats(rerank_results)

    def _fmt_score(val: float | None) -> str:
        return f"{val:.4f}" if val is not None else "N/A"

    def _fmt_pct(val: float) -> str:
        return f"{val:.1f}%"

    def _fmt_delta(b_val: float | None, r_val: float | None, is_pct: bool = False) -> str:
        if b_val is None or r_val is None:
            return "N/A"
        diff = r_val - b_val
        prefix = "+" if diff > 0 else ""
        return f"{prefix}{diff:.1f}%" if is_pct else f"{prefix}{diff:.4f}"

    w = 92
    print("\n" + "=" * w)
    print("   LEGAL RAG CHATBOT: WITHOUT RE-RANKING vs WITH CROSS-ENCODER RE-RANKING   ")
    print("=" * w)
    print(f"Retrieval Mode (Stage 1) : {mode.upper()} (Top 30 initial candidates retrieved)")
    print(f"Re-ranking Model (Stage 2) : {cfg.rerank_model} (Top 5 selected)")
    print(f"Chat Model                 : {cfg.chat_provider} / {cfg.chat_model}")
    print(f"Embedding Model            : {cfg.embedding_provider} / {cfg.embedding_model}")
    print(f"Evaluation Cases           : {s_b.get('total', 0)} ({s_b.get('n_answerable', 0)} answerable, {s_b.get('n_unanswerable', 0)} unanswerable)")
    print("-" * w)

    col_w_metric = 36
    col_w_val = 16
    header = f"{'Metric / Dimension':<{col_w_metric}} | {'Without Re-rank':^{col_w_val}} | {'With Re-ranker':^{col_w_val}} | {'Delta (R - W)':^{col_w_val}}"
    print(header)
    print("-" * w)

    # 1. Retrieval Quality
    print("1. RETRIEVAL QUALITY (Top Chunks)")
    rec_b, rec_r = s_b.get("avg_recall"), s_r.get("avg_recall")
    print(f"  * Contextual Recall                | { _fmt_score(rec_b):^{col_w_val}} | { _fmt_score(rec_r):^{col_w_val}} | { _fmt_delta(rec_b, rec_r):^{col_w_val}}")

    prec_b, prec_r = s_b.get("avg_precision"), s_r.get("avg_precision")
    print(f"  * Contextual Precision             | { _fmt_score(prec_b):^{col_w_val}} | { _fmt_score(prec_r):^{col_w_val}} | { _fmt_delta(prec_b, prec_r):^{col_w_val}}")

    retr_b, retr_r = s_b.get("retrieved_rate", 0.0), s_r.get("retrieved_rate", 0.0)
    print(f"  * Expected Source In Top-k         | { _fmt_pct(retr_b):^{col_w_val}} | { _fmt_pct(retr_r):^{col_w_val}} | { _fmt_delta(retr_b, retr_r, is_pct=True):^{col_w_val}}")
    print("-" * w)

    # 2. Answer Quality & Grounding
    print("2. ANSWER QUALITY & GROUNDING")
    faith_b, faith_r = s_b.get("avg_faith"), s_r.get("avg_faith")
    print(f"  * Faithfulness (Hallucination-free)| { _fmt_score(faith_b):^{col_w_val}} | { _fmt_score(faith_r):^{col_w_val}} | { _fmt_delta(faith_b, faith_r):^{col_w_val}}")

    rel_b, rel_r = s_b.get("avg_relevancy"), s_r.get("avg_relevancy")
    print(f"  * Answer Relevancy                 | { _fmt_score(rel_b):^{col_w_val}} | { _fmt_score(rel_r):^{col_w_val}} | { _fmt_delta(rel_b, rel_r):^{col_w_val}}")

    cite_b, cite_r = s_b.get("cited_rate", 0.0), s_r.get("cited_rate", 0.0)
    print(f"  * Citation Compliance              | { _fmt_pct(cite_b):^{col_w_val}} | { _fmt_pct(cite_r):^{col_w_val}} | { _fmt_delta(cite_b, cite_r, is_pct=True):^{col_w_val}}")
    print("-" * w)

    # 3. Safety & Refusal
    print("3. SAFETY & REFUSAL (NOT_FOUND)")
    ref_b, ref_r = s_b.get("refusal_rate", 0.0), s_r.get("refusal_rate", 0.0)
    print(f"  * NOT_FOUND Refusal Accuracy       | { _fmt_pct(ref_b):^{col_w_val}} | { _fmt_pct(ref_r):^{col_w_val}} | { _fmt_delta(ref_b, ref_r, is_pct=True):^{col_w_val}}")
    print("-" * w)

    # 4. Latency
    print("4. PERFORMANCE & LATENCY (Response Time Change)")
    lat_b = s_b.get("avg_latency", 0.0)
    lat_r = s_r.get("avg_latency", 0.0)
    lat_diff = lat_r - lat_b
    lat_diff_str = f"{'+' if lat_diff > 0 else ''}{lat_diff:.2f}s"
    print(f"  * Average Question Latency         | {f'{lat_b:.2f}s':^{col_w_val}} | {f'{lat_r:.2f}s':^{col_w_val}} | {lat_diff_str:^{col_w_val}}")
    print(f"  * Total Evaluation Time            | {f'{t_base:.2f}s':^{col_w_val}} | {f'{t_rerank:.2f}s':^{col_w_val}} | {f'{t_rerank - t_base:+.2f}s':^{col_w_val}}")
    print("=" * w)

    # Per-Question Comparison Table
    print("\nPER-QUESTION COMPARISON (Without vs With Re-ranking):")
    sub_header = f"{'ID':<11} | {'Type':<6} | {'BaseRetr':<8} | {'RrkRetr':<8} | {'BaseCite':<8} | {'RrkCite':<8} | {'BaseLat':<8} | {'RrkLat':<8} | {'LatChg':<8} | {'BaseRec':<7} | {'RrkRec':<7}"
    print(sub_header)
    print("-" * len(sub_header))

    r_map = {r.id: r for r in rerank_results}
    for rb in base_results:
        rr = r_map.get(rb.id)
        if not rr:
            continue
        qtype = "Unans" if rb.is_unanswerable else "Ans"
        br = "YES" if rb.retrieved_expected_source is True else ("NO" if rb.retrieved_expected_source is False else "N/A")
        rr_retr = "YES" if rr.retrieved_expected_source is True else ("NO" if rr.retrieved_expected_source is False else "N/A")
        bc = "YES" if rb.cites_expected_source is True else ("NO" if rb.cites_expected_source is False else "N/A")
        rc = "YES" if rr.cites_expected_source is True else ("NO" if rr.cites_expected_source is False else "N/A")

        blat = f"{rb.latency_seconds:.2f}s"
        rlat = f"{rr.latency_seconds:.2f}s"
        dlat = rr.latency_seconds - rb.latency_seconds
        dlat_str = f"{'+' if dlat > 0 else ''}{dlat:.2f}s"

        b_rec = f"{rb.contextual_recall.score:.2f}" if rb.contextual_recall.score is not None else "-"
        r_rec = f"{rr.contextual_recall.score:.2f}" if rr.contextual_recall.score is not None else "-"

        print(f"{rb.id:<11} | {qtype:<6} | {br:<8} | {rr_retr:<8} | {bc:<8} | {rc:<8} | {blat:<8} | {rlat:<8} | {dlat_str:<8} | {b_rec:<7} | {r_rec:<7}")
    print("-" * len(sub_header) + "\n")


def print_comparison_scorecard(
    prev_results: list[CaseEvalResult],
    new_results: list[CaseEvalResult],
    cfg: RagConfig,
    t_prev: float,
    t_new: float,
    prev_title: str = "Previous (Re-ranker)",
    new_title: str = "New (Filter+Rewrite)",
) -> None:
    """Print two scorecards side-by-side comparing the previous system against the improved system."""
    s_p = _compute_summary_stats(prev_results)
    s_n = _compute_summary_stats(new_results)

    def _fmt_score(val: float | None) -> str:
        return f"{val:.4f}" if val is not None else "N/A"

    def _fmt_pct(val: float) -> str:
        return f"{val:.1f}%"

    def _fmt_delta(p_val: float | None, n_val: float | None, is_pct: bool = False) -> str:
        if p_val is None or n_val is None:
            return "N/A"
        diff = n_val - p_val
        prefix = "+" if diff > 0 else ""
        return f"{prefix}{diff:.1f}%" if is_pct else f"{prefix}{diff:.4f}"

    w = 94
    print("\n" + "=" * w)
    print(f"   LEGAL RAG CHATBOT: {prev_title.upper()} vs {new_title.upper()} (SIDE-BY-SIDE)   ")
    print("=" * w)
    print(f"Chat Model         : {cfg.chat_provider} / {cfg.chat_model}")
    print(f"Embedding Model    : {cfg.embedding_provider} / {cfg.embedding_model}")
    print(f"Evaluation Cases   : {s_p.get('total', 0)} ({s_p.get('n_answerable', 0)} answerable, {s_p.get('n_unanswerable', 0)} unanswerable)")
    print(f"Pipeline Config    : Hybrid Search [ON], Cross-Encoder [ON], Metadata Filter [ON], Query Rewriter [ON]")
    print("-" * w)

    col_w_metric = 36
    col_w_val = 17
    header = f"{'Metric / Dimension':<{col_w_metric}} | {prev_title:^{col_w_val}} | {new_title:^{col_w_val}} | {'Delta (New - Prev)':^{col_w_val}}"
    print(header)
    print("-" * w)

    # 1. Retrieval Quality
    print("1. RETRIEVAL QUALITY (Top Chunks)")
    rec_p, rec_n = s_p.get("avg_recall"), s_n.get("avg_recall")
    print(f"  * Contextual Recall                | { _fmt_score(rec_p):^{col_w_val}} | { _fmt_score(rec_n):^{col_w_val}} | { _fmt_delta(rec_p, rec_n):^{col_w_val}}")

    prec_p, prec_n = s_p.get("avg_precision"), s_n.get("avg_precision")
    print(f"  * Contextual Precision             | { _fmt_score(prec_p):^{col_w_val}} | { _fmt_score(prec_n):^{col_w_val}} | { _fmt_delta(prec_p, prec_n):^{col_w_val}}")

    retr_p, retr_n = s_p.get("retrieved_rate", 0.0), s_n.get("retrieved_rate", 0.0)
    print(f"  * Expected Source In Top-k         | { _fmt_pct(retr_p):^{col_w_val}} | { _fmt_pct(retr_n):^{col_w_val}} | { _fmt_delta(retr_p, retr_n, is_pct=True):^{col_w_val}}")
    print("-" * w)

    # 2. Answer Quality & Grounding
    print("2. ANSWER QUALITY & GROUNDING")
    faith_p, faith_n = s_p.get("avg_faith"), s_n.get("avg_faith")
    print(f"  * Faithfulness (Hallucination-free)| { _fmt_score(faith_p):^{col_w_val}} | { _fmt_score(faith_n):^{col_w_val}} | { _fmt_delta(faith_p, faith_n):^{col_w_val}}")

    rel_p, rel_n = s_p.get("avg_relevancy"), s_n.get("avg_relevancy")
    print(f"  * Answer Relevancy                 | { _fmt_score(rel_p):^{col_w_val}} | { _fmt_score(rel_n):^{col_w_val}} | { _fmt_delta(rel_p, rel_n):^{col_w_val}}")

    cite_p, cite_n = s_p.get("cited_rate", 0.0), s_n.get("cited_rate", 0.0)
    print(f"  * Citation Compliance              | { _fmt_pct(cite_p):^{col_w_val}} | { _fmt_pct(cite_n):^{col_w_val}} | { _fmt_delta(cite_p, cite_n, is_pct=True):^{col_w_val}}")
    print("-" * w)

    # 3. Safety & Refusal
    print("3. SAFETY & REFUSAL (NOT_FOUND)")
    ref_p, ref_n = s_p.get("refusal_rate", 0.0), s_n.get("refusal_rate", 0.0)
    print(f"  * NOT_FOUND Refusal Accuracy       | { _fmt_pct(ref_p):^{col_w_val}} | { _fmt_pct(ref_n):^{col_w_val}} | { _fmt_delta(ref_p, ref_n, is_pct=True):^{col_w_val}}")
    print("-" * w)

    # 4. Latency
    print("4. PERFORMANCE & LATENCY (Response Time Change)")
    lat_p = s_p.get("avg_latency", 0.0)
    lat_n = s_n.get("avg_latency", 0.0)
    lat_diff = lat_n - lat_p
    lat_diff_str = f"{'+' if lat_diff > 0 else ''}{lat_diff:.2f}s"
    print(f"  * Average Question Latency         | {f'{lat_p:.2f}s':^{col_w_val}} | {f'{lat_n:.2f}s':^{col_w_val}} | {lat_diff_str:^{col_w_val}}")
    print(f"  * Total Evaluation Time            | {f'{t_prev:.2f}s':^{col_w_val}} | {f'{t_new:.2f}s':^{col_w_val}} | {f'{t_new - t_prev:+.2f}s':^{col_w_val}}")
    print("=" * w)

    # Per-Question Comparison Table
    print("\nPER-QUESTION COMPARISON (Previous vs New Improved):")
    sub_header = f"{'ID':<11} | {'Type':<6} | {'PrevRetr':<8} | {'NewRetr':<8} | {'PrevCite':<8} | {'NewCite':<8} | {'PrevLat':<8} | {'NewLat':<8} | {'LatChg':<8} | {'PrevRec':<7} | {'NewRec':<7}"
    print(sub_header)
    print("-" * len(sub_header))

    n_map = {r.id: r for r in new_results}
    for rp in prev_results:
        rn = n_map.get(rp.id)
        if not rn:
            continue
        qtype = "Unans" if rp.is_unanswerable else "Ans"
        pr_retr = "YES" if rp.retrieved_expected_source is True else ("NO" if rp.retrieved_expected_source is False else "N/A")
        nr_retr = "YES" if rn.retrieved_expected_source is True else ("NO" if rn.retrieved_expected_source is False else "N/A")
        pc = "YES" if rp.cites_expected_source is True else ("NO" if rp.cites_expected_source is False else "N/A")
        nc = "YES" if rn.cites_expected_source is True else ("NO" if rn.cites_expected_source is False else "N/A")

        plat = f"{rp.latency_seconds:.2f}s"
        nlat = f"{rn.latency_seconds:.2f}s"
        dlat = rn.latency_seconds - rp.latency_seconds
        dlat_str = f"{'+' if dlat > 0 else ''}{dlat:.2f}s"

        p_rec = f"{rp.contextual_recall.score:.2f}" if rp.contextual_recall.score is not None else "-"
        n_rec = f"{rn.contextual_recall.score:.2f}" if rn.contextual_recall.score is not None else "-"

        print(f"{rp.id:<11} | {qtype:<6} | {pr_retr:<8} | {nr_retr:<8} | {pc:<8} | {nc:<8} | {plat:<8} | {nlat:<8} | {dlat_str:<8} | {p_rec:<7} | {n_rec:<7}")
    print("-" * len(sub_header) + "\n")


def run_evaluation_suite(
    cases: list[EvalCase],
    cfg: RagConfig,
    metrics: dict[str, Any] | None,
    mode: str,
    top_k: int,
    output_path: Path,
    csv_output_path: Path | None,
    rerank: bool = False,
    rerank_candidates: int = 30,
    filter_document: bool = False,
    rewrite_query: bool = False,
    parent_child: bool = False,
) -> tuple[list[CaseEvalResult], float]:
    """Execute evaluation for one configuration run."""
    rerank_str = " + CrossEncoder" if rerank else ""
    filter_str = " + DocFilter" if filter_document else ""
    rewrite_str = " + Rewriter" if rewrite_query else ""
    pc_str = " + ParentChild" if parent_child else ""
    print(f"\n>>> Running Evaluation in [{mode.upper()}{rerank_str}{filter_str}{rewrite_str}{pc_str}] ({len(cases)} cases) <<<")
    t_start = time.perf_counter()
    results: list[CaseEvalResult] = []

    for idx, case in enumerate(cases, start=1):
        print(f"[{idx}/{len(cases)}] [{mode.upper()}{'+RERANK' if rerank else ''}{'+PC' if parent_child else ''}] {case.id}: {case.question[:60]}...")
        res = evaluate_single_case(
            case,
            top_k=top_k,
            config=cfg,
            metrics=metrics,
            mode=mode,
            rerank=rerank,
            rerank_candidates=rerank_candidates,
            filter_document=filter_document,
            rewrite_query=rewrite_query,
            parent_child=parent_child,
        )
        results.append(res)
        rec_txt = f"recall={res.contextual_recall.score:.2f}" if res.contextual_recall.score is not None else ""
        faith_txt = f"faith={res.faithfulness.score:.2f}" if res.faithfulness.score is not None else ""
        print(f"       -> Done in {res.latency_seconds:.2f}s | Cites: {res.cites_expected_source} | Refused: {res.not_found_refused} {rec_txt} {faith_txt}")

    elapsed = time.perf_counter() - t_start
    save_results(results, output_path, csv_output_path, cfg, mode=mode, rerank=rerank)
    print(f"Results saved to: {output_path.resolve()}")
    return results, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG chatbot with Vector vs Hybrid retrieval.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("CA_RAG_Golden_Eval_Dataset_Notepad.txt"),
        help="Path to golden dataset file (default: CA_RAG_Golden_Eval_Dataset_Notepad.txt).",
    )
    parser.add_argument(
        "--unanswerable-dataset",
        type=Path,
        default=Path("synthetic_golden_dataset.json"),
        help="Path to synthetic unanswerable dataset (default: synthetic_golden_dataset.json).",
    )
    parser.add_argument(
        "--no-unanswerable",
        action="store_true",
        help="Skip unanswerable questions.",
    )
    parser.add_argument(
        "--case-ids",
        nargs="+",
        default=None,
        help="Specific case IDs to evaluate (e.g. CA2013-001 CA2013-004 SYN-001).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of test cases to evaluate.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per question (default: 5).",
    )
    parser.add_argument(
        "--compare",
        type=str,
        choices=["parent_child", "improvements", "rerank", "mode", "none"],
        default="parent_child",
        help="Comparison benchmark: 'parent_child' (Before vs After Parent-Child), 'improvements' (Previous Re-ranker vs New Filter+Rewriter), 'rerank' (with/without rerank), 'mode' (vector vs hybrid), or 'none' (default: parent_child).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["vector", "hybrid", "both"],
        default="hybrid",
        help="Retrieval search mode: 'vector', 'hybrid', or 'both' (default: hybrid).",
    )
    parser.add_argument(
        "--rerank",
        type=str,
        choices=["false", "true", "both"],
        default="true",
        help="Re-ranking flag: 'false' (disabled), 'true' (enabled), or 'both' (default: true).",
    )
    parser.add_argument(
        "--filter-doc",
        type=str,
        choices=["true", "false"],
        default="true",
        help="Document/version metadata filtering: 'true' or 'false' (default: true).",
    )
    parser.add_argument(
        "--rewrite-query",
        type=str,
        choices=["true", "false"],
        default="true",
        help="Query rewriting: 'true' or 'false' (default: true).",
    )
    parser.add_argument(
        "--parent-child",
        type=str,
        choices=["true", "false"],
        default="true",
        help="Parent-Child expansion: 'true' or 'false' (default: true).",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=30,
        help="Initial candidates to retrieve before cross-encoder re-ranking (default: 30).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_results.json"),
        help="Path to save detailed JSON evaluation results.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("eval_results.csv"),
        help="Path to save summary CSV.",
    )
    parser.add_argument(
        "--eval-model",
        type=str,
        default=None,
        help="DeepEval judge LLM model name (default: gemini-3.5-flash-lite).",
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Skip LLM judge metrics (fast execution for testing retrieval/citations/refusals/timing).",
    )
    args = parser.parse_args()

    cfg = load_rag_config(
        retrieval_mode=args.mode if args.mode != "both" else "hybrid",
        rerank_candidates=args.candidates,
        filter_document=args.filter_doc == "true",
        rewrite_query=args.rewrite_query == "true",
        parent_child=args.parent_child == "true",
    )
    print(f"Loading evaluation dataset from {args.dataset}...")
    cases = load_evaluation_dataset(
        dataset_path=args.dataset,
        unanswerable_path=args.unanswerable_dataset,
        include_unanswerable=not args.no_unanswerable,
    )

    if args.case_ids:
        case_id_set = set(args.case_ids)
        cases = [c for c in cases if c.id in case_id_set]

    if args.limit:
        cases = cases[: args.limit]

    print(f"Loaded {len(cases)} cases for evaluation.")

    # Initialize DeepEval judge metrics if not skipped
    metrics: dict[str, Any] | None = None
    if not args.skip_metrics:
        print("Initializing DeepEval LLM judge model...")
        judge_model = _init_eval_model(cfg, model_name=args.eval_model)
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            FaithfulnessMetric,
        )

        metrics = {
            "recall": ContextualRecallMetric(model=judge_model, async_mode=False),
            "precision": ContextualPrecisionMetric(model=judge_model, async_mode=False),
            "faithfulness": FaithfulnessMetric(model=judge_model, async_mode=False),
            "relevancy": AnswerRelevancyMetric(model=judge_model, async_mode=False),
        }
        print("DeepEval metrics ready: ContextualRecall, ContextualPrecision, Faithfulness, AnswerRelevancy.\n")

    if args.compare == "parent_child":
        eval_mode = "hybrid" if args.mode == "both" else args.mode

        # 1. Run System BEFORE Parent-Child (Child-Only: Hybrid + CrossEncoder + DocFilter + Rewriter, parent_child=False)
        before_out_json = args.output.with_name(f"{args.output.stem}_before_pc.json")
        before_out_csv = args.csv_output.with_name(f"{args.csv_output.stem}_before_pc.csv") if args.csv_output else None
        before_results, t_before = run_evaluation_suite(
            cases,
            cfg,
            metrics,
            mode=eval_mode,
            top_k=args.top_k,
            output_path=before_out_json,
            csv_output_path=before_out_csv,
            rerank=True,
            rerank_candidates=args.candidates,
            filter_document=True,
            rewrite_query=True,
            parent_child=False,
        )

        # 2. Run System AFTER Parent-Child (Hybrid + CrossEncoder + DocFilter + Rewriter + Parent-Child Expansion)
        after_out_json = args.output.with_name(f"{args.output.stem}_parent_child.json")
        after_out_csv = args.csv_output.with_name(f"{args.csv_output.stem}_parent_child.csv") if args.csv_output else None
        after_results, t_after = run_evaluation_suite(
            cases,
            cfg,
            metrics,
            mode=eval_mode,
            top_k=args.top_k,
            output_path=after_out_json,
            csv_output_path=after_out_csv,
            rerank=True,
            rerank_candidates=args.candidates,
            filter_document=True,
            rewrite_query=True,
            parent_child=True,
        )

        # 3. Print individual scorecards and side-by-side comparison
        print_scorecard(before_results, cfg, t_before, mode=f"{eval_mode} (BEFORE: CHILD-ONLY RETRIEVAL)")
        print_scorecard(after_results, cfg, t_after, mode=f"{eval_mode} (AFTER: PARENT-CHILD RETRIEVAL)")
        print_comparison_scorecard(
            before_results,
            after_results,
            cfg,
            t_before,
            t_after,
            prev_title="Before (Child-Only)",
            new_title="After (Parent-Child)",
        )

    elif args.compare == "improvements":
        eval_mode = "hybrid" if args.mode == "both" else args.mode

        # 1. Run Previous System (Hybrid + Cross-Encoder Re-ranker, NO doc-filter, NO rewriter)
        prev_out_json = args.output.with_name(f"{args.output.stem}_prev.json")
        prev_out_csv = args.csv_output.with_name(f"{args.csv_output.stem}_prev.csv") if args.csv_output else None
        prev_results, t_prev = run_evaluation_suite(
            cases,
            cfg,
            metrics,
            mode=eval_mode,
            top_k=args.top_k,
            output_path=prev_out_json,
            csv_output_path=prev_out_csv,
            rerank=True,
            rerank_candidates=args.candidates,
            filter_document=False,
            rewrite_query=False,
        )

        # 2. Run New Improved System (Hybrid + Cross-Encoder + Doc/Version Filter + Query Rewriter)
        new_out_json = args.output.with_name(f"{args.output.stem}_improved.json")
        new_out_csv = args.csv_output.with_name(f"{args.csv_output.stem}_improved.csv") if args.csv_output else None
        new_results, t_new = run_evaluation_suite(
            cases,
            cfg,
            metrics,
            mode=eval_mode,
            top_k=args.top_k,
            output_path=new_out_json,
            csv_output_path=new_out_csv,
            rerank=True,
            rerank_candidates=args.candidates,
            filter_document=True,
            rewrite_query=True,
        )

        # 3. Print individual scorecards and side-by-side comparison
        print_scorecard(prev_results, cfg, t_prev, mode=f"{eval_mode} (PREVIOUS: RE-RANKER ONLY)")
        print_scorecard(new_results, cfg, t_new, mode=f"{eval_mode} (NEW: FILTER + REWRITE + RE-RANK)")
        print_comparison_scorecard(
            prev_results,
            new_results,
            cfg,
            t_prev,
            t_new,
            prev_title="Previous (Rerank)",
            new_title="New (Filter+Rewrite)",
        )

    elif args.compare == "rerank" or args.rerank == "both":
        eval_mode = "hybrid" if args.mode == "both" else args.mode

        # 1. Run Without Re-ranking
        base_out_json = args.output.with_name(f"{args.output.stem}_no_rerank.json")
        base_out_csv = args.csv_output.with_name(f"{args.csv_output.stem}_no_rerank.csv") if args.csv_output else None
        base_results, t_base = run_evaluation_suite(
            cases,
            cfg,
            metrics,
            mode=eval_mode,
            top_k=args.top_k,
            output_path=base_out_json,
            csv_output_path=base_out_csv,
            rerank=False,
        )

        # 2. Run With Cross-Encoder Re-ranking
        rerank_out_json = args.output.with_name(f"{args.output.stem}_rerank.json")
        rerank_out_csv = args.csv_output.with_name(f"{args.csv_output.stem}_rerank.csv") if args.csv_output else None
        rerank_results, t_rerank = run_evaluation_suite(
            cases,
            cfg,
            metrics,
            mode=eval_mode,
            top_k=args.top_k,
            output_path=rerank_out_json,
            csv_output_path=rerank_out_csv,
            rerank=True,
            rerank_candidates=args.candidates,
        )

        # 3. Print individual and side-by-side scorecards
        print_scorecard(base_results, cfg, t_base, mode=f"{eval_mode} (NO RE-RANK)")
        print_scorecard(rerank_results, cfg, t_rerank, mode=f"{eval_mode} (CROSS-ENCODER RE-RANK)")
        print_rerank_comparison_scorecard(base_results, rerank_results, cfg, t_base, t_rerank, mode=eval_mode)

    elif args.compare == "mode" or args.mode == "both":
        # 1. Run Vector mode
        v_out_json = args.output.with_name(f"{args.output.stem}_vector.json")
        v_out_csv = args.csv_output.with_name(f"{args.csv_output.stem}_vector.csv") if args.csv_output else None
        v_results, t_vector = run_evaluation_suite(
            cases, cfg, metrics, mode="vector", top_k=args.top_k, output_path=v_out_json, csv_output_path=v_out_csv
        )

        # 2. Run Hybrid mode
        h_out_json = args.output.with_name(f"{args.output.stem}_hybrid.json")
        h_out_csv = args.csv_output.with_name(f"{args.csv_output.stem}_hybrid.csv") if args.csv_output else None
        h_results, t_hybrid = run_evaluation_suite(
            cases, cfg, metrics, mode="hybrid", top_k=args.top_k, output_path=h_out_json, csv_output_path=h_out_csv
        )

        # 3. Print side-by-side scorecard
        print_side_by_side_scorecard(v_results, h_results, cfg, t_vector, t_hybrid)
    else:
        use_rerank = args.rerank == "true"
        use_filter = args.filter_doc == "true"
        use_rewrite = args.rewrite_query == "true"
        results, elapsed = run_evaluation_suite(
            cases,
            cfg,
            metrics,
            mode=args.mode,
            top_k=args.top_k,
            output_path=args.output,
            csv_output_path=args.csv_output,
            rerank=use_rerank,
            rerank_candidates=args.candidates,
            filter_document=use_filter,
            rewrite_query=use_rewrite,
        )
        tag = f"{args.mode} (rerank={use_rerank}, filter={use_filter}, rewrite={use_rewrite})"
        print_scorecard(results, cfg, elapsed, mode=tag)


if __name__ == "__main__":
    main()
