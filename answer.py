"""RAG answer generator with deterministic grounded citations and NOT_FOUND refusal.

Retrieves the top-k chunks via `retrieve.py`, labels each chunk with a citation
built from its metadata (document, section/unit, page), and prompts the chat
model deterministically (temperature=0.0, seed=42) to answer strictly from the
retrieved chunks or return `'I could not find this in the documents.'`.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any

from rag_config import RagConfig, get_secret, load_rag_config
from retrieve import retrieve


NOT_FOUND_RESPONSE = "I could not find this in the documents."

SYSTEM_PROMPT = (
    "You are a strict Indian legal and tax RAG assistant.\n"
    "Rules you MUST obey without exception:\n"
    "1. Answer ONLY using facts explicitly stated in the provided CONTEXT CHUNKS.\n"
    "2. Do NOT use outside legal knowledge, assumptions, or unstated thresholds/dates.\n"
    "3. Cite the exact chunk citation label (e.g., [Companies Act, 2013 | Section 135 | p.93]) "
    "immediately after every claim or sentence supported by that chunk.\n"
    "4. If the provided CONTEXT CHUNKS do not contain enough information to answer the question, "
    f"reply with this exact sentence and nothing else:\n{NOT_FOUND_RESPONSE}"
)


def format_citation(metadata: dict[str, Any]) -> str:
    """Build a human- and eval-readable citation label from chunk metadata."""
    doc = str(metadata.get("document_name") or metadata.get("source_file") or "Unknown Document")
    unit_type = str(metadata.get("unit_type") or "")
    unit_num = str(metadata.get("unit_number") or "")

    if unit_type == "section" and unit_num:
        unit_label = f"Section {unit_num}"
    elif unit_num:
        unit_label = unit_num
    else:
        unit_label = unit_type or "Excerpt"

    page = metadata.get("page")
    page_end = metadata.get("page_end")
    if page is not None and page_end is not None and page_end != page:
        page_label = f"p.{page}-{page_end}"
    elif page is not None:
        page_label = f"p.{page}"
    else:
        page_label = "p.?"

    return f"[{doc} | {unit_label} | {page_label}]"


def _call_chat_model(config: RagConfig, system_prompt: str, user_prompt: str) -> str:
    """Invoke the configured chat model deterministically."""
    provider = config.chat_provider
    model = config.chat_model

    if provider == "google":
        from google import genai
        from google.genai import types

        api_key = get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required for chat_provider='google'.")
        client = genai.Client(api_key=api_key)
        for attempt in range(4):
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.0,
                        top_p=1.0,
                        seed=42,
                    ),
                )
                return (resp.text or "").strip()
            except Exception as exc:
                msg = str(exc)
                if ("503" in msg or "UNAVAILABLE" in msg or "429" in msg) and attempt < 3:
                    time.sleep(3.0 * (attempt + 1))
                    continue
                raise

    if provider == "openai":
        from openai import OpenAI

        api_key = get_secret("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for chat_provider='openai'.")
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            seed=42,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (resp.choices[0].message.content or "").strip()

    if provider == "ollama":
        import ollama

        client = ollama.Client(host=config.ollama_base_url)
        resp = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": 0.0, "seed": 42},
        )
        return (resp["message"]["content"] or "").strip()

    raise ValueError(f"Unsupported chat provider: {provider}")


def core_answer(
    question: str,
    top_k: int = 5,
    config: RagConfig | None = None,
    mode: str | None = None,
    rerank: bool | None = None,
    rerank_candidates: int | None = None,
    filter_document: bool | None = None,
    rewrite_query: bool | None = None,
    parent_child: bool | None = None,
    allowed_documents: list[str] | set[str] | None = None,
) -> dict[str, Any]:
    """Retrieve top-k chunks, label citations, and generate a deterministic grounded answer."""
    cfg = config or load_rag_config()
    chosen_mode = (mode or cfg.retrieval_mode or "vector").strip().lower()
    use_rerank = cfg.rerank_enabled if rerank is None else bool(rerank)
    use_filter = cfg.filter_document if filter_document is None else bool(filter_document)
    use_rewrite = cfg.rewrite_query if rewrite_query is None else bool(rewrite_query)
    use_parent_child = cfg.parent_child if parent_child is None else bool(parent_child)

    retrieved_chunks = retrieve(
        question,
        top_k=top_k,
        config=cfg,
        mode=chosen_mode,
        rerank=use_rerank,
        rerank_candidates=rerank_candidates,
        filter_document=use_filter,
        rewrite_query=use_rewrite,
        parent_child=use_parent_child,
        allowed_documents=allowed_documents,
    )

    labeled_chunks: list[dict[str, Any]] = []
    context_blocks: list[str] = []
    rewritten_query = question
    was_rewritten = False
    detected_doc = None
    detected_ver = None
    is_older = False

    for chunk in retrieved_chunks:
        citation = format_citation(chunk["metadata"])
        enriched = {**chunk, "citation": citation}
        labeled_chunks.append(enriched)
        context_blocks.append(f"CITATION LABEL: {citation}\n{chunk['text']}")
        rewritten_query = chunk.get("search_query", rewritten_query)
        was_rewritten = chunk.get("was_rewritten", was_rewritten)
        detected_doc = chunk.get("target_document", detected_doc)
        detected_ver = chunk.get("target_version", detected_ver)
        is_older = chunk.get("is_older_requested", is_older)

    if not labeled_chunks:
        return {
            "question": question,
            "answer": NOT_FOUND_RESPONSE,
            "not_found": True,
            "chunks": [],
            "retrieval_mode": chosen_mode,
            "rerank_enabled": use_rerank,
            "rerank_model": cfg.rerank_model if use_rerank else None,
            "filter_document_enabled": use_filter,
            "rewrite_query_enabled": use_rewrite,
            "parent_child_enabled": use_parent_child,
            "detected_document": detected_doc,
            "detected_version": detected_ver,
            "rewritten_query": rewritten_query,
            "was_rewritten": was_rewritten,
            "chat_model": f"{cfg.chat_provider}/{cfg.chat_model}",
            "embedding_model": f"{cfg.embedding_provider}/{cfg.embedding_model}",
        }

    user_prompt = (
        "CONTEXT CHUNKS:\n\n"
        + "\n\n---\n\n".join(context_blocks)
        + f"\n\nQUESTION: {question}\n\n"
        f"Answer strictly from the CONTEXT CHUNKS above, citing the exact CITATION LABEL after every claim, "
        f"or reply exactly '{NOT_FOUND_RESPONSE}' if the chunks do not contain the answer."
    )

    answer_text = _call_chat_model(cfg, SYSTEM_PROMPT, user_prompt)
    if not answer_text or NOT_FOUND_RESPONSE in answer_text and len(answer_text) <= len(NOT_FOUND_RESPONSE) + 10:
        if answer_text.strip(" .'\"") == NOT_FOUND_RESPONSE.strip("."):
            answer_text = NOT_FOUND_RESPONSE

    return {
        "question": question,
        "answer": answer_text,
        "not_found": answer_text == NOT_FOUND_RESPONSE,
        "chunks": labeled_chunks,
        "retrieval_mode": chosen_mode,
        "rerank_enabled": use_rerank,
        "rerank_model": cfg.rerank_model if use_rerank else None,
        "filter_document_enabled": use_filter,
        "rewrite_query_enabled": use_rewrite,
        "parent_child_enabled": use_parent_child,
        "detected_document": detected_doc,
        "detected_version": detected_ver,
        "rewritten_query": rewritten_query,
        "was_rewritten": was_rewritten,
        "chat_model": f"{cfg.chat_provider}/{cfg.chat_model}",
        "embedding_model": f"{cfg.embedding_provider}/{cfg.embedding_model}",
    }


def answer(
    question: str,
    top_k: int = 5,
    config: RagConfig | None = None,
    mode: str | None = None,
    rerank: bool | None = None,
    rerank_candidates: int | None = None,
    filter_document: bool | None = None,
    rewrite_query: bool | None = None,
    parent_child: bool | None = None,
    guardrails: bool | None = None,
    allowed_documents: list[str] | set[str] | None = None,
) -> dict[str, Any]:
    """Execute RAG answer with optional NVIDIA NeMo Guardrails protection and role-based document access."""
    cfg = config or load_rag_config()
    use_guardrails = cfg.guardrails_enabled if guardrails is None else bool(guardrails)

    if use_guardrails:
        from guardrails_pipeline import guarded_answer
        return guarded_answer(
            question,
            top_k=top_k,
            config=cfg,
            mode=mode,
            rerank=rerank,
            rerank_candidates=rerank_candidates,
            filter_document=filter_document,
            rewrite_query=rewrite_query,
            parent_child=parent_child,
            enable_guardrails=True,
            allowed_documents=allowed_documents,
        )

    return core_answer(
        question,
        top_k=top_k,
        config=cfg,
        mode=mode,
        rerank=rerank,
        rerank_candidates=rerank_candidates,
        filter_document=filter_document,
        rewrite_query=rewrite_query,
        parent_child=parent_child,
        allowed_documents=allowed_documents,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question using the RAG answer pipeline.")
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="Question to answer (omit to run built-in answerable + unanswerable verification).",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve (default: 5).")
    parser.add_argument(
        "--candidates",
        type=int,
        default=30,
        help="Number of initial candidates for re-ranking (default: 30).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["vector", "hybrid"],
        default="hybrid",
        help="Retrieval mode: vector or hybrid (default: hybrid).",
    )
    parser.add_argument(
        "--rerank",
        dest="rerank",
        action="store_true",
        default=None,
        help="Enable cross-encoder re-ranking stage.",
    )
    parser.add_argument(
        "--no-rerank",
        dest="rerank",
        action="store_false",
        help="Disable cross-encoder re-ranking stage.",
    )
    parser.add_argument(
        "--filter-doc",
        dest="filter_doc",
        action="store_true",
        default=None,
        help="Enable document and version metadata filtering.",
    )
    parser.add_argument(
        "--no-filter-doc",
        dest="filter_doc",
        action="store_false",
        help="Disable document and version metadata filtering.",
    )
    parser.add_argument(
        "--rewrite-query",
        dest="rewrite_query",
        action="store_true",
        default=None,
        help="Enable rewriting short/vague queries before search.",
    )
    parser.add_argument(
        "--no-rewrite-query",
        dest="rewrite_query",
        action="store_false",
        help="Disable query rewriting.",
    )
    parser.add_argument(
        "--parent-child",
        dest="parent_child",
        action="store_true",
        default=None,
        help="Enable Parent-Child retrieval (expand chunks to parent statutory sections).",
    )
    parser.add_argument(
        "--no-parent-child",
        dest="parent_child",
        action="store_false",
        help="Disable Parent-Child retrieval.",
    )
    parser.add_argument(
        "--guardrails",
        dest="guardrails",
        action="store_true",
        default=None,
        help="Enable NVIDIA NeMo Guardrails (input and output rails).",
    )
    parser.add_argument(
        "--no-guardrails",
        dest="guardrails",
        action="store_false",
        help="Disable NVIDIA NeMo Guardrails.",
    )
    args = parser.parse_args()

    questions = (
        [args.question]
        if args.question
        else [
            "Explain the restrictions imposed under Section 185 on loans to directors and interested entities.",
            "What is the GST rate and SAC code for legal consultancy services under the CGST Act, 2017?",
        ]
    )

    for q in questions:
        res = answer(
            q,
            top_k=args.top_k,
            mode=args.mode,
            rerank=args.rerank,
            rerank_candidates=args.candidates,
            filter_document=args.filter_doc,
            rewrite_query=args.rewrite_query,
            parent_child=args.parent_child,
            guardrails=args.guardrails,
        )
        print("=" * 72)
        print(f"Question : {res['question']}")
        if res.get("was_rewritten"):
            print(f"Rewritten: {res['rewritten_query']}")
        if res.get("detected_document"):
            print(f"Filter   : {res['detected_document']} (ver: {res.get('detected_version')})")
        guard_txt = f" | guardrails={res.get('guardrails_provider')}:{res.get('guardrails_model')}" if res.get("guardrails_enabled") else ""
        if res.get("is_blocked"):
            guard_txt += f" [BLOCKED by {res.get('block_rail')} rail: {res.get('block_action')}]"
        rerank_txt = f" | rerank={res['rerank_model']}" if res.get("rerank_enabled") else ""
        print(f"Mode     : {res['retrieval_mode'].upper()}{rerank_txt}{guard_txt} | chat={res['chat_model']} | embed={res['embedding_model']}")
        print(f"Answer   :\n{res['answer']}\n")
        if res.get("chunks"):
            print("Sources used:")
            for ch in res["chunks"]:
                rerank_val = f", rerank_score={ch['rerank_score']:.4f} (init #{ch.get('initial_rank')})" if "rerank_score" in ch else ""
                print(f"  #{ch['rank']} {ch['citation']} (sim={ch['similarity']:.4f}{rerank_val}, id={ch['chunk_id']})")


if __name__ == "__main__":
    main()
