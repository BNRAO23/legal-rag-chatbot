"""Unified End-to-End Document Updater for Legal RAG Chatbot.

Executes the full pipeline in one command:
1. Parse: Converts PDF source documents to clean Markdown with page annotations.
2. Chunk: Splits Markdown along natural statutory units with full metadata.
3. Supersede: If the document is a new version of an already-indexed document,
   marks older chunks as superseded in both chunks.jsonl and ChromaDB so only the
   current version is searched.
4. Embed & Index: Embeds new chunks and updates ChromaDB vector index.
5. Invalidate: Clears in-memory BM25 and vector search caches.
6. Evaluate: Automatically runs the evaluation script and prints the scorecard.

Usage:
    python update_document.py <path_to_pdf_or_md> [--version <ver>] [--doc-name <name>] [--eval-limit <N>]
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from build_vector_db import sanitize_metadata
from chunk_documents import process_markdown_file
from parse_documents import process_pdf
from rag_config import Embedder, RagConfig, load_rag_config
from retrieve import DOCUMENT_REGISTRY, clear_caches


def resolve_doc_name_and_version(
    file_path: Path,
    doc_name_override: str | None = None,
    version_override: str | None = None,
) -> tuple[str, str]:
    """Resolve standard document name and version string from filename or overrides."""
    if doc_name_override:
        doc_name = doc_name_override
    else:
        # Match against known registry
        fn_lower = file_path.name.lower()
        matched = False
        for reg_name, info in DOCUMENT_REGISTRY.items():
            if any(alias in fn_lower for alias in info["aliases"]):
                doc_name = reg_name
                matched = True
                break
        if not matched:
            doc_name = file_path.stem.replace("_", " ").title()

    if version_override:
        version_str = version_override
    else:
        # Default timestamped version string
        now = datetime.datetime.now().strftime("%Y-%m-%d")
        version_str = f"Updated as on {now}"

    return doc_name, version_str


def parse_file(file_path: Path, parsed_dir: Path) -> Path:
    """Parse PDF or read existing Markdown into parsed_docs/."""
    parsed_dir.mkdir(parents=True, exist_ok=True)
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        print(f"[Stage 1/5] Parsing PDF '{file_path.name}' with PyMuPDF...")
        summary = process_pdf(file_path, parsed_dir)
        md_path = Path(summary.output_file)
        print(f"  Extracted {summary.total_words:,} words across {summary.total_pages} pages ({summary.elapsed_seconds}s)")
        return md_path
    elif suffix in (".md", ".markdown"):
        print(f"[Stage 1/5] Reusing pre-parsed Markdown '{file_path.name}'...")
        dest = parsed_dir / file_path.name
        if file_path.resolve() != dest.resolve():
            dest.write_text(file_path.read_text(encoding="utf-8"), encoding="utf-8")
        return dest
    else:
        raise ValueError(f"Unsupported file format '{suffix}'. Expected .pdf or .md")


def chunk_document(
    md_path: Path,
    doc_name: str,
    version_str: str,
    max_tokens: int = 450,
) -> list[dict[str, Any]]:
    """Chunk Markdown along natural statutory units, stamping version metadata."""
    print(f"[Stage 2/5] Chunking '{md_path.name}' along natural statutory units (max_tokens={max_tokens})...")
    doc_meta, units, raw_chunks = process_markdown_file(md_path, max_tokens=max_tokens)

    # Stamp document_name, version, and active status
    stamped_chunks: list[dict[str, Any]] = []
    ver_clean = "".join(c for c in version_str if c.isalnum() or c in ("-", "_")).lower()

    for idx, ch in enumerate(raw_chunks, start=1):
        item = dict(ch)
        meta = dict(item["metadata"])
        meta["document_name"] = doc_name
        meta["version_or_year"] = version_str
        meta["is_superseded"] = False
        meta["is_current"] = True
        meta["status"] = "active"
        meta["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Ensure unique ID with version marker
        base_id = item["chunk_id"]
        if ver_clean and ver_clean not in base_id:
            item["chunk_id"] = f"{base_id}__v{ver_clean}"

        item["metadata"] = meta
        stamped_chunks.append(item)

    sec_count = sum(1 for u in units if u.unit_type == "section")
    sched_count = sum(1 for u in units if u.unit_type == "schedule")
    print(f"  Generated {len(stamped_chunks)} chunks from {len(units)} units ({sec_count} sections, {sched_count} schedules)")
    return stamped_chunks


def update_chunks_store(
    chunks_path: Path,
    new_chunks: list[dict[str, Any]],
    doc_name: str,
    new_version: str,
) -> tuple[int, int]:
    """Mark old chunks of doc_name as superseded and append new chunks to chunks.jsonl."""
    print(f"[Stage 3/5] Updating chunk database ({chunks_path})...")
    existing_chunks: list[dict[str, Any]] = []
    if chunks_path.exists():
        with chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    existing_chunks.append(json.loads(s))

    superseded_count = 0
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Mark existing chunks of doc_name as superseded
    updated_existing: list[dict[str, Any]] = []
    for c in existing_chunks:
        c_copy = dict(c)
        meta = dict(c_copy["metadata"])
        if meta.get("document_name") == doc_name and meta.get("status") != "superseded":
            meta["is_superseded"] = True
            meta["is_current"] = False
            meta["status"] = "superseded"
            meta["superseded_by_version"] = new_version
            meta["superseded_at"] = now_iso
            c_copy["metadata"] = meta
            superseded_count += 1
        updated_existing.append(c_copy)

    # Append new active chunks
    final_chunks = updated_existing + new_chunks

    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = chunks_path.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        for ch in final_chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")
    temp_path.replace(chunks_path)

    if superseded_count > 0:
        print(f"  ⚠️  Detected previous version: marked {superseded_count} older chunks as SUPERSEDED.")
    else:
        print(f"  Added as a new document (0 older chunks superseded).")
    print(f"  Appended {len(new_chunks)} active chunks (Total in file: {len(final_chunks)}).")

    return superseded_count, len(new_chunks)


def index_into_vector_db(
    config: RagConfig,
    new_chunks: list[dict[str, Any]],
    doc_name: str,
    new_version: str,
    superseded_count: int,
) -> int:
    """Update ChromaDB: mark old vectors as superseded and upsert new embeddings."""
    print(f"[Stage 4/5] Indexing vectors into ChromaDB (collection='{config.collection_name}')...")
    import chromadb

    client = chromadb.PersistentClient(path=str(config.chroma_dir))
    collection = client.get_or_create_collection(
        name=config.collection_name,
        metadata={
            "hnsw:space": "cosine",
            "embedding_provider": config.embedding_provider,
            "embedding_model": config.embedding_model,
        },
    )

    # 1. Update metadata of superseded vectors in ChromaDB
    if superseded_count > 0:
        old_data = collection.get(
            where={"document_name": doc_name},
            include=["metadatas"],
        )
        old_ids = old_data.get("ids", [])
        old_metas = old_data.get("metadatas", [])
        if old_ids:
            updated_metas: list[dict[str, Any]] = []
            now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
            for m in old_metas:
                m_copy = dict(m)
                m_copy["is_superseded"] = True
                m_copy["is_current"] = False
                m_copy["status"] = "superseded"
                m_copy["superseded_by_version"] = new_version
                m_copy["superseded_at"] = now_iso
                updated_metas.append(sanitize_metadata(m_copy))

            collection.update(ids=old_ids, metadatas=updated_metas)
            print(f"  Marked {len(old_ids)} existing ChromaDB vectors as SUPERSEDED.")

    # 2. Embed and upsert new chunks in batches
    embedder = Embedder(config)
    batch_size = config.batch_size or 64

    for start in range(0, len(new_chunks), batch_size):
        batch = new_chunks[start : start + batch_size]
        ids = [c["chunk_id"] for c in batch]
        texts = [c["text"] for c in batch]
        metadatas = [sanitize_metadata(c["metadata"]) for c in batch]

        embeddings = embedder.embed_documents(texts)
        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    # Invalidate in-memory caches
    clear_caches()

    stored_count = collection.count()
    print(f"  Upserted {len(new_chunks)} new vectors (Total in ChromaDB: {stored_count}).")
    return stored_count


def run_evaluation(limit: int = 5, skip_metrics: bool = True) -> None:
    """Run evaluate_rag.py and stream the evaluation scorecard."""
    print(f"[Stage 5/5] Automatically running evaluation script (cases limit={limit})...\n")

    cmd = [
        sys.executable,
        "evaluate_rag.py",
        "--compare",
        "none",
        "--mode",
        "hybrid",
        "--rerank",
        "true",
        "--filter-doc",
        "true",
        "--rewrite-query",
        "true",
        "--parent-child",
        "true",
        "--limit",
        str(limit),
    ]
    if skip_metrics:
        cmd.append("--skip-metrics")

    subprocess.run(cmd, check=True)


def update_document(
    file_path: str | Path,
    doc_name: str | None = None,
    version: str | None = None,
    eval_limit: int = 5,
    skip_eval: bool = False,
    skip_judge_metrics: bool = True,
) -> None:
    """End-to-end command to add or replace a document in the RAG chatbot."""
    t0 = time.perf_counter()
    p = Path(file_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Document file not found: {p}")

    resolved_name, resolved_version = resolve_doc_name_and_version(p, doc_name, version)
    cfg = load_rag_config()

    print("=" * 80)
    print("LEGAL RAG DOCUMENT UPDATER (END-TO-END PIPELINE)")
    print("=" * 80)
    print(f"Source Document : {p.name}")
    print(f"Target Document : {resolved_name}")
    print(f"Target Version  : {resolved_version}")
    print(f"Embedding Model : {cfg.embedding_provider}/{cfg.embedding_model}")
    print(f"Vector Database : {cfg.chroma_dir} (collection='{cfg.collection_name}')")
    print("=" * 80)
    print()

    # Step 1: Parse
    md_path = parse_file(p, Path("parsed_docs"))

    # Step 2: Chunk
    new_chunks = chunk_document(md_path, resolved_name, resolved_version)

    # Step 3: Supersede old chunks in chunks.jsonl
    superseded, num_new = update_chunks_store(cfg.chunks_path, new_chunks, resolved_name, resolved_version)

    # Step 4: Index into ChromaDB
    total_vectors = index_into_vector_db(cfg, new_chunks, resolved_name, resolved_version, superseded)

    total_time = round(time.perf_counter() - t0, 2)
    print()
    print("=" * 80)
    print(f"UPDATE COMPLETED SUCCESSFULLY in {total_time}s")
    print(f"- Document   : {resolved_name} [{resolved_version}]")
    print(f"- Status     : {'Replaced (superseded previous version)' if superseded > 0 else 'Added as new'}")
    print(f"- Vectors    : {num_new} indexed, {superseded} superseded (Collection total: {total_vectors})")
    print("=" * 80)
    print()

    # Step 5: Run Evaluation Script
    if not skip_eval:
        run_evaluation(limit=eval_limit, skip_metrics=skip_judge_metrics)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add or replace a document in the legal RAG chatbot end to end, superseding old chunks and re-evaluating."
    )
    parser.add_argument("document", help="Path to document file (.pdf or .md) to add or replace.")
    parser.add_argument("--doc-name", default=None, help="Document name in registry (default: auto-detected).")
    parser.add_argument("--version", default=None, help="Version string for the document (default: auto/timestamp).")
    parser.add_argument("--eval-limit", type=int, default=5, help="Number of eval cases to run after update (default: 5).")
    parser.add_argument("--skip-eval", action="store_true", help="Skip running the evaluation script after update.")
    parser.add_argument(
        "--full-metrics",
        action="store_true",
        help="Run full DeepEval LLM judge metrics during post-update evaluation (takes longer).",
    )
    args = parser.parse_args()

    update_document(
        file_path=args.document,
        doc_name=args.doc_name,
        version=args.version,
        eval_limit=args.eval_limit,
        skip_eval=args.skip_eval,
        skip_judge_metrics=not args.full_metrics,
    )


if __name__ == "__main__":
    main()

