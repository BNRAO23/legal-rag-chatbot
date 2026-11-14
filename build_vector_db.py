"""Embed every chunk from parsed_docs/chunks.jsonl and store in local ChromaDB.

Uses `rag_config.py` so the embedding provider/model name stays centralized and
queries reuse the exact same model later. Processes chunks in batches, displays
progress, stores full citation metadata, and prints the final DB count vs chunk count.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import chromadb
from tqdm import tqdm

from rag_config import Embedder, RagConfig, load_rag_config


def sanitize_metadata(raw_meta: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Convert chunk metadata into ChromaDB-safe scalar values without dropping fields."""
    clean: dict[str, str | int | float | bool] = {}
    for key, val in raw_meta.items():
        if val is None:
            clean[key] = ""
        elif isinstance(val, list):
            clean[key] = ",".join(str(x) for x in val)
        elif isinstance(val, (str, int, float, bool)):
            clean[key] = val
        else:
            clean[key] = json.dumps(val, ensure_ascii=False)
    return clean


def load_chunks(chunks_path: Path) -> list[dict[str, Any]]:
    """Load all chunk records from chunks.jsonl."""
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
    records: list[dict[str, Any]] = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def index_chunks(config: RagConfig, *, reset: bool = False) -> tuple[int, int]:
    """Embed chunks in batches and upsert text, vectors, and metadata into ChromaDB."""
    chunks = load_chunks(config.chunks_path)
    total_chunks = len(chunks)
    if total_chunks == 0:
        raise ValueError(f"No chunks found in {config.chunks_path}")

    print(f"Loaded {total_chunks} chunks from {config.chunks_path}")
    print(f"Embedding provider : {config.embedding_provider}")
    print(f"Embedding model    : {config.embedding_model}")
    print(f"Vector DB (local)  : {config.chroma_dir.resolve()} (collection='{config.collection_name}')")
    print(f"Batch size         : {config.batch_size}\n")

    embedder = Embedder(config)
    client = chromadb.PersistentClient(path=str(config.chroma_dir))

    if reset:
        try:
            client.delete_collection(config.collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=config.collection_name,
        metadata={
            "hnsw:space": "cosine",
            "embedding_provider": config.embedding_provider,
            "embedding_model": config.embedding_model,
        },
    )

    existing_model = (collection.metadata or {}).get("embedding_model")
    if existing_model and existing_model != config.embedding_model and collection.count() > 0:
        raise RuntimeError(
            f"Collection '{config.collection_name}' was indexed with '{existing_model}', "
            f"but current config specifies '{config.embedding_model}'. "
            "Pass --reset to rebuild with the new model."
        )

    existing_ids: set[str] = set()
    if not reset and collection.count() > 0:
        existing_ids = set(collection.get(include=[])["ids"])
        if existing_ids:
            print(f"Resuming: {len(existing_ids)} chunks already in collection; embedding remaining {total_chunks - len(existing_ids)}.")

    pending = [c for c in chunks if c["chunk_id"] not in existing_ids]
    with tqdm(total=total_chunks, initial=len(existing_ids), desc="Embedding & upserting", unit="chunk") as pbar:
        for start in range(0, len(pending), config.batch_size):
            batch = pending[start : start + config.batch_size]
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
            pbar.update(len(batch))

    stored_count = collection.count()
    print("\n=== VECTOR DB INDEXING SUMMARY ===")
    print(f"Source chunks count : {total_chunks}")
    print(f"Vectors in ChromaDB : {stored_count}")
    if stored_count == total_chunks:
        print("Status              : MATCH (all chunks indexed)")
    else:
        print(f"Status              : MISMATCH (diff = {stored_count - total_chunks})")

    return total_chunks, stored_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed all chunks from chunks.jsonl and store them in local ChromaDB."
    )
    parser.add_argument("--chunks", type=Path, default=None, help="Path to chunks.jsonl.")
    parser.add_argument("--chroma-dir", type=Path, default=None, help="Local ChromaDB directory.")
    parser.add_argument("--collection", type=str, default=None, help="ChromaDB collection name.")
    parser.add_argument("--provider", type=str, default=None, help="Override embedding provider.")
    parser.add_argument("--model", type=str, default=None, help="Override embedding model name.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default: 64).")
    parser.add_argument("--reset", action="store_true", help="Delete and rebuild the collection.")
    args = parser.parse_args()

    try:
        config = load_rag_config(
            provider=args.provider,
            model=args.model,
            chunks_path=args.chunks,
            chroma_dir=args.chroma_dir,
            collection_name=args.collection,
            batch_size=args.batch_size,
        )
    except RuntimeError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(2)

    index_chunks(config, reset=args.reset)


if __name__ == "__main__":
    main()

