"""Retrieve top-k chunks from ChromaDB using Vector search or Hybrid (Vector + BM25) search.

Supports:
1. Vector-only retrieval using the exact embedding model stored in ChromaDB metadata.
2. Keyword search (BM25) over the exact same chunks.
3. Hybrid search combining both ranked lists via Reciprocal Rank Fusion (RRF).
4. Configurable via `config.retrieval_mode` or the `mode` parameter ('vector' or 'hybrid').
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

from rag_config import Embedder, RagConfig, get_secret, load_rag_config


_CACHED_CLIENT: dict[tuple[str, str], tuple[chromadb.Collection, Embedder]] = {}
_CACHED_BM25: dict[tuple[str, str], "BM25Index"] = {}
_CACHED_RERANKER: dict[str, Any] = {}


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric words for BM25 matching."""
    return re.findall(r"\b[a-z0-9_]+\b", text.lower())


def clear_caches() -> None:
    """Clear in-memory caches for Chroma collections and BM25 indexes."""
    _CACHED_CLIENT.clear()
    _CACHED_BM25.clear()


def _match_metadata_filter(meta: dict[str, Any], where: dict[str, Any]) -> bool:
    """Evaluate complex ChromaDB-style where filters ($and, $or, $in, $ne, $eq) against metadata."""
    if "$and" in where:
        return all(_match_metadata_filter(meta, sub) for sub in where["$and"])
    if "$or" in where:
        return any(_match_metadata_filter(meta, sub) for sub in where["$or"])

    for wk, wv in where.items():
        if wk in ("$and", "$or"):
            continue
        val = meta.get(wk)
        if isinstance(wv, dict):
            if "$in" in wv and val not in wv["$in"]:
                return False
            if "$nin" in wv and val in wv["$nin"]:
                return False
            if "$ne" in wv and val == wv["$ne"]:
                return False
            if "$eq" in wv and val != wv["$eq"]:
                return False
        elif isinstance(wv, (list, tuple, set)):
            if val not in wv:
                return False
        elif val != wv:
            return False
    return True


@dataclass
class BM25Index:
    ids: list[str]
    documents: list[str]
    metadatas: list[dict[str, Any]]
    bm25: Any

    def query(
        self,
        query_text: str,
        top_n: int = 20,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        tokens = _tokenize(query_text)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results: list[dict[str, Any]] = []
        rank = 1
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0.0:
                continue
            meta = self.metadatas[idx]
            if where and not _match_metadata_filter(meta, where):
                continue
            results.append(
                {
                    "bm25_rank": rank,
                    "chunk_id": self.ids[idx],
                    "text": self.documents[idx],
                    "metadata": meta,
                    "bm25_score": round(score, 4),
                }
            )
            rank += 1
            if len(results) >= top_n:
                break
        return results


class _PurePythonBM25Okapi:
    """Pure-Python fallback implementation of BM25Okapi if rank-bm25 is missing."""

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus_tokens)
        self.doc_lens = [len(doc) for doc in corpus_tokens]
        self.avgdl = sum(self.doc_lens) / (self.corpus_size or 1)
        self.doc_freqs: list[Counter[str]] = []
        self.nd: Counter[str] = Counter()
        for doc in corpus_tokens:
            counts = Counter(doc)
            self.doc_freqs.append(counts)
            for word in counts.keys():
                self.nd[word] += 1
        self.idf = {
            word: math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)
            for word, freq in self.nd.items()
        }

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * self.corpus_size
        for q in query_tokens:
            if q not in self.idf:
                continue
            idf_q = self.idf[q]
            for i, counts in enumerate(self.doc_freqs):
                freq = counts.get(q, 0)
                if freq == 0:
                    continue
                num = freq * (self.k1 + 1.0)
                denom = freq + self.k1 * (1.0 - self.b + self.b * (self.doc_lens[i] / self.avgdl))
                scores[i] += idf_q * (num / denom)
        return scores


def _get_collection_and_embedder(config: RagConfig) -> tuple[chromadb.Collection, Embedder]:
    cache_key = (str(config.chroma_dir.resolve()), config.collection_name)
    if cache_key in _CACHED_CLIENT:
        return _CACHED_CLIENT[cache_key]

    client = chromadb.PersistentClient(path=str(config.chroma_dir))
    collection = client.get_collection(name=config.collection_name)
    col_meta = collection.metadata or {}
    indexed_model = col_meta.get("embedding_model")
    indexed_provider = col_meta.get("embedding_provider")

    if indexed_model and indexed_model != config.embedding_model:
        raise RuntimeError(
            f"Model mismatch: collection '{config.collection_name}' was indexed with "
            f"'{indexed_provider}/{indexed_model}', but current config uses "
            f"'{config.embedding_provider}/{config.embedding_model}'."
        )

    embedder = Embedder(config)
    _CACHED_CLIENT[cache_key] = (collection, embedder)
    return collection, embedder


def _get_bm25_index(config: RagConfig) -> BM25Index:
    """Load all chunks from the ChromaDB collection and construct/cache a BM25 index."""
    cache_key = (str(config.chroma_dir.resolve()), config.collection_name)
    if cache_key in _CACHED_BM25:
        return _CACHED_BM25[cache_key]

    collection, _ = _get_collection_and_embedder(config)
    data = collection.get(include=["documents", "metadatas"])
    ids: list[str] = data.get("ids", [])
    documents: list[str] = data.get("documents", [])
    metadatas: list[dict[str, Any]] = data.get("metadatas", [])

    tokenized_corpus = [_tokenize(doc) for doc in documents]

    try:
        from rank_bm25 import BM25Okapi

        bm25_model = BM25Okapi(tokenized_corpus)
    except Exception:
        bm25_model = _PurePythonBM25Okapi(tokenized_corpus)

    index = BM25Index(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        bm25=bm25_model,
    )
    _CACHED_BM25[cache_key] = index
    return index


def _vector_search(
    question: str,
    top_n: int,
    config: RagConfig,
    where: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Retrieve top-n candidates using vector embedding cosine similarity."""
    collection, embedder = _get_collection_and_embedder(config)
    query_vec = embedder.embed_query(question)
    query_kwargs: dict[str, Any] = {
        "query_embeddings": [query_vec],
        "n_results": top_n,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        query_kwargs["where"] = where

    res = collection.query(**query_kwargs)

    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    results: list[dict[str, Any]] = []
    for rank, (cid, doc_text, meta, dist) in enumerate(zip(ids, docs, metas, dists), start=1):
        # hnsw:space="cosine", so similarity = 1 - cosine_distance.
        similarity = round(1.0 - float(dist), 4)
        results.append(
            {
                "vector_rank": rank,
                "chunk_id": cid,
                "text": doc_text,
                "metadata": meta,
                "similarity": similarity,
                "distance": round(float(dist), 4),
            }
        )
    return results


def _reciprocal_rank_fusion(
    vector_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    k: int = 60,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Merge ranked vector results and ranked BM25 results using Reciprocal Rank Fusion."""
    rrf_scores: dict[str, float] = {}
    chunk_store: dict[str, dict[str, Any]] = {}
    vector_ranks: dict[str, int] = {}
    bm25_ranks: dict[str, int] = {}
    similarities: dict[str, float] = {}
    bm25_scores: dict[str, float] = {}

    for item in vector_results:
        cid = item["chunk_id"]
        v_rank = item["vector_rank"]
        vector_ranks[cid] = v_rank
        similarities[cid] = item.get("similarity", 0.0)
        chunk_store[cid] = item
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + v_rank))

    for item in bm25_results:
        cid = item["chunk_id"]
        b_rank = item["bm25_rank"]
        bm25_ranks[cid] = b_rank
        bm25_scores[cid] = item.get("bm25_score", 0.0)
        if cid not in chunk_store:
            chunk_store[cid] = item
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + b_rank))

    # Sort candidates by combined RRF score descending
    sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

    merged_results: list[dict[str, Any]] = []
    for rank, cid in enumerate(sorted_cids, start=1):
        item = chunk_store[cid]
        sim = similarities.get(cid)
        if sim is None:
            # If item was retrieved only by BM25, approximate similarity from top rank
            sim = round(0.5 + min(0.4, bm25_scores.get(cid, 0.0) / 50.0), 4)

        merged_results.append(
            {
                "rank": rank,
                "chunk_id": cid,
                "text": item["text"],
                "metadata": item["metadata"],
                "similarity": sim,
                "rrf_score": round(rrf_scores[cid], 6),
                "vector_rank": vector_ranks.get(cid),
                "bm25_rank": bm25_ranks.get(cid),
                "bm25_score": bm25_scores.get(cid),
                "retrieval_mode": "hybrid",
            }
        )

    return merged_results


def _get_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> Any:
    """Load and cache the CrossEncoder model in memory."""
    if model_name in _CACHED_RERANKER:
        return _CACHED_RERANKER[model_name]
    try:
        from sentence_transformers import CrossEncoder

        try:
            model = CrossEncoder(model_name, local_files_only=True)
        except Exception:
            model = CrossEncoder(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Could not initialize CrossEncoder model '{model_name}'. "
            f"Ensure sentence-transformers and torch are installed: {exc}"
        )
    _CACHED_RERANKER[model_name] = model
    return model


def re_rank_candidates(
    question: str,
    candidates: list[dict[str, Any]],
    top_k: int = 5,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> list[dict[str, Any]]:
    """Re-rank candidate chunks using a cross-encoder model and return top-k."""
    if not candidates:
        return []
    reranker = _get_cross_encoder(model_name)
    pairs = [(question, c.get("text", "")) for c in candidates]
    scores = reranker.predict(pairs)

    enriched: list[dict[str, Any]] = []
    for i, c in enumerate(candidates):
        item = dict(c)
        item["initial_rank"] = item.get("rank") or (i + 1)
        item["rerank_score"] = round(float(scores[i]), 4)
        enriched.append(item)

    # Sort descending by cross-encoder score
    enriched.sort(key=lambda x: x["rerank_score"], reverse=True)
    top_reranked = enriched[:top_k]

    for new_rank, item in enumerate(top_reranked, start=1):
        item["rank"] = new_rank
        item["rerank_enabled"] = True
        item["rerank_model"] = model_name

    return top_reranked


# Backward-compatibility alias
rerank_chunks = re_rank_candidates


DOCUMENT_REGISTRY: dict[str, dict[str, Any]] = {
    "Companies Act, 2013": {
        "aliases": [
            "companies act",
            "companies act, 2013",
            "companies act 2013",
            "ca 2013",
            "company law",
            "private limited company",
            "public limited company",
            "memorandum of association",
            "articles of association",
            "director",
            "directors",
            "board meeting",
            "statutory auditor",
            "corporate social responsibility",
            "csr",
            "section 7",
            "section 185",
            "section 135",
            "section 188",
        ],
        "current_version": "2013 (as amended up to 1-4-2021)",
        "current_year": "2013",
        "older_keywords": ["1956", "companies act, 1956", "companies act 1956", "older act", "previous act", "prior act", "old companies act", "repealed act"],
    },
    "Foreign Exchange Management Act, 1999": {
        "aliases": [
            "fema",
            "foreign exchange management act",
            "foreign exchange management act, 1999",
            "fema 1999",
            "fema, 1999",
            "ecb",
            "external commercial borrowing",
            "fdi",
            "foreign direct investment",
            "lrs",
            "liberalised remittance",
            "fc-gpr",
            "fc-trs",
            "person resident in india",
            "pri",
            "proi",
        ],
        "current_version": "1999 (Act 42 of 1999; As on 1st June, 2026)",
        "current_year": "1999",
        "older_keywords": ["fera", "1973", "older act", "previous act", "prior act", "foreign exchange regulation"],
    },
    "Income-tax Act, 2025": {
        "aliases": [
            "income-tax",
            "income tax",
            "income-tax act",
            "income tax act",
            "it act",
            "advance tax",
            "tds",
            "tcs",
            "tax deduction at source",
            "capital gains tax",
        ],
        "current_version": "2025 (Act 30 of 2025; As amended by Finance Act, 2026)",
        "current_year": "2025",
        "older_keywords": ["1961", "income-tax act, 1961", "income tax act 1961", "older act", "previous act", "prior act", "old act", "1961 act"],
    },
}


def detect_document_and_version(
    question: str,
    available_documents: set[str] | None = None,
) -> tuple[str | None, str | None, bool]:
    """Detect if question specifies a document and whether an older or current version is requested.

    Returns:
        (target_document_name, target_version, is_older_requested)
        When two versions exist, defaults to current version unless the user asks for an older one.
    """
    q_lower = question.lower()

    # 1. Match known legal registry
    for doc_name, info in DOCUMENT_REGISTRY.items():
        if any(alias in q_lower for alias in info["aliases"]):
            is_older = any(k in q_lower for k in info["older_keywords"])
            target_version = "older" if is_older else info["current_version"]
            return doc_name, target_version, is_older

    # 2. Check for explicit out-of-corpus Acts (e.g. CGST Act, 2017)
    if "cgst" in q_lower or "goods and services tax" in q_lower:
        return "Central Goods and Services Tax (CGST) Act, 2017", "2017", False
    if "customs act" in q_lower:
        return "Customs Act, 1962", "1962", False
    if "insolvency and bankruptcy" in q_lower or "ibc" in q_lower:
        return "Insolvency and Bankruptcy Code, 2016", "2016", False

    # 3. Dynamic match against available collection documents
    if available_documents:
        for doc in available_documents:
            if doc.lower() in q_lower:
                return doc, None, False

    return None, None, False


def rewrite_search_query(question: str, config: RagConfig) -> tuple[str, bool]:
    """Rewrite short or vague questions into a full, keyword-rich search query.

    If query is already complete and detailed (8+ words with context), leaves it unchanged.
    """
    words = re.findall(r"\b\w+\b", question)
    is_short = len(words) < 8
    is_vague = not any(doc in question.lower() for doc in ("companies act", "income-tax", "income tax", "fema", "act"))

    if not (is_short or is_vague):
        return question, False

    # Fast offline rule-based dictionary for standard legal queries
    q_clean = question.strip().lower()
    short_lookup: dict[str, str] = {
        "section 185": "Companies Act, 2013 Section 185 restrictions on loans to directors and entities in which directors are interested exceptions and penalties",
        "section 7": "Companies Act, 2013 Section 7 legal requirements and step-by-step procedure for incorporation of private limited company documents filed with registrar",
        "csr": "Companies Act, 2013 Section 135 Corporate Social Responsibility CSR applicability net worth turnover profit spending requirements",
        "section 135": "Companies Act, 2013 Section 135 Corporate Social Responsibility CSR applicability threshold and committee obligations",
        "auditor rotation": "Companies Act, 2013 Section 139 Section 140 appointment rotation and removal of statutory auditors",
        "board meeting quorum": "Companies Act, 2013 Section 173 Section 174 board meeting frequency notice and quorum requirements",
        "related party transaction": "Companies Act, 2013 Section 188 related party transactions approval criteria and thresholds",
    }
    for key, expanded in short_lookup.items():
        if key in q_clean and len(words) <= 6:
            return expanded, True

    # Call LLM rewriter if API key is present
    try:
        import os
        from google import genai
        from google.genai import types

        api_key = get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY")
        if api_key:
            client = genai.Client(api_key=api_key)
            prompt = (
                "You are a specialized query expander for Indian statutory law "
                "(Companies Act, 2013; Foreign Exchange Management Act, 1999; Income-tax Act, 2025).\n"
                "If the query is short (fewer than 8 words), incomplete, or vague, expand it into a complete, "
                "keyword-rich search query specifying the relevant Act, section number, and statutory subject.\n"
                "If already clear and specific, return it unchanged.\n"
                "Output ONLY the search query on a single line with no markdown or formatting.\n\n"
                f"Query: {question}\n"
                "Rewritten query:"
            )
            resp = client.models.generate_content(
                model=config.chat_model or "gemini-3.5-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0, top_p=1.0, seed=42),
            )
            expanded_text = (resp.text or "").strip()
            if expanded_text and len(expanded_text) > len(question):
                return expanded_text, True
    except Exception:
        pass

    return question, False


_PARENT_REGISTRY_CACHE: dict[Path, dict[tuple[str, str, str], list[dict[str, Any]]]] = {}


def get_parent_registry(chunks_path: Path) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    """Load and index all raw chunks by parent statutory unit (document_name, unit_type, unit_number)."""
    p = Path(chunks_path).resolve()
    if p in _PARENT_REGISTRY_CACHE:
        return _PARENT_REGISTRY_CACHE[p]

    registry: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    if p.exists():
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    c = json.loads(line)
                    meta = c.get("metadata", {})
                    doc_name = meta.get("document_name") or "Unknown"
                    unit_type = meta.get("unit_type") or "unit"
                    unit_num = str(meta.get("unit_number") or meta.get("unit_title") or "0")
                    key = (doc_name, unit_type, unit_num)
                    registry.setdefault(key, []).append(c)
                except Exception:
                    continue

    _PARENT_REGISTRY_CACHE[p] = registry
    return registry


def expand_to_parents(
    candidates: list[dict[str, Any]],
    chunks_path: Path,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Expand child candidate chunks to their parent statutory unit, merging sibling parts and deduplicating."""
    registry = get_parent_registry(chunks_path)
    parent_results: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for item in candidates:
        meta = item.get("metadata", {})
        doc_name = meta.get("document_name") or "Unknown"
        unit_type = meta.get("unit_type") or "unit"
        unit_num = str(meta.get("unit_number") or meta.get("unit_title") or "0")
        key = (doc_name, unit_type, unit_num)

        if key in seen:
            continue
        seen.add(key)

        siblings = registry.get(key, [item])
        # Sort siblings in natural order by part number
        siblings_sorted = sorted(siblings, key=lambda s: int(s.get("metadata", {}).get("part", 1) or 1))
        merged_text = "\n\n".join(s.get("text", "") for s in siblings_sorted)

        # Collect and union page numbers
        all_pages: list[int] = []
        for s in siblings_sorted:
            smeta = s.get("metadata", {})
            p_val = smeta.get("page")
            if p_val is not None:
                try:
                    all_pages.append(int(p_val))
                except (ValueError, TypeError):
                    pass
            pe_val = smeta.get("page_end")
            if pe_val is not None:
                try:
                    all_pages.append(int(pe_val))
                except (ValueError, TypeError):
                    pass
            if "pages" in smeta and isinstance(smeta["pages"], list):
                for pg in smeta["pages"]:
                    try:
                        all_pages.append(int(pg))
                    except (ValueError, TypeError):
                        pass

        all_pages_sorted = sorted(set(all_pages))
        min_p = all_pages_sorted[0] if all_pages_sorted else meta.get("page")
        max_p = all_pages_sorted[-1] if all_pages_sorted else meta.get("page_end", min_p)

        parent_meta = dict(meta)
        parent_meta["page"] = min_p
        parent_meta["page_end"] = max_p
        parent_meta["pages"] = all_pages_sorted
        parent_meta["total_parts"] = len(siblings_sorted)
        parent_meta["is_parent"] = True

        parent_item = dict(item)
        parent_item["text"] = merged_text
        parent_item["metadata"] = parent_meta
        parent_item["is_parent_expanded"] = True
        parent_item["parent_child_enabled"] = True
        parent_item["child_parts_count"] = len(siblings_sorted)
        parent_item["rank"] = len(parent_results) + 1
        parent_results.append(parent_item)

        if len(parent_results) >= top_k:
            break

    return parent_results


def retrieve(
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
) -> list[dict[str, Any]]:
    """Retrieve top-k chunks using vector or hybrid search, with optional re-ranking, doc filtering, rewriting, parent-child expansion, and role-based document access control."""
    if not question or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    cfg = config or load_rag_config()
    chosen_mode = (mode or cfg.retrieval_mode or "vector").strip().lower()
    use_rerank = cfg.rerank_enabled if rerank is None else bool(rerank)
    use_filter = cfg.filter_document if filter_document is None else bool(filter_document)
    use_rewrite = cfg.rewrite_query if rewrite_query is None else bool(rewrite_query)
    use_parent_child = cfg.parent_child if parent_child is None else bool(parent_child)
    candidate_pool_size = rerank_candidates or cfg.rerank_candidates or 30

    # 1. RBAC Check BEFORE Search
    allowed_list: list[str] | None = list(allowed_documents) if allowed_documents is not None else None
    if allowed_list is not None and len(allowed_list) == 0:
        # User role has 0 authorized documents
        return []

    search_query = question
    was_rewritten = False
    if use_rewrite:
        search_query, was_rewritten = rewrite_search_query(question, config=cfg)

    where_filter: dict[str, Any] | None = None
    target_doc: str | None = None
    target_ver: str | None = None
    is_older: bool = False
    if use_filter:
        target_doc, target_ver, is_older = detect_document_and_version(question)

    # 2. Check document permissions against role before querying
    if allowed_list is not None:
        if target_doc:
            if target_doc not in allowed_list:
                # Question targets a document that user's role is forbidden to see
                return []
            if is_older:
                where_filter = {"version_or_year": target_ver}
            else:
                where_filter = {"document_name": target_doc}
        else:
            if len(allowed_list) == 1:
                where_filter = {"document_name": allowed_list[0]}
            else:
                where_filter = {"document_name": {"$in": allowed_list}}
    else:
        if target_doc:
            if is_older:
                where_filter = {"version_or_year": target_ver}
            else:
                where_filter = {"document_name": target_doc}

    # 3. Ensure superseded chunks are excluded unless user explicitly asked for an older version
    if not is_older:
        active_clause = {"is_superseded": {"$ne": True}}
        if where_filter:
            where_filter = {"$and": [where_filter, active_clause]}
        else:
            where_filter = active_clause

    initial_top_n = max(top_k * 3 if use_parent_child else top_k, candidate_pool_size) if use_rerank else (top_k * 3 if use_parent_child else top_k)

    if chosen_mode == "hybrid":
        # Candidate pool size for fusion
        candidate_n = max(20, initial_top_n * 2)
        vector_candidates = _vector_search(search_query, top_n=candidate_n, config=cfg, where=where_filter)
        bm25_index = _get_bm25_index(cfg)
        bm25_candidates = bm25_index.query(search_query, top_n=candidate_n, where=where_filter)
        initial_results = _reciprocal_rank_fusion(
            vector_candidates,
            bm25_candidates,
            k=cfg.rrf_k,
            top_k=initial_top_n,
        )
    else:
        # Default: Vector-only search
        initial_results = _vector_search(search_query, top_n=initial_top_n, config=cfg, where=where_filter)
        for item in initial_results:
            item["rank"] = item.pop("vector_rank")
            item["retrieval_mode"] = "vector"

    # Defense-in-depth: enforce RBAC boundary on retrieved candidates
    if allowed_list is not None and initial_results:
        initial_results = [
            c for c in initial_results
            if str(c.get("metadata", {}).get("document_name")) in allowed_list
        ]

    # Defense-in-depth: exclude superseded chunks unless older requested
    if not is_older and initial_results:
        initial_results = [
            c for c in initial_results
            if c.get("metadata", {}).get("is_superseded") is not True
            and c.get("metadata", {}).get("status") != "superseded"
        ]

    if use_rerank and initial_results:
        rerank_keep = max(top_k * 3, 15) if use_parent_child else top_k
        final_candidates = re_rank_candidates(
            question=search_query,
            candidates=initial_results,
            top_k=rerank_keep,
            model_name=cfg.rerank_model,
        )
    else:
        final_candidates = initial_results

    if use_parent_child and final_candidates:
        final_results = expand_to_parents(
            candidates=final_candidates,
            chunks_path=cfg.chunks_path,
            top_k=top_k,
        )
    else:
        final_results = final_candidates[:top_k]

    for item in final_results:
        item["search_query"] = search_query
        item["was_rewritten"] = was_rewritten
        item["parent_child_enabled"] = use_parent_child
        if allowed_list is not None:
            item["allowed_documents"] = allowed_list
        if target_doc:
            item["target_document"] = target_doc
            item["target_version"] = target_ver
            item["is_older_requested"] = is_older

    return final_results


def load_golden_cases(path: Path) -> list[dict[str, Any]]:
    """Load JSONL test cases from CA_RAG_Golden_Eval_Dataset_Notepad.txt or JSON."""
    raw = path.read_text(encoding="utf-8")
    stripped = raw.strip()
    if stripped.startswith("["):
        return json.loads(stripped)

    cases: list[dict[str, Any]] = []
    for line in raw.splitlines():
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            cases.append(json.loads(s))
    return cases


def extract_expected_sections(case: dict[str, Any]) -> tuple[str, list[str]]:
    """Extract expected Act keyword and section numbers from `must_cite`."""
    subject = case.get("subject", "").lower()
    if "companies act" in subject:
        expected_doc = "Companies Act, 2013"
    elif "fema" in subject:
        expected_doc = "Foreign Exchange Management Act, 1999"
    elif "income-tax" in subject or "income tax" in subject:
        expected_doc = "Income-tax Act, 2025"
    else:
        expected_doc = ""

    sections: list[str] = []
    for cite in case.get("must_cite", []):
        if any(k in cite.lower() for k in ("companies act", "fema", "income-tax", "section")):
            for num in re.findall(r"\b(\d+[A-Z]*)\b", cite):
                if num not in {"2013", "1999", "2025", "2026", "2014", "2019", "2022", "2000"}:
                    sections.append(num)
    # Deduplicate preserving order
    deduped = list(dict.fromkeys(sections))
    return expected_doc, deduped


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve top chunks using Vector or Hybrid search, with optional Cross-Encoder re-ranking.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("CA_RAG_Golden_Eval_Dataset_Notepad.txt"),
        help="Path to golden dataset file.",
    )
    parser.add_argument(
        "--case-ids",
        nargs="+",
        default=["CA2013-001", "CA2013-004", "CA2013-005", "CA2013-006", "CA2013-008"],
        help="Case IDs to evaluate.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to return (default: 5).")
    parser.add_argument(
        "--candidates",
        type=int,
        default=30,
        help="Number of initial candidates to retrieve before re-ranking (default: 30).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["vector", "hybrid"],
        default="hybrid",
        help="Initial retrieval mode: vector or hybrid (default: hybrid).",
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
    parser.add_argument("--provider", type=str, default=None, help="Optional embedding provider override.")
    parser.add_argument("--model", type=str, default=None, help="Optional embedding model override.")
    args = parser.parse_args()

    cfg = load_rag_config(
        provider=args.provider,
        model=args.model,
        retrieval_mode=args.mode,
        rerank_enabled=args.rerank,
        rerank_candidates=args.candidates,
        filter_document=args.filter_doc,
        rewrite_query=args.rewrite_query,
        parent_child=args.parent_child,
    )
    cases = {c["id"]: c for c in load_golden_cases(args.dataset) if "id" in c}
    selected = [cases[cid] for cid in args.case_ids if cid in cases]
    if not selected:
        selected = list(cases.values())[:5]

    col, _ = _get_collection_and_embedder(cfg)
    rerank_flag = cfg.rerank_enabled if args.rerank is None else args.rerank
    filter_flag = cfg.filter_document if args.filter_doc is None else args.filter_doc
    rewrite_flag = cfg.rewrite_query if args.rewrite_query is None else args.rewrite_query
    parent_child_flag = cfg.parent_child if args.parent_child is None else args.parent_child

    print(f"Vector DB collection : {cfg.collection_name} ({col.count()} vectors)")
    print(f"Embedding model      : {cfg.embedding_provider} / {cfg.embedding_model}")
    print(f"Retrieval Mode       : {args.mode.upper()}")
    print(f"Re-ranking           : {'ENABLED (' + cfg.rerank_model + ', candidates=' + str(args.candidates) + ')' if rerank_flag else 'DISABLED'}")
    print(f"Doc/Version Filter   : {'ENABLED' if filter_flag else 'DISABLED'}")
    print(f"Query Rewriter       : {'ENABLED' if rewrite_flag else 'DISABLED'}")
    print(f"Parent-Child Expand  : {'ENABLED' if parent_child_flag else 'DISABLED'}")
    print(f"Evaluating {len(selected)} questions from {args.dataset}\n")

    hits = 0
    for case in selected:
        cid = case.get("id", "?")
        question = case.get("question") or case.get("input", "")
        expected_doc, expected_secs = extract_expected_sections(case)

        results = retrieve(
            question,
            top_k=args.top_k,
            config=cfg,
            mode=args.mode,
            rerank=rerank_flag,
            rerank_candidates=args.candidates,
            filter_document=filter_flag,
            rewrite_query=rewrite_flag,
            parent_child=parent_child_flag,
        )

        print("=" * 72)
        print(f"[{cid}] {question}")
        print(f"Expected source : {expected_doc} | Section(s): {', '.join(expected_secs) or 'N/A'}")
        status_tag = f"{args.mode} + cross-encoder" if rerank_flag else args.mode
        print(f"Returned top {args.top_k} ({status_tag}):")

        matched_positions: list[tuple[int, str]] = []
        for item in results:
            m = item["metadata"]
            doc = m.get("document_name", "")
            unit_type = m.get("unit_type", "")
            unit_num = str(m.get("unit_number", ""))
            unit_title = m.get("unit_title", "")
            page = m.get("page", "?")
            sim = item.get("similarity", 0.0)
            rrf = f" | rrf={item['rrf_score']:.5f}" if "rrf_score" in item else ""
            rerank_info = (
                f" | re-rank score={item['rerank_score']:.4f} (init #{item.get('initial_rank')})"
                if "rerank_score" in item
                else ""
            )

            label = f"Sec {unit_num}" if unit_type == "section" else f"{unit_type} {unit_num}"
            print(f"  #{item['rank']}  {doc} | {label}: {unit_title} | p.{page} | sim={sim:.4f}{rrf}{rerank_info}")

            doc_ok = (not expected_doc) or (doc == expected_doc)
            if doc_ok and unit_type == "section" and unit_num in expected_secs:
                matched_positions.append((item["rank"], f"Sec {unit_num}"))

        if matched_positions:
            hits += 1
            pos_str = ", ".join(f"#{r} ({sec})" for r, sec in matched_positions)
            print(f"Expected found? : YES -> at position(s): {pos_str}")
        else:
            print("Expected found? : NO (not in top 5)")

    print("=" * 72)
    print(f"Summary [{args.mode.upper()}{'+RERANK' if rerank_flag else ''}]: {hits} / {len(selected)} questions had an expected section in top {args.top_k}.")


if __name__ == "__main__":
    main()
