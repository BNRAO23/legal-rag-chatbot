"""Shared RAG configuration for indexing and querying legal chunks.

Both `build_vector_db.py` and downstream question-answering scripts should import
`load_rag_config()` and `Embedder` from this module so indexing and querying
always use the exact same embedding provider, model name, and ChromaDB collection.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv()


def sync_streamlit_secrets() -> None:
    """Populate environment variables from Streamlit secrets, if available."""
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            for key in st.secrets:
                val = st.secrets[key]
                if isinstance(val, (str, int, float, bool)) and str(val).strip():
                    os.environ[key] = str(val).strip()
    except Exception:
        pass


sync_streamlit_secrets()


def get_secret(key: str, default: str | None = None) -> str | None:
    """Read a setting or secret from Streamlit secrets, falling back to os.environ / .env."""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            val = st.secrets[key]
            if val is not None and str(val).strip():
                return str(val).strip()
    except Exception:
        pass
    return os.getenv(key, default)


ProviderName = Literal["google", "openai", "ollama", "sentence_transformers"]

DEFAULT_MODELS: dict[str, str] = {
    "google": "gemini-embedding-001",
    "openai": "text-embedding-3-small",
    "ollama": "llama3.2:latest",
    "sentence_transformers": "sentence-transformers/all-MiniLM-L6-v2",
}

DEFAULT_CHAT_MODELS: dict[str, str] = {
    "google": "gemini-3.5-flash-lite",
    "openai": "gpt-4o-mini",
    "ollama": "llama3.2:latest",
}


@dataclass(frozen=True)
class RagConfig:
    chunks_path: Path
    chroma_dir: Path
    collection_name: str
    embedding_provider: str
    embedding_model: str
    chat_provider: str
    chat_model: str
    batch_size: int = 64
    ollama_base_url: str = "http://localhost:11434"
    retrieval_mode: str = "vector"
    rrf_k: int = 60
    rerank_enabled: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 30
    filter_document: bool = False
    rewrite_query: bool = False
    parent_child: bool = False
    guardrails_enabled: bool = True


def detect_keyed_provider() -> tuple[str, str] | None:
    """Return (provider, default_model) for an available API key, if present."""
    if get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY"):
        return "google", DEFAULT_MODELS["google"]
    if get_secret("OPENAI_API_KEY"):
        return "openai", DEFAULT_MODELS["openai"]
    return None


def load_rag_config(
    *,
    provider: str | None = None,
    model: str | None = None,
    chat_provider: str | None = None,
    chat_model: str | None = None,
    chunks_path: Path | None = None,
    chroma_dir: Path | None = None,
    collection_name: str | None = None,
    batch_size: int | None = None,
    retrieval_mode: str | None = None,
    rrf_k: int | None = None,
    rerank_enabled: bool | None = None,
    rerank_model: str | None = None,
    rerank_candidates: int | None = None,
    filter_document: bool | None = None,
    rewrite_query: bool | None = None,
    parent_child: bool | None = None,
    guardrails_enabled: bool | None = None,
) -> RagConfig:
    """Resolve configuration from CLI overrides, Streamlit secrets, environment/.env, and defaults."""
    resolved_provider = (provider or get_secret("EMBEDDING_PROVIDER") or "").strip().lower()
    resolved_model = (model or get_secret("EMBEDDING_MODEL") or "").strip()

    if not resolved_provider:
        detected = detect_keyed_provider()
        if detected is not None:
            resolved_provider, detected_model = detected
            if not resolved_model:
                resolved_model = detected_model

    if not resolved_provider:
        raise RuntimeError(
            "No embedding provider API key found in Streamlit secrets, environment, or .env "
            "(checked GEMINI_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY). "
            "Set one of those keys in Streamlit secrets or .env (see .env.example) or pass "
            "--provider sentence_transformers / --provider ollama explicitly."
        )

    if not resolved_model:
        if resolved_provider not in DEFAULT_MODELS:
            raise ValueError(f"Unsupported embedding provider: {resolved_provider}")
        resolved_model = DEFAULT_MODELS[resolved_provider]

    resolved_chat_provider = (chat_provider or get_secret("CHAT_PROVIDER") or "").strip().lower()
    if not resolved_chat_provider:
        detected_chat = detect_keyed_provider()
        resolved_chat_provider = detected_chat[0] if detected_chat else "ollama"
    resolved_chat_model = (chat_model or get_secret("CHAT_MODEL") or "").strip()
    if not resolved_chat_model:
        resolved_chat_model = DEFAULT_CHAT_MODELS.get(resolved_chat_provider, DEFAULT_CHAT_MODELS["ollama"])

    resolved_mode = (retrieval_mode or get_secret("RETRIEVAL_MODE") or "vector").strip().lower()
    if resolved_mode not in ("vector", "hybrid"):
        resolved_mode = "vector"
    resolved_rrf_k = rrf_k if rrf_k is not None else int(get_secret("RRF_K", "60"))

    if rerank_enabled is not None:
        resolved_rerank_enabled = bool(rerank_enabled)
    else:
        resolved_rerank_enabled = (get_secret("RERANK_ENABLED", "") or "").strip().lower() in ("true", "1", "yes")

    resolved_rerank_model = (
        rerank_model
        or get_secret("RERANK_MODEL")
        or "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ).strip()

    resolved_rerank_candidates = (
        rerank_candidates
        if rerank_candidates is not None
        else int(get_secret("RERANK_CANDIDATES", get_secret("RERANK_TOP_N", "30")))
    )

    if filter_document is not None:
        resolved_filter_document = bool(filter_document)
    else:
        resolved_filter_document = (get_secret("FILTER_DOCUMENT", "") or "").strip().lower() in ("true", "1", "yes")

    if rewrite_query is not None:
        resolved_rewrite_query = bool(rewrite_query)
    else:
        resolved_rewrite_query = (get_secret("REWRITE_QUERY", "") or "").strip().lower() in ("true", "1", "yes")

    if parent_child is not None:
        resolved_parent_child = bool(parent_child)
    else:
        resolved_parent_child = (get_secret("PARENT_CHILD", "") or "").strip().lower() in ("true", "1", "yes")

    if guardrails_enabled is not None:
        resolved_guardrails_enabled = bool(guardrails_enabled)
    else:
        resolved_guardrails_enabled = (get_secret("GUARDRAILS_ENABLED", "true") or "true").strip().lower() in ("true", "1", "yes")

    return RagConfig(
        chunks_path=chunks_path or Path(get_secret("CHUNKS_PATH", "parsed_docs/chunks.jsonl")),
        chroma_dir=chroma_dir or Path(get_secret("CHROMA_DIR", "chroma_db")),
        collection_name=collection_name or get_secret("CHROMA_COLLECTION", "ca_legal_rag"),
        embedding_provider=resolved_provider,
        embedding_model=resolved_model,
        chat_provider=resolved_chat_provider,
        chat_model=resolved_chat_model,
        batch_size=batch_size or int(get_secret("EMBED_BATCH_SIZE", "64")),
        ollama_base_url=get_secret("OLLAMA_BASE_URL", "http://localhost:11434"),
        retrieval_mode=resolved_mode,
        rrf_k=resolved_rrf_k,
        rerank_enabled=resolved_rerank_enabled,
        rerank_model=resolved_rerank_model,
        rerank_candidates=resolved_rerank_candidates,
        filter_document=resolved_filter_document,
        rewrite_query=resolved_rewrite_query,
        parent_child=resolved_parent_child,
        guardrails_enabled=resolved_guardrails_enabled,
    )


class Embedder:
    """Unified batch embedder so indexing and querying share one model interface."""

    def __init__(self, config: RagConfig) -> None:
        self.provider = config.embedding_provider
        self.model = config.embedding_model
        self._client = None
        self._st_model = None

        if self.provider == "google":
            from google import genai

            api_key = get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required for provider='google'.")
            self._client = genai.Client(api_key=api_key)
        elif self.provider == "openai":
            from openai import OpenAI

            api_key = get_secret("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is required for provider='openai'.")
            self._client = OpenAI(api_key=api_key)
        elif self.provider == "ollama":
            import ollama

            self._client = ollama.Client(host=config.ollama_base_url)
        elif self.provider == "sentence_transformers":
            from sentence_transformers import SentenceTransformer

            try:
                self._st_model = SentenceTransformer(self.model, local_files_only=True)
            except Exception:
                self._st_model = SentenceTransformer(self.model)
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document chunk texts."""
        return self._embed(texts, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, query: str) -> list[float]:
        """Embed a single user question using the same configured model."""
        return self._embed([query], task_type="RETRIEVAL_QUERY")[0]

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        if not texts:
            return []

        if self.provider == "google":
            from google.genai import types

            for attempt in range(8):
                try:
                    resp = self._client.models.embed_content(
                        model=self.model,
                        contents=texts,
                        config=types.EmbedContentConfig(task_type=task_type),
                    )
                    return [list(e.values) for e in resp.embeddings]
                except Exception as exc:
                    msg = str(exc)
                    if ("429" in msg or "RESOURCE_EXHAUSTED" in msg) and attempt < 7:
                        if "PerDay" in msg:
                            raise RuntimeError(
                                "Gemini Free Tier daily embedding quota (1,000 requests/day) exhausted. "
                                "Already-embedded chunks are saved in ChromaDB. Re-run `python build_vector_db.py` "
                                "after quota resets, enable billing, or rebuild locally with "
                                "`python build_vector_db.py --provider sentence_transformers --reset`."
                            ) from exc
                        m = re.search(r"retry in (\d+(?:\.\d+)?)s", msg, re.IGNORECASE)
                        wait_s = float(m.group(1)) + 2.0 if m else min(65.0, 15.0 * (attempt + 1))
                        print(f"\nRate limit hit; sleeping {wait_s:.1f}s before retry ({attempt + 1}/7)...")
                        time.sleep(wait_s)
                        continue
                    raise

        if self.provider == "openai":
            resp = self._client.embeddings.create(model=self.model, input=texts)
            return [list(item.embedding) for item in resp.data]

        if self.provider == "ollama":
            resp = self._client.embed(model=self.model, input=texts)
            return [list(v) for v in resp.embeddings]

        if self.provider == "sentence_transformers":
            vecs = self._st_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return vecs.tolist()

        raise ValueError(f"Unsupported embedding provider: {self.provider}")

