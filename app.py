"""Single-file Streamlit chat UI for the legal RAG chatbot.

Designed to run locally (`streamlit run app.py`) and deploy directly to
Streamlit Community Cloud (reading API keys from `st.secrets` or `.env`).
"""

from __future__ import annotations

import os
import time
from typing import Any

import streamlit as st

_SECRET_ENV_KEYS = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "NVIDIA_API_KEY",
    "NGC_API_KEY",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "CHAT_PROVIDER",
    "CHAT_MODEL",
    "CHROMA_DIR",
    "CHROMA_COLLECTION",
    "CHUNKS_PATH",
    "OLLAMA_BASE_URL",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
)


def _sync_streamlit_secrets() -> None:
    """Populate environment variables from `st.secrets` for Streamlit Cloud."""
    try:
        secrets = st.secrets
        for key in _SECRET_ENV_KEYS:
            if key in secrets and secrets[key]:
                os.environ[key] = str(secrets[key]).strip()
        for key in secrets:
            val = secrets[key]
            if isinstance(val, (str, int, float, bool)) and str(val).strip():
                os.environ[key] = str(val).strip()
    except Exception:
        # Local runs without .streamlit/secrets.toml rely on .env via rag_config.
        pass


# Run immediately on module import so all downstream modules see secrets in os.environ
_sync_streamlit_secrets()

from answer import NOT_FOUND_RESPONSE, answer
from auth import (
    authenticate_user,
    get_allowed_documents,
    get_role_description,
    usage_tracker,
)
from observability import tracer
from rag_config import load_rag_config


def _extract_source_details(chunk: dict[str, Any]) -> tuple[str, str, str]:
    """Return `(document, section, page)` strings from a retrieved chunk."""
    meta = chunk.get("metadata") or {}
    document = str(meta.get("document_name") or meta.get("source_file") or "Unknown Document")

    unit_type = str(meta.get("unit_type") or "").strip()
    unit_number = str(meta.get("unit_number") or "").strip()
    unit_title = str(meta.get("unit_title") or "").strip()

    if unit_type == "section" and unit_number:
        section = f"Section {unit_number}"
    elif unit_number:
        section = unit_number
    else:
        section = unit_type or "Excerpt"

    if unit_title:
        section = f"{section} — {unit_title}"

    page = meta.get("page")
    page_end = meta.get("page_end")
    if page is not None and page_end is not None and page_end != page:
        page_str = f"p. {page}-{page_end}"
    elif page is not None:
        page_str = f"p. {page}"
    else:
        page_str = "p. ?"

    return document, section, page_str


def _render_sources(chunks: list[dict[str, Any]]) -> None:
    """Display retrieved sources (document, section, page) under an answer."""
    if not chunks:
        st.caption("No source chunks were retrieved from the database.")
        return

    st.markdown("**Sources used:**")
    for chunk in chunks:
        document, section, page_str = _extract_source_details(chunk)
        if "rerank_score" in chunk:
            init_r = chunk.get("initial_rank")
            init_str = f", initial rank #{init_r}" if init_r else ""
            sim_text = f" *(re-rank score: {chunk['rerank_score']:.4f}{init_str})*"
        else:
            similarity = chunk.get("similarity")
            sim_text = f" *(similarity: {similarity:.4f})*" if isinstance(similarity, (int, float)) else ""
        rank = chunk.get("rank", "")
        prefix = f"#{rank} " if rank != "" else ""
        st.markdown(f"- {prefix}**Document:** {document} | **Section:** {section} | **Page:** {page_str}{sim_text}")

    with st.expander("View retrieved chunk text", expanded=False):
        for chunk in chunks:
            document, section, page_str = _extract_source_details(chunk)
            st.markdown(f"**{document} | {section} | {page_str}**")
            st.text(chunk.get("text", ""))


def _render_assistant_message(
    msg: dict[str, Any],
    msg_index: int = 0,
    user_info: dict[str, Any] | None = None,
) -> None:
    """Render assistant answer, sources, guardrail badge, trace inspector, and feedback controls."""
    if msg.get("error"):
        st.error(f"Could not generate an answer: {msg['error']}")
        return

    if msg.get("block_rail") == "quota":
        st.warning(f"⚠️ **Daily Question Limit Exceeded**\n\n{msg.get('content', '')}")
        return

    if msg.get("is_blocked"):
        rail = msg.get("block_rail") or "Guardrail"
        action = msg.get("block_action") or "blocked"
        st.error(
            f"🛡️ **Blocked by NVIDIA NeMo {rail.capitalize()} Rail** (`{action}`)\n\n"
            f"{msg.get('content', '')}"
        )
        return

    if msg.get("not_found"):
        st.warning(
            f"**Answer not found in documents:** {NOT_FOUND_RESPONSE}\n\n"
            "The retrieved sections did not contain enough information to answer your question."
        )
    else:
        st.markdown(msg.get("content", ""))

    if msg.get("guardrails_enabled"):
        provider = msg.get("guardrails_provider", "active")
        model = msg.get("guardrails_model", "")
        st.caption(f"🛡️ NeMo Guardrails: Passed & Verified ({provider}: `{model}`)")

    _render_sources(msg.get("chunks") or [])

    # 4. Observability Trace Inspector
    trace_id = msg.get("trace_id")
    if trace_id:
        trace_data = tracer.get_trace(trace_id) or {
            "trace_id": trace_id,
            "latency": msg.get("latency_seconds"),
            "chunks_count": len(msg.get("chunks", [])),
        }
        with st.expander(f"🔍 Observability Trace (`{trace_id[:8]}...`)", expanded=False):
            st.markdown(f"**Trace ID:** `{trace_id}`")
            trace_url = msg.get("trace_url") or (trace_data.get("trace_url") if trace_data else None)
            if trace_url and not trace_url.startswith("local://"):
                st.markdown(f"🔗 **[Open in Langfuse Cloud Dashboard]({trace_url})**")
            else:
                st.caption(f"📍 Stored locally in `logs/traces.jsonl` | CLI: `python view_trace.py {trace_id}`")

            if trace_data and "total_latency_seconds" in trace_data:
                st.markdown(
                    f"**User:** `@{trace_data.get('user', 'user')}` | "
                    f"**Role:** `{trace_data.get('role', 'general')}` | "
                    f"**Latency:** `{trace_data.get('total_latency_seconds')}s`"
                )
            st.json(trace_data, expanded=False)

    # 5. User Feedback (Thumbs Up / Thumbs Down)
    if trace_id and not msg.get("is_blocked") and not msg.get("error"):
        st.markdown("---")
        fb_col1, fb_col2 = st.columns([2, 5])
        with fb_col1:
            st.caption("Was this response accurate?")
        with fb_col2:
            fb_key = f"feedback_{trace_id}_{msg_index}"
            # Streamlit st.feedback returns 0 for thumbs down, 1 for thumbs up
            fb_score = st.feedback("thumbs", key=fb_key)
            if fb_score is not None:
                fb_type = "thumbs_up" if fb_score == 1 else "thumbs_down"
                if msg.get("feedback_state") != fb_type:
                    msg["feedback_state"] = fb_type
                    u_name = user_info.get("username", "user") if user_info else "user"
                    u_role = user_info.get("role", "general") if user_info else "general"
                    res = tracer.record_feedback(
                        trace_id=trace_id,
                        feedback=fb_type,
                        question=msg.get("question", ""),
                        answer=msg.get("content", ""),
                        chunks=msg.get("chunks", []),
                        user=u_name,
                        role=u_role,
                        comment="Recorded via Streamlit thumbs feedback",
                    )
                    if fb_type == "thumbs_up":
                        st.toast("👍 Thank you for your positive feedback!", icon="✅")
                    else:
                        st.toast("👎 Question saved to Golden Dataset Review List!", icon="📋")

            if msg.get("feedback_state") == "thumbs_up":
                st.caption("✅ Feedback: Helpful (Thumbs Up)")
            elif msg.get("feedback_state") == "thumbs_down":
                st.caption("📋 Flagged: Added to `golden_review_list.json` for review")


def main() -> None:
    st.set_page_config(page_title="Legal RAG Chatbot", page_icon="⚖️", layout="centered")
    _sync_streamlit_secrets()

    st.title("Legal Document RAG Chatbot")
    st.caption("Ask questions grounded strictly in your indexed legal documents, with section and page citations.")

    if "authenticated_user" not in st.session_state:
        st.session_state.authenticated_user = None

    # 1. Login Gate: Nobody can ask a question without signing in
    if st.session_state.authenticated_user is None:
        st.markdown("### 🔐 Authentication Required")
        st.markdown("Please sign in with your credentials to access the legal statutory database.")

        with st.form("login_form", clear_on_submit=False):
            username_input = st.text_input("Username", placeholder="e.g. corp_user, tax_user, partner_user")
            password_input = st.text_input("Password", type="password", placeholder="Enter your secret password")
            submit_login = st.form_submit_button("Sign In", use_container_width=True)

            if submit_login:
                user_info = authenticate_user(username_input, password_input)
                if user_info:
                    st.session_state.authenticated_user = user_info
                    st.session_state.messages = []
                    st.success(f"Welcome, {user_info['name']}! Signed in as `{user_info['role']}`.")
                    st.rerun()
                else:
                    st.error("Authentication failed. Invalid username or password.")

        with st.expander("ℹ️ Role-Based Access & Secret Configuration", expanded=True):
            st.markdown(
                "- **Security**: All user credentials and passwords reside in `.streamlit/secrets.toml`.\n"
                "- **Role-Based Search**: Chunks from unauthorized documents are blocked before search.\n"
                "- **Daily Quota**: Each account has a daily question cap enforced automatically.\n"
                "- **Demo Accounts**: `corp_user`, `tax_user`, `compliance_user`, `partner_user`, `capped_user`."
            )
        st.chat_input("Please sign in above to ask questions...", disabled=True)
        return

    # User is authenticated
    user = st.session_state.authenticated_user
    allowed_docs = get_allowed_documents(user["role"])
    role_desc = get_role_description(user["role"])
    current_usage = usage_tracker.get_usage(user["username"])

    with st.sidebar:
        st.subheader("👤 User Profile")
        st.markdown(f"**Name:** {user['name']}\n\n**Username:** `@{user['username']}`")
        st.markdown(f"**Role:** `{user['role']}`\n\n*({role_desc})*")
        st.markdown(f"**Daily Quota:** `{current_usage} / {user['daily_limit']}` questions used today")

        with st.expander("📄 Permitted Documents", expanded=False):
            if allowed_docs:
                for d in allowed_docs:
                    st.markdown(f"- `{d}`")
            else:
                st.caption("No documents permitted for this role.")

        if st.button("Sign Out", use_container_width=True):
            st.session_state.authenticated_user = None
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.header("Settings")
        search_mode = st.radio("Retrieval Mode", options=["Hybrid", "Vector"], index=0, help="Hybrid combines vector embedding with BM25 keyword matching via Reciprocal Rank Fusion.")
        enable_rerank = st.toggle(
            "Cross-Encoder Re-ranking",
            value=True,
            help="Retrieve top 30 candidates and re-rank with MS-MARCO Cross-Encoder down to top 5.",
        )
        enable_doc_filter = st.toggle(
            "Document & Version Filter",
            value=True,
            help="Detect named document/version in query and filter metadata. Defaults to current version unless older is requested.",
        )
        enable_rewrite = st.toggle(
            "Query Rewriter",
            value=True,
            help="Automatically expand short or vague questions into full statutory search queries.",
        )
        enable_parent_child = st.toggle(
            "Parent-Child Expansion",
            value=True,
            help="Index small child chunks for search precision, but expand and merge into the full parent section when answering.",
        )
        enable_guardrails = st.toggle(
            "NVIDIA NeMo Guardrails",
            value=True,
            help="Enforce input rails (topic control & jailbreak blocking) and output rails (content safety & hallucination checks).",
        )
        top_k = st.slider("Chunks to return (top-k)", min_value=1, max_value=10, value=5)
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        try:
            cfg = load_rag_config()
            st.divider()
            st.caption(f"Chat model: `{cfg.chat_provider}/{cfg.chat_model}`")
            st.caption(f"Embedding model: `{cfg.embedding_provider}/{cfg.embedding_model}`")
            st.caption(f"NeMo Guardrails: `{'Enabled' if enable_guardrails else 'Disabled'}`")
            st.caption(f"Re-ranker: `{'Enabled' if enable_rerank else 'Disabled'}`")
            st.caption(f"Doc Filter: `{'Enabled' if enable_doc_filter else 'Disabled'}`")
            st.caption(f"Rewriter: `{'Enabled' if enable_rewrite else 'Disabled'}`")
            st.caption(f"Parent-Child: `{'Enabled' if enable_parent_child else 'Disabled'}`")
            st.caption(f"Collection: `{cfg.collection_name}`")
        except Exception as exc:
            st.error(f"Configuration error: {exc}")

        st.divider()
        review_items = tracer.get_review_items()
        st.caption(f"📋 **Golden Review List:** `{len(review_items)} flagged`")
        if review_items:
            with st.expander(f"Review Flagged Questions ({len(review_items)})", expanded=False):
                for r_idx, r_item in enumerate(review_items[-5:], 1):
                    st.markdown(f"**{r_idx}.** {r_item.get('question', '')[:60]}...")
                    st.caption(f"Flagged by @{r_item.get('user', 'user')} at {r_item.get('timestamp', '')[:16]}")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                _render_assistant_message(msg, msg_index=idx, user_info=user)

    if prompt := st.chat_input(f"Ask a question about your authorized legal documents..."):
        user_msg = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_msg)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # 2. Daily Quota Check BEFORE search or inference
            is_cap_ok, cur_usage, daily_lim, cap_msg = usage_tracker.check_daily_cap(
                user["username"], user["daily_limit"]
            )
            if not is_cap_ok:
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "question": prompt,
                    "content": cap_msg,
                    "not_found": False,
                    "is_blocked": True,
                    "block_rail": "quota",
                    "block_action": "daily_limit_exceeded",
                    "chunks": [],
                    "error": None,
                }
                _render_assistant_message(assistant_msg, msg_index=len(st.session_state.messages), user_info=user)
            else:
                # Increment user's question usage for today
                usage_tracker.increment_usage(user["username"])

                # 3. Observability: Start trace
                t0 = time.perf_counter()
                trace_id = tracer.start_trace(
                    question=prompt,
                    user=user["username"],
                    role=user["role"],
                )

                with st.spinner("Searching documents and generating answer..."):
                    try:
                        res = answer(
                            prompt,
                            top_k=top_k,
                            mode=search_mode.lower(),
                            rerank=enable_rerank,
                            filter_document=enable_doc_filter,
                            rewrite_query=enable_rewrite,
                            parent_child=enable_parent_child,
                            guardrails=enable_guardrails,
                            allowed_documents=allowed_docs,
                        )
                        latency_total = round(time.perf_counter() - t0, 3)

                        # Observability: Log retrieval span and complete trace
                        tracer.log_retrieval(
                            trace_id=trace_id,
                            search_query=res.get("rewritten_query") or prompt,
                            retrieved_chunks=res.get("chunks", []),
                            latency_seconds=res.get("search_latency_seconds", 0.0),
                            retrieval_mode=search_mode.lower(),
                        )
                        tracer.end_trace(
                            trace_id=trace_id,
                            answer=res.get("answer", ""),
                            retrieved_chunks=res.get("chunks", []),
                            latency_seconds=latency_total,
                            is_blocked=bool(res.get("is_blocked")),
                            not_found=bool(res.get("not_found")),
                        )

                        content = res["answer"]
                        notes = []
                        if res.get("was_rewritten"):
                            notes.append(f"🔍 *Expanded search query: \"{res.get('rewritten_query')}\"*")
                        if res.get("detected_document"):
                            ver_str = f" ({res.get('detected_version')})" if res.get('detected_version') else ""
                            notes.append(f"📑 *Filtered to: {res.get('detected_document')}{ver_str}*")
                        if notes:
                            content = "\n\n".join(notes) + "\n\n" + content

                        assistant_msg = {
                            "role": "assistant",
                            "question": prompt,
                            "content": content,
                            "not_found": bool(res.get("not_found")),
                            "is_blocked": bool(res.get("is_blocked")),
                            "block_rail": res.get("block_rail"),
                            "block_action": res.get("block_action"),
                            "guardrails_enabled": bool(res.get("guardrails_enabled")),
                            "guardrails_provider": res.get("guardrails_provider"),
                            "guardrails_model": res.get("guardrails_model"),
                            "chunks": res.get("chunks", []),
                            "trace_id": trace_id,
                            "latency_seconds": latency_total,
                            "error": None,
                        }
                    except Exception as exc:
                        latency_total = round(time.perf_counter() - t0, 3)
                        tracer.end_trace(
                            trace_id=trace_id,
                            answer=f"Error: {exc}",
                            retrieved_chunks=[],
                            latency_seconds=latency_total,
                            is_blocked=False,
                            not_found=False,
                        )
                        assistant_msg = {
                            "role": "assistant",
                            "question": prompt,
                            "content": "",
                            "not_found": False,
                            "is_blocked": False,
                            "block_rail": None,
                            "block_action": None,
                            "guardrails_enabled": False,
                            "guardrails_provider": None,
                            "guardrails_model": None,
                            "chunks": [],
                            "trace_id": trace_id,
                            "latency_seconds": latency_total,
                            "error": str(exc),
                        }
                _render_assistant_message(assistant_msg, msg_index=len(st.session_state.messages), user_info=user)

        st.session_state.messages.append(assistant_msg)


if __name__ == "__main__":
    main()

