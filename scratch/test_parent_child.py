import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from answer import format_citation, _call_chat_model, SYSTEM_PROMPT, NOT_FOUND_RESPONSE
from retrieve import retrieve, load_rag_config

cfg = load_rag_config()
q = "Explain the legal requirements and procedure for incorporating a private limited company under Section 7."
res = retrieve(q, top_k=10, mode="hybrid", rerank=True, filter_document=True, rewrite_query=True)

chunks_all = [json.loads(line) for line in open("parsed_docs/chunks.jsonl", encoding="utf-8")]
parent_map = {}
for c in chunks_all:
    key = (c["metadata"].get("document_name"), c["metadata"].get("unit_type"), c["metadata"].get("unit_number"))
    parent_map.setdefault(key, []).append(c)

seen = set()
parent_chunks = []
for r in res:
    key = (r["metadata"].get("document_name"), r["metadata"].get("unit_type"), r["metadata"].get("unit_number"))
    if key in seen:
        continue
    seen.add(key)
    siblings = sorted(parent_map.get(key, [r]), key=lambda x: x.get("metadata", {}).get("part", 1))
    merged_text = "\n".join(s["text"] for s in siblings)
    parent_r = dict(r)
    parent_r["text"] = merged_text
    parent_r["is_parent"] = True
    parent_r["total_parts"] = len(siblings)
    parent_chunks.append(parent_r)
    if len(parent_chunks) >= 5:
        break

print(f"Parent chunks count: {len(parent_chunks)}")
for p in parent_chunks:
    print(f" - {p['metadata'].get('unit_type')} {p['metadata'].get('unit_number')}: {len(p['text'])} chars (parts={p['total_parts']})")

context_blocks = [f"CITATION LABEL: {format_citation(c['metadata'])}\n{c['text']}" for c in parent_chunks]
user_prompt = (
    "CONTEXT CHUNKS:\n\n"
    + "\n\n---\n\n".join(context_blocks)
    + f"\n\nQUESTION: {q}\n\n"
    f"Answer strictly from the CONTEXT CHUNKS above, citing the exact CITATION LABEL after every claim, "
    f"or reply exactly '{NOT_FOUND_RESPONSE}' if the chunks do not contain the answer."
)
ans = _call_chat_model(cfg, SYSTEM_PROMPT, user_prompt)
print("\n=== GENERATED ANSWER ===")
print(ans)
