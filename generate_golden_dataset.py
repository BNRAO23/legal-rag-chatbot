"""Generate a synthetic golden evaluation dataset using DeepEval's Synthesizer.

Builds ~30 answerable questions (mixing single-section lookups and multi-section
questions across Companies Act 2013, FEMA 1999, and Income-tax Act 2025) from
`parsed_docs/chunks.jsonl`, plus 5 unanswerable questions with expected output
`NOT_FOUND`. Saves the editable JSON dataset to `synthetic_golden_dataset.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from deepeval.models import OllamaModel
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import EvolutionConfig, FiltrationConfig, StylingConfig
from deepeval.synthesizer.types import Evolution


# Single-section lookup seeds: (document_name, unit_number)
SINGLE_SECTION_SEEDS: list[tuple[str, str]] = [
    ("Companies Act, 2013", "7"),       # Incorporation of company
    ("Companies Act, 2013", "42"),      # Private placement
    ("Companies Act, 2013", "63"),      # Issue of bonus shares
    ("Companies Act, 2013", "135"),     # Corporate Social Responsibility
    ("Companies Act, 2013", "185"),     # Loans to directors
    ("Companies Act, 2013", "188"),     # Related party transactions
    ("Foreign Exchange Management Act, 1999", "3"),   # Dealing in foreign exchange
    ("Foreign Exchange Management Act, 1999", "7"),   # Export of goods and services
    ("Foreign Exchange Management Act, 1999", "8"),   # Realisation & repatriation
    ("Foreign Exchange Management Act, 1999", "10"),  # Authorised person
    ("Foreign Exchange Management Act, 1999", "42"),  # Contravention by companies
    ("Income-tax Act, 2025", "1"),      # Short title, extent and commencement
    ("Income-tax Act, 2025", "3"),      # Definition of tax year
    ("Income-tax Act, 2025", "13"),     # Heads of income
    ("Income-tax Act, 2025", "18"),     # Profits in lieu of salary
    ("Income-tax Act, 2025", "19"),     # Deductions from salaries
]

# Multi-section seeds: pairs of (document_name, unit_number) that complement each other
MULTI_SECTION_SEEDS: list[list[tuple[str, str]]] = [
    [("Companies Act, 2013", "4"), ("Companies Act, 2013", "5")],       # MOA & AOA
    [("Companies Act, 2013", "173"), ("Companies Act, 2013", "174")],   # Board meetings & quorum
    [("Companies Act, 2013", "139"), ("Companies Act, 2013", "140")],   # Auditors appointment & removal
    [("Companies Act, 2013", "129"), ("Companies Act, 2013", "137")],   # Financial statements & filing
    [("Companies Act, 2013", "143"), ("Companies Act, 2013", "145")],   # Auditor duties & signing
    [("Companies Act, 2013", "62"), ("Companies Act, 2013", "63")],     # Rights issue & bonus shares
    [("Foreign Exchange Management Act, 1999", "5"), ("Foreign Exchange Management Act, 1999", "6")],     # Current & capital account
    [("Foreign Exchange Management Act, 1999", "13"), ("Foreign Exchange Management Act, 1999", "15")],   # Penalties & compounding
    [("Foreign Exchange Management Act, 1999", "11"), ("Foreign Exchange Management Act, 1999", "12")],   # RBI directions & inspection
    [("Foreign Exchange Management Act, 1999", "7"), ("Foreign Exchange Management Act, 1999", "8")],     # Export & repatriation
    [("Foreign Exchange Management Act, 1999", "2"), ("Foreign Exchange Management Act, 1999", "49")],    # Definitions & repeal/saving
    [("Income-tax Act, 2025", "4"), ("Income-tax Act, 2025", "5")],     # Charge & scope of total income
    [("Income-tax Act, 2025", "5"), ("Income-tax Act, 2025", "6")],     # Scope & residence in India
    [("Income-tax Act, 2025", "15"), ("Income-tax Act, 2025", "16")],   # Salaries & income from salary
    [("Income-tax Act, 2025", "67"), ("Income-tax Act, 2025", "70")],   # Capital gains & exempt transfers
    [("Income-tax Act, 2025", "393"), ("Income-tax Act, 2025", "394")], # TDS & TCS
]

UNANSWERABLE_QUESTIONS: list[dict[str, str]] = [
    {
        "input": "Under the Central Goods and Services Tax (CGST) Act, 2017, what is the GST rate and SAC code for corporate legal consultancy services?",
        "reason": "GST / CGST Act, 2017 is not part of the project corpus (only Companies Act 2013, FEMA 1999, and Income-tax Act 2025 are included).",
    },
    {
        "input": "What is the current RBI Liberalised Remittance Scheme (LRS) annual USD limit per financial year for resident individuals?",
        "reason": "The FEMA Act, 1999 statute text does not state the RBI Master Direction annual USD limit for LRS.",
    },
    {
        "input": "Under the Insolvency and Bankruptcy Code, 2016, what is the maximum statutory timeline including extensions for completing a Corporate Insolvency Resolution Process (CIRP)?",
        "reason": "The Insolvency and Bankruptcy Code, 2016 is not present in the source documents.",
    },
    {
        "input": "What exact MCA filing fee and Maharashtra stamp duty apply when filing Form SPICe+ (INC-32) for a company with Rs. 10 lakh authorised capital?",
        "reason": "MCA fee schedules, rule-level filing fees, and state stamp duty rates are outside the Companies Act, 2013 PDF.",
    },
    {
        "input": "Under Regulation 23 of the SEBI (Listing Obligations and Disclosure Requirements) Regulations, 2015, what is the numerical materiality threshold for listed-company related party transactions?",
        "reason": "SEBI LODR Regulations, 2015 are not included in the project's source documents.",
    },
]


def load_section_index(chunks_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Load the first chunk of each statutory section keyed by (document_name, unit_number)."""
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for line in chunks_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        chunk = json.loads(line)
        meta = chunk["metadata"]
        if meta["unit_type"] == "section" and meta["part"] == 1:
            index[(meta["document_name"], str(meta["unit_number"]))] = chunk
    return index


def format_context_passage(chunk: dict[str, Any]) -> str:
    """Attach clear citation header to each chunk passed to DeepEval Synthesizer."""
    m = chunk["metadata"]
    header = f"[{m['document_name']} ({m['version_or_year']}) | Section {m['unit_number']}: {m['unit_title']} | Page {m['page']}]"
    return f"{header}\n{chunk['text']}"


def build_source_entry(chunk: dict[str, Any]) -> dict[str, Any]:
    m = chunk["metadata"]
    return {
        "chunk_id": chunk["chunk_id"],
        "document_name": m["document_name"],
        "source_file": m["source_file"],
        "version_or_year": m["version_or_year"],
        "unit_number": m["unit_number"],
        "unit_title": m["unit_title"],
        "page": m["page"],
    }


def synthesize_batch(
    model: OllamaModel,
    context_groups: list[list[str]],
    source_meta_groups: list[list[dict[str, Any]]],
    question_type: str,
    evolution_config: EvolutionConfig,
    styling_config: StylingConfig,
    target_count: int,
) -> list[dict[str, Any]]:
    """Run DeepEval Synthesizer on a batch of contexts and attach source metadata."""
    synthesizer = Synthesizer(
        model=model,
        async_mode=False,
        filtration_config=FiltrationConfig(
            critic_model=model,
            synthetic_input_quality_threshold=0.3,
            max_quality_retries=1,
        ),
        evolution_config=evolution_config,
        styling_config=styling_config,
    )

    source_labels = [
        " + ".join(f"{s['document_name']} Sec {s['unit_number']} (p.{s['page']})" for s in srcs)
        for srcs in source_meta_groups
    ]
    chunk_source_lists = [
        [f"{s['document_name']} | Section {s['unit_number']} ({s['unit_title']}) | p.{s['page']}" for s in srcs]
        for srcs in source_meta_groups
    ]

    records: list[dict[str, Any]] = []
    # Process one context at a time so we keep exact 1-to-1 alignment with source metadata
    # and can stop right at target_count without wasting local LLM calls.
    for idx, (ctx_list, src_metas, src_label, chunk_srcs) in enumerate(
        zip(context_groups, source_meta_groups, source_labels, chunk_source_lists),
        start=1,
    ):
        if len(records) >= target_count:
            break
        print(f"[{question_type}] ({idx}/{len(context_groups)}) Synthesizing from: {src_label}")
        try:
            goldens = synthesizer.generate_goldens_from_contexts(
                contexts=[ctx_list],
                include_expected_output=True,
                max_goldens_per_context=1,
                source_files=[src_label],
                context_chunk_source_files=[chunk_srcs],
                _send_data=False,
            )
        except Exception as exc:
            print(f"  Warning: skipped due to synthesis error: {exc}")
            continue

        for g in goldens:
            records.append(
                {
                    "input": g.input,
                    "expected_output": g.expected_output,
                    "context": g.context,
                    "source_file": g.source_file,
                    "additional_metadata": {
                        "question_type": question_type,
                        "sources": src_metas,
                        "evolutions": (g.additional_metadata or {}).get("evolutions", []),
                        "synthetic_input_quality": (g.additional_metadata or {}).get("synthetic_input_quality"),
                    },
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic RAG golden dataset with DeepEval.")
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("parsed_docs/chunks.jsonl"),
        help="Input chunks JSONL produced by chunk_documents.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("synthetic_golden_dataset.json"),
        help="Editable output JSON file path.",
    )
    parser.add_argument(
        "--model",
        default="llama3.2:latest",
        help="Local Ollama model name (default: llama3.2:latest).",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434",
        help="Ollama base URL (default: http://localhost:11434).",
    )
    parser.add_argument(
        "--single-count",
        type=int,
        default=15,
        help="Target number of single-section lookup questions (default: 15).",
    )
    parser.add_argument(
        "--multi-count",
        type=int,
        default=15,
        help="Target number of multi-section questions (default: 15).",
    )
    args = parser.parse_args()

    section_index = load_section_index(args.chunks)
    model = OllamaModel(model=args.model, base_url=args.base_url, temperature=0.2)

    styling = StylingConfig(
        scenario="Indian corporate, foreign exchange, and income-tax statutory compliance Q&A.",
        task="Ask and answer legal compliance questions strictly grounded in the provided statutory sections.",
        input_format="A clear, specific question a Chartered Accountant or legal researcher would ask.",
        expected_output_format="Concise, accurate answer grounded strictly in the provided context, citing exact section numbers and conditions.",
    )

    # 1. Single-section lookup contexts
    single_contexts: list[list[str]] = []
    single_sources: list[list[dict[str, Any]]] = []
    for key in SINGLE_SECTION_SEEDS:
        chunk = section_index.get(key)
        if chunk is not None:
            single_contexts.append([format_context_passage(chunk)])
            single_sources.append([build_source_entry(chunk)])

    single_records = synthesize_batch(
        model=model,
        context_groups=single_contexts,
        source_meta_groups=single_sources,
        question_type="single_section_lookup",
        evolution_config=EvolutionConfig(num_evolutions=0),
        styling_config=styling,
        target_count=args.single_count,
    )

    # 2. Multi-section contexts
    multi_contexts: list[list[str]] = []
    multi_sources: list[list[dict[str, Any]]] = []
    for pair in MULTI_SECTION_SEEDS:
        chunks_pair = [section_index.get(key) for key in pair]
        if all(c is not None for c in chunks_pair):
            multi_contexts.append([format_context_passage(c) for c in chunks_pair if c is not None])
            multi_sources.append([build_source_entry(c) for c in chunks_pair if c is not None])

    multi_records = synthesize_batch(
        model=model,
        context_groups=multi_contexts,
        source_meta_groups=multi_sources,
        question_type="multi_section",
        evolution_config=EvolutionConfig(
            num_evolutions=1,
            evolutions={
                Evolution.MULTICONTEXT: 0.6,
                Evolution.COMPARATIVE: 0.2,
                Evolution.REASONING: 0.2,
            },
        ),
        styling_config=styling,
        target_count=args.multi_count,
    )

    # 3. Unanswerable questions (expected_output = NOT_FOUND)
    unanswerable_records = [
        {
            "input": item["input"],
            "expected_output": "NOT_FOUND",
            "context": [],
            "source_file": None,
            "additional_metadata": {
                "question_type": "unanswerable",
                "sources": [],
                "unanswerable_reason": item["reason"],
            },
        }
        for item in UNANSWERABLE_QUESTIONS
    ]

    all_records = single_records + multi_records + unanswerable_records
    for idx, rec in enumerate(all_records, start=1):
        rec["additional_metadata"]["id"] = f"SYN-{idx:03d}"

    args.output.write_text(json.dumps(all_records, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== DATASET SUMMARY ===")
    print(f"Saved to       : {args.output.resolve()}")
    print(f"Single-section : {len(single_records)}")
    print(f"Multi-section  : {len(multi_records)}")
    print(f"NOT_FOUND      : {len(unanswerable_records)}")
    print(f"Total goldens  : {len(all_records)}")

    # Print 5 representative samples: 2 single, 2 multi, 1 NOT_FOUND
    samples: list[dict[str, Any]] = []
    samples.extend(single_records[:2])
    samples.extend(multi_records[:2])
    samples.extend(unanswerable_records[:1])

    print("\n=== FIVE SAMPLE GOLDENS ===")
    for i, rec in enumerate(samples, start=1):
        meta = rec["additional_metadata"]
        print(f"\n--- Sample {i} [{meta['id']} | {meta['question_type']}] ---")
        print(f"Question        : {rec['input']}")
        print(f"Expected Answer : {rec['expected_output']}")
        print(f"Source File     : {rec['source_file']}")
        if rec["context"]:
            ctx_preview = " || ".join(c[:180].replace("\n", " ") + "..." for c in rec["context"])
            print(f"Source Context  : {ctx_preview}")
        else:
            print(f"Source Context  : [] (Reason: {meta.get('unanswerable_reason')})")


if __name__ == "__main__":
    main()

