"""Convert project legal PDF source documents into clean page-numbered Markdown.

Chosen open-source parser: PyMuPDF (`pymupdf`)
- Uses the C-based MuPDF engine for fast extraction even on large statutory PDFs
  (e.g., the 118 MB / 674-page Income-tax Act 2025).
- Exposes exact span/line coordinates, font sizes/flags, vector drawings,
  table geometry, and embedded image bounding boxes without dropping text behind
  full-page white background rectangles or clipping drop-cap headers.
- Keeps 1-based PDF page markers (`<!-- page: N -->`) in each output Markdown
  file so downstream RAG chunkers can attach exact page citations.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf


HEADER_FOOTER_PATTERNS = (
    re.compile(r"^Income Tax Department$", re.IGNORECASE),
    re.compile(r"^Ministry of Finance,\s*Government of India$", re.IGNORECASE),
    re.compile(r"^Downloaded/Printed on .+ from www\.incometaxindia\.gov\.in", re.IGNORECASE),
    re.compile(r"^Page \d+ of \d+$", re.IGNORECASE),
)

WATERMARK_PATTERNS = (
    re.compile(r"^IndiaCode$", re.IGNORECASE),
)

HEADING_PREFIX_RE = re.compile(
    r"^(CHAPTER\s+[IVXLC0-9A-Z]+|PART\s+[IVXLC0-9A-Z]+|"
    r"(?:THE\s+)?(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|"
    r"TENTH|ELEVENTH|TWELFTH|THIRTEENTH|FOURTEENTH|FIFTEENTH|SIXTEENTH|"
    r"SEVENTEENTH|EIGHTEENTH|NINETEENTH|TWENTIETH|\d+(?:ST|ND|RD|TH))\s+SCHEDULE|"
    r"SCHEDULE\s+[IVXLC0-9A-Z]+|ARRANGEMENT OF SECTIONS|"
    r"STATEMENT OF OBJECTS AND REASONS|LIST OF AMENDING ACTS)$",
    re.IGNORECASE,
)


@dataclass
class PageFlag:
    page: int
    reasons: list[str]
    details: str
    char_count: int


@dataclass
class DocumentSummary:
    source_file: str
    output_file: str
    total_pages: int
    total_chars: int
    total_words: int
    elapsed_seconds: float
    flagged_pages: list[PageFlag] = field(default_factory=list)


def normalize_text(text: str) -> str:
    """Normalize PDF whitespace artifacts without altering legal punctuation."""
    text = text.replace("\xa0", " ").replace("\xad", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def is_boilerplate_line(text: str, y0: float, y1: float, page_height: float, page_num: int) -> bool:
    """Return True if a line is a repeating header, footer, watermark, or margin page number."""
    cleaned = text.strip()
    if not cleaned:
        return True

    if any(pat.match(cleaned) for pat in WATERMARK_PATTERNS):
        return True

    in_top_margin = y0 < 58.0
    in_bottom_margin = y1 > page_height - 62.0

    if in_top_margin or in_bottom_margin:
        if cleaned == str(page_num) or re.fullmatch(r"\d{1,4}", cleaned):
            return True
        if any(pat.search(cleaned) for pat in HEADER_FOOTER_PATTERNS):
            return True

    # Bottom print stamp may contain both URL timestamp and "Page X of Y" on one line.
    if y0 > page_height - 48.0 and "incometaxindia.gov.in" in cleaned.lower():
        return True

    return False


def extract_clean_lines(page: pymupdf.Page, page_num: int) -> tuple[list[dict[str, Any]], float]:
    """Extract non-boilerplate text lines with coordinates and font metadata."""
    text_dict = page.get_text("dict")
    page_height = page.rect.height
    raw_lines: list[dict[str, Any]] = []
    font_sizes: list[float] = []

    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            raw_text = "".join(s.get("text", "") for s in spans)
            cleaned = normalize_text(raw_text)
            if not cleaned:
                continue

            x0, y0, x1, y1 = line["bbox"]
            if is_boilerplate_line(cleaned, y0, y1, page_height, page_num):
                continue

            weighted_sizes = [
                (s.get("size", 10.0), len(s.get("text", "").strip()))
                for s in spans
                if s.get("text", "").strip()
            ]
            if weighted_sizes:
                max_span = max(weighted_sizes, key=lambda pair: pair[1])
                font_size = max_span[0]
            else:
                font_size = 10.0

            is_bold = any(bool(s.get("flags", 0) & 16) for s in spans if s.get("text", "").strip())
            font_sizes.append(font_size)
            raw_lines.append(
                {
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "text": cleaned,
                    "size": font_size,
                    "bold": is_bold,
                }
            )

    median_size = statistics.median(font_sizes) if font_sizes else 10.0
    return raw_lines, median_size


def count_multicolumn_rows(lines: list[dict[str, Any]]) -> tuple[int, list[list[dict[str, Any]]]]:
    """Group lines into horizontal y-bands and count rows with >= 2 x-separated columns."""
    if not lines:
        return 0, []

    ordered = sorted(lines, key=lambda ln: (round(ln["y0"] / 4.0), ln["x0"]))
    y_rows: list[list[dict[str, Any]]] = []
    for ln in ordered:
        if not y_rows or abs(ln["y0"] - y_rows[-1][0]["y0"]) > 4.0:
            y_rows.append([ln])
        else:
            y_rows[-1].append(ln)

    multi_col_count = 0
    for row in y_rows:
        row.sort(key=lambda ln: ln["x0"])
        col_groups = [row[0]]
        for ln in row[1:]:
            if ln["x0"] - col_groups[-1]["x1"] > 12.0:
                col_groups.append(ln)
        if len(col_groups) >= 2:
            multi_col_count += 1

    return multi_col_count, y_rows


def extract_grid_tables(page: pymupdf.Page) -> list[dict[str, Any]]:
    """Run PyMuPDF line-based table extraction and clip-read cells cleanly."""
    found = page.find_tables(strategy="lines")
    rendered_tables: list[dict[str, Any]] = []

    for tbl in found.tables:
        matrix: list[list[str]] = []
        none_cells = 0
        total_cells = 0
        empty_cells = 0
        multiline_cells = 0

        for row in tbl.rows:
            row_cells: list[str] = []
            for box in row.cells:
                total_cells += 1
                if box is None:
                    none_cells += 1
                    row_cells.append("")
                    continue
                cell_txt = page.get_text("text", clip=pymupdf.Rect(box))
                cell_txt = normalize_text(cell_txt)
                if "\n" in cell_txt:
                    multiline_cells += 1
                cell_txt = re.sub(r"\s*\n\s*", " ", cell_txt).replace("|", "\\|")
                if not cell_txt:
                    empty_cells += 1
                row_cells.append(cell_txt)
            if any(row_cells):
                matrix.append(row_cells)

        if not matrix:
            continue

        col_count = max(len(r) for r in matrix)
        padded = [r + [""] * (col_count - len(r)) for r in matrix]

        # Build GitHub-flavored Markdown table.
        header = padded[0]
        sep = ["---"] * col_count
        md_lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(sep) + " |",
        ]
        for r in padded[1:]:
            md_lines.append("| " + " | ".join(r) + " |")

        empty_ratio = (empty_cells / total_cells) if total_cells else 0.0
        is_complex = (
            none_cells > 0
            or empty_ratio > 0.35
            or multiline_cells >= 2
            or col_count >= 4
        )
        rendered_tables.append(
            {
                "bbox": tbl.bbox,
                "markdown": "\n".join(md_lines),
                "rows": len(padded),
                "cols": col_count,
                "complex": is_complex,
                "empty_ratio": round(empty_ratio, 2),
                "merged_cells": none_cells,
            }
        )

    return rendered_tables


def line_inside_any_bbox(ln: dict[str, Any], bboxes: list[tuple[float, float, float, float]]) -> bool:
    """Check whether a line's center falls inside any extracted table bounding box."""
    cx = (ln["x0"] + ln["x1"]) / 2.0
    cy = (ln["y0"] + ln["y1"]) / 2.0
    return any(x0 - 2 <= cx <= x1 + 2 and y0 - 2 <= cy <= y1 + 2 for x0, y0, x1, y1 in bboxes)


def format_line_markdown(ln: dict[str, Any], median_size: float) -> str:
    """Format a non-table line as a Markdown heading when appropriate."""
    txt = ln["text"]
    if HEADING_PREFIX_RE.match(txt):
        return f"### {txt}"
    if ln["size"] >= median_size + 2.5 and len(txt) <= 120:
        return f"## {txt}"
    return txt


def render_page_markdown(
    lines: list[dict[str, Any]],
    y_rows: list[list[dict[str, Any]]],
    grid_tables: list[dict[str, Any]],
    median_size: float,
    multi_col_rows: int,
) -> str:
    """Combine narrative text and extracted tables in top-to-bottom reading order."""
    table_bboxes = [t["bbox"] for t in grid_tables]
    elements: list[tuple[float, str]] = []

    for tbl in grid_tables:
        elements.append((tbl["bbox"][1], "\n" + tbl["markdown"] + "\n"))

    # If borderless multi-column rows exist without grid tables, keep row cells aligned with " | ".
    is_borderless_tabular = multi_col_rows >= 3 and not grid_tables

    if is_borderless_tabular:
        for row in y_rows:
            row_filtered = [ln for ln in row if not line_inside_any_bbox(ln, table_bboxes)]
            if not row_filtered:
                continue
            row_filtered.sort(key=lambda ln: ln["x0"])
            merged_cells: list[str] = [row_filtered[0]["text"]]
            prev_x1 = row_filtered[0]["x1"]
            for ln in row_filtered[1:]:
                if ln["x0"] - prev_x1 > 12.0:
                    merged_cells.append(ln["text"])
                else:
                    merged_cells[-1] = f"{merged_cells[-1]} {ln['text']}"
                prev_x1 = max(prev_x1, ln["x1"])
            if len(merged_cells) >= 2:
                elements.append((row_filtered[0]["y0"], " | ".join(merged_cells)))
            else:
                elements.append((row_filtered[0]["y0"], format_line_markdown(row_filtered[0], median_size)))
    else:
        for ln in lines:
            if line_inside_any_bbox(ln, table_bboxes):
                continue
            elements.append((ln["y0"], format_line_markdown(ln, median_size)))

    elements.sort(key=lambda item: item[0])

    # Merge consecutive narrative lines into clean paragraphs while keeping headings/tables distinct.
    output_lines: list[str] = []
    for _, text in elements:
        if text.startswith("## ") or text.startswith("### ") or text.startswith("\n| "):
            if output_lines and output_lines[-1] != "":
                output_lines.append("")
            output_lines.append(text.strip())
            output_lines.append("")
        else:
            output_lines.append(text)

    rendered = "\n".join(output_lines)
    rendered = re.sub(r"\n{3,}", "\n\n", rendered).strip()
    return rendered


def inspect_page_flags(
    page: pymupdf.Page,
    page_num: int,
    body_text: str,
    multi_col_rows: int,
    grid_tables: list[dict[str, Any]],
) -> PageFlag | None:
    """Flag pages likely to have degraded extraction (tables, scans, formula images, etc.)."""
    reasons: list[str] = []
    details_parts: list[str] = []
    char_count = len(body_text)

    # 1. Image / scanned / embedded-formula detection
    page_area = max(page.rect.width * page.rect.height, 1.0)
    im_info = page.get_image_info(xrefs=False)
    nontrivial_imgs = []
    img_area = 0.0
    for im in im_info:
        x0, y0, x1, y1 = im["bbox"]
        w_px, h_px = im.get("width", 0), im.get("height", 0)
        area = abs((x1 - x0) * (y1 - y0))
        img_area += area
        if w_px >= 20 and h_px >= 15:
            nontrivial_imgs.append((w_px, h_px, area))

    area_ratio = img_area / page_area
    if nontrivial_imgs:
        if area_ratio >= 0.25 and char_count < 150:
            reasons.append("SCANNED_OR_IMAGE_ONLY")
            details_parts.append(
                f"{len(nontrivial_imgs)} large image(s) covering {area_ratio:.0%} of page with only {char_count} chars"
            )
        elif not (page_num == 1 and char_count < 300):
            reasons.append("EMBEDDED_IMAGE_OR_FORMULA")
            dims = ", ".join(f"{w}x{h}px" for w, h, _ in nontrivial_imgs[:4])
            details_parts.append(
                f"{len(nontrivial_imgs)} embedded image(s) ({dims}) - may contain formula/diagram omitted from text"
            )

    # 2. Table / multi-column detection
    if grid_tables:
        complex_tbls = [t for t in grid_tables if t["complex"]]
        if complex_tbls:
            reasons.append("COMPLEX_GRID_TABLE")
            summary = "; ".join(
                f"{t['rows']}x{t['cols']} (empty={t['empty_ratio']:.0%}, merged={t['merged_cells']})"
                for t in complex_tbls[:3]
            )
            details_parts.append(f"{len(complex_tbls)}/{len(grid_tables)} grid table(s) complex: {summary}")
        else:
            reasons.append("GRID_TABLE")
            summary = "; ".join(f"{t['rows']}x{t['cols']}" for t in grid_tables[:3])
            details_parts.append(f"{len(grid_tables)} grid table(s) extracted ({summary})")
    elif multi_col_rows >= 3:
        reasons.append("BORDERLESS_TABLE_OR_MULTICOL")
        details_parts.append(
            f"{multi_col_rows} multi-column y-aligned rows without grid lines (borderless table/schedule)"
        )

    # 3. Low text / empty page detection
    if char_count < 100 and "SCANNED_OR_IMAGE_ONLY" not in reasons:
        reasons.append("LOW_TEXT")
        details_parts.append(f"only {char_count} chars of body text")

    # 4. Garbled encoding detection
    if "\ufffd" in body_text or "(cid:" in body_text:
        reasons.append("GARBLED_ENCODING")
        details_parts.append("contains replacement character or unresolved (cid:) glyph")

    if not reasons:
        return None

    return PageFlag(
        page=page_num,
        reasons=reasons,
        details="; ".join(details_parts),
        char_count=char_count,
    )


def process_pdf(pdf_path: Path, output_dir: Path) -> DocumentSummary:
    """Parse one PDF into clean page-numbered Markdown and collect quality flags."""
    t0 = time.perf_counter()
    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)

    md_sections: list[str] = [
        "---",
        f"source_file: {pdf_path.name}",
        f"total_pages: {total_pages}",
        "parser: pymupdf",
        "---",
        "",
    ]

    total_chars = 0
    total_words = 0
    flagged_pages: list[PageFlag] = []

    for idx in range(total_pages):
        page_num = idx + 1
        page = doc[idx]

        lines, median_size = extract_clean_lines(page, page_num)
        multi_col_rows, y_rows = count_multicolumn_rows(lines)

        # Only run expensive line-table detection on pages with multi-column alignment.
        grid_tables = extract_grid_tables(page) if multi_col_rows >= 2 else []

        body_md = render_page_markdown(lines, y_rows, grid_tables, median_size, multi_col_rows)
        total_chars += len(body_md)
        total_words += len(body_md.split())

        md_sections.append(f"<!-- page: {page_num} -->")
        md_sections.append(f"**[Page {page_num}]**")
        md_sections.append("")
        md_sections.append(body_md if body_md else "*[No extractable body text on this page]*")
        md_sections.append("")

        flag = inspect_page_flags(page, page_num, body_md, multi_col_rows, grid_tables)
        if flag is not None:
            flagged_pages.append(flag)

    doc.close()

    output_path = output_dir / f"{pdf_path.stem}.md"
    output_path.write_text("\n".join(md_sections), encoding="utf-8")

    return DocumentSummary(
        source_file=pdf_path.name,
        output_file=str(output_path),
        total_pages=total_pages,
        total_chars=total_chars,
        total_words=total_words,
        elapsed_seconds=round(time.perf_counter() - t0, 2),
        flagged_pages=flagged_pages,
    )


def format_page_list(pages: list[int], limit: int = 25) -> str:
    """Compact page numbers into ranges for readable console summaries."""
    if not pages:
        return "none"
    pages = sorted(set(pages))
    ranges: list[str] = []
    start = prev = pages[0]
    for p in pages[1:]:
        if p == prev + 1:
            prev = p
        else:
            ranges.append(str(start) if start == prev else f"{start}-{prev}")
            start = prev = p
    ranges.append(str(start) if start == prev else f"{start}-{prev}")
    if len(ranges) > limit:
        return ", ".join(ranges[:limit]) + f", ... (+{len(ranges) - limit} more ranges)"
    return ", ".join(ranges)


def print_summary(summaries: list[DocumentSummary], skipped_files: list[Path]) -> None:
    """Print per-file conversion summary and flagged pages."""
    print("\n=== DOCUMENT PARSING SUMMARY ===")
    for s in summaries:
        reason_counts: Counter[str] = Counter()
        pages_by_reason: dict[str, list[int]] = {}
        for f in s.flagged_pages:
            for r in f.reasons:
                reason_counts[r] += 1
                pages_by_reason.setdefault(r, []).append(f.page)

        print(f"\nFile: {s.source_file}")
        print(f"  Output      : {s.output_file}")
        print(f"  Pages       : {s.total_pages}")
        print(f"  Extracted   : {s.total_words:,} words / {s.total_chars:,} chars ({s.elapsed_seconds}s)")
        print(f"  Flagged pgs : {len(s.flagged_pages)} / {s.total_pages}")
        if reason_counts:
            for reason, count in sorted(reason_counts.items()):
                pg_str = format_page_list(pages_by_reason[reason])
                print(f"    - {reason} ({count}): pages {pg_str}")
        else:
            print("    - No problematic pages flagged.")

    if skipped_files:
        print("\nSkipped non-PDF project files (eval/reference artifacts, not converted as legal source PDFs):")
        for p in skipped_files:
            print(f"  - {p.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert project PDF source documents into clean page-numbered Markdown for RAG."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("."),
        help="Directory containing source PDF documents (default: current directory).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("parsed_docs"),
        help="Directory to write one .md file per source document (default: ./parsed_docs).",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_files = sorted(p for p in input_dir.iterdir() if p.is_file())
    pdf_files = [p for p in all_files if p.suffix.lower() == ".pdf"]
    skipped_files = [
        p for p in all_files
        if p.suffix.lower() != ".pdf" and p.suffix.lower() in {".docx", ".txt", ".doc"}
    ]

    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return

    summaries: list[DocumentSummary] = []
    for pdf_path in pdf_files:
        print(f"Parsing {pdf_path.name} ...")
        summaries.append(process_pdf(pdf_path, output_dir))

    report_path = output_dir / "parse_report.json"
    serializable = [
        {
            "source_file": s.source_file,
            "output_file": s.output_file,
            "total_pages": s.total_pages,
            "total_chars": s.total_chars,
            "total_words": s.total_words,
            "elapsed_seconds": s.elapsed_seconds,
            "flagged_pages": [
                {
                    "page": f.page,
                    "reasons": f.reasons,
                    "details": f.details,
                    "char_count": f.char_count,
                }
                for f in s.flagged_pages
            ],
        }
        for s in summaries
    ]
    report_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    print_summary(summaries, skipped_files)
    print(f"\nDetailed page-level report written to: {report_path}")


if __name__ == "__main__":
    main()

