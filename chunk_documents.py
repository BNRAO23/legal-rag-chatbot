"""Split parsed legal Markdown documents along natural statutory units for RAG.

Natural units detected across the three Acts:
1. Front Matter / Arrangement of Sections (before Section 1 of the Act body,
   plus FEMA's Statement of Objects and Reasons).
2. Statutory Sections in the Act body:
   - Inline em-dash style (Companies Act 2013 & FEMA 1999):
     `<footnote?>[<num>[A-Z]*]. <Title>.—<body>` or `[<Title>.] Omitted...`
   - Title-above style (Income-tax Act 2025):
     `<Title>` line immediately above `<num>[A-Z]*. (<sub-section>|...)`
   Monotonic section-number validation filters out per-page amendment footnotes.
3. Schedules (and Part / Table sub-headings within Schedules) at the end of
   Companies Act 2013 and Income-tax Act 2025.

Oversized units are split only when they exceed the embedding token budget,
keeping Markdown table headers (`| ... |` + `| --- |`) and borderless
multi-column table headers (`col1 | col2 | ...`) attached to their rows.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PAGE_MARKER_RE = re.compile(r"^<!--\s*page:\s*(\d+)\s*-->$")
PAGE_LABEL_RE = re.compile(r"^\*\*\[Page\s+\d+\]\*\*$")

# Footnote lines look like: "1. Subs. by Act...", "95. Sub. for...", "1. 1st June, 2000..."
FOOTNOTE_RE = re.compile(
    r"^(?:\d+[a-z]*)\.\s+(?:"
    r"Subs?\.|Ins\.|Omitted|The\s+(?:words|figures|Explanation|proviso|brackets|letters|entries|section|clause|expression)|"
    r"w\.e\.f\.|See\s+|Vide\s+|Notification|Certain\s+words|Earlier|Prior\s+to|"
    r"\d+(?:st|nd|rd|th)\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)|"
    r"Now\s+see|Substituted|Inserted|Added|Re-numbered"
    r")",
    re.IGNORECASE,
)

CHAPTER_RE = re.compile(r"^(?:###|##)?\s*(CHAPTER\s+[IVXLC0-9A-Z-]+.*)$", re.IGNORECASE)
SCHEDULE_RE = re.compile(r"^(?:###\s*|[\['\"]+)?(SCHEDULE\s+[IVXLC]+)\b(.*)$")
SUB_SCHEDULE_RE = re.compile(r"^(?:###|##)?\s*(PART\s+[IVXLC]+|TABLE\s*[–-]\s*[A-Z])\b(.*)$")
OBJECTS_RE = re.compile(r"^(?:###\s*)?(STATEMENT OF OBJECTS AND REASONS)\b", re.IGNORECASE)

# Companies Act / FEMA inline section header patterns
INLINE_SEC_RE = re.compile(
    r"^(?:##\s*)?(?:\d+[a-z]?\[)?(\d+[A-Z]*)\.\s+(.+?)\.?—(.*)$"
)
OMITTED_SEC_RE = re.compile(
    r"^(?:##\s*)?(?:\d+[a-z]?\[)?(\d+[A-Z]*)\.\s+\[(.+?)\]\s*(Omitted\b.*)$",
    re.IGNORECASE,
)
WRAPPED_SEC_START_RE = re.compile(
    r"^(?:##\s*)?(?:\d+[a-z]?\[)?(\d+[A-Z]*)\.\s+([A-Z\"'].{3,150})$"
)

# Income-tax Act 2025 section body start (following a title line)
ITA_SEC_START_RE = re.compile(
    r"^(?:\d+[a-z]?\[)?(\d+[A-Z]*)\.\s*(?:$|(?:\[\s*)?\(\d+\)|[A-Z\"(])"
)
FOOTNOTE_SUP_RE = re.compile(r"^\d{1,3}[a-z]?$")


@dataclass
class DocMetadata:
    document_name: str
    source_file: str
    version_or_year: str
    style: str  # "em_dash_inline" or "title_above"


@dataclass
class Unit:
    unit_type: str
    unit_number: str
    unit_title: str
    chapter: str
    lines: list[tuple[int, str]] = field(default_factory=list)


def estimate_tokens(text: str) -> int:
    """Offline, deterministic token count estimate safe for 512-token embedders."""
    if not text:
        return 0
    words = len(text.split())
    return max(math.ceil(words * 1.35), math.ceil(len(text) / 3.6))


def parse_frontmatter(lines: list[str], md_path: Path) -> tuple[DocMetadata, int]:
    """Extract document name, version/year, and section formatting style."""
    source_file = md_path.with_suffix(".pdf").name
    idx = 0
    if lines and lines[0].strip() == "---":
        idx = 1
        while idx < len(lines) and lines[idx].strip() != "---":
            line = lines[idx].strip()
            if line.startswith("source_file:"):
                source_file = line.split(":", 1)[1].strip()
            idx += 1
        idx += 1

    head_blob = "\n".join(lines[idx : idx + 35])
    lower_name = source_file.lower()

    if "companies act" in lower_name:
        doc_name = "Companies Act, 2013"
        m = re.search(r"Last update[-:\s]*([0-9.\-]+)", head_blob, re.IGNORECASE)
        ver = f"2013 (as amended up to {m.group(1)})" if m else "2013 (amended up to 01.04.2021)"
        style = "em_dash_inline"
    elif "fema" in lower_name:
        doc_name = "Foreign Exchange Management Act, 1999"
        m = re.search(r"\[?\s*As on the ([^\]]+?)\]?$", head_blob, re.IGNORECASE | re.MULTILINE)
        as_on = m.group(1).strip() if m else "1st June, 2026"
        ver = f"1999 (Act 42 of 1999; As on {as_on})"
        style = "em_dash_inline"
    elif "income-tax" in lower_name or "income_tax" in lower_name:
        doc_name = "Income-tax Act, 2025"
        m = re.search(r"\[AS AMENDED BY ([^\]]+)\]", head_blob, re.IGNORECASE)
        amended = m.group(1).strip().title() if m else "Finance Act, 2026"
        ver = f"2025 (Act 30 of 2025; As amended by {amended})"
        style = "title_above"
    else:
        doc_name = md_path.stem
        ver = "unknown"
        style = "em_dash_inline"

    return DocMetadata(doc_name, source_file, ver, style), idx


def section_int(sec_num: str) -> int:
    m = re.match(r"(\d+)", sec_num)
    return int(m.group(1)) if m else 0


def next_non_marker_line(lines: list[str], start: int) -> tuple[int, str]:
    """Return index and text of the next non-empty, non-page-marker line."""
    for j in range(start, min(len(lines), start + 6)):
        s = lines[j].strip()
        if not s or PAGE_MARKER_RE.match(s) or PAGE_LABEL_RE.match(s):
            continue
        return j, s
    return -1, ""


def is_valid_ita_title(text: str) -> bool:
    """Check whether a line looks like an Income-tax Act section heading above `N.`."""
    if not text or len(text) > 180:
        return False
    if text.startswith(("(", "###", "##", "|", "[", "Note", "Provided", "Explanation", "Where", "For the")):
        return False
    if re.match(r"^(?:\d+\.|[a-z]\)|\(\w+\)|[A-Z]\.—)", text):
        return False
    if text.endswith((",", ";", "—", ":")):
        return False
    return True


def split_em_dash_document(lines: list[str], start_idx: int) -> list[Unit]:
    """Split Companies Act 2013 or FEMA 1999 into Front Matter, Sections, and Schedules."""
    units: list[Unit] = []
    current_page = 1
    current_chapter = ""
    pending_chapter_desc = False
    in_body = False
    in_schedules = False
    last_sec_num = 0

    current_unit = Unit("front_matter", "Front Matter", "Arrangement of Sections & Preliminaries", "")

    i = start_idx
    while i < len(lines):
        raw = lines[i]
        s = raw.strip()

        m_pg = PAGE_MARKER_RE.match(s)
        if m_pg:
            current_page = int(m_pg.group(1))
            i += 1
            continue
        if PAGE_LABEL_RE.match(s):
            i += 1
            continue

        # Detect start of body (Section 1: Short title...)
        if not in_body and re.match(r"^(?:##\s*)?1\.\s+Short title.+?—", s):
            in_body = True
            if current_unit.lines:
                units.append(current_unit)
            m1 = INLINE_SEC_RE.match(s)
            title = m1.group(2).strip() if m1 else "Short title, extent, commencement and application"
            last_sec_num = 1
            current_unit = Unit("section", "1", title, current_chapter, [(current_page, s.lstrip("# "))])
            i += 1
            continue

        # Track CHAPTER headings
        m_chap = CHAPTER_RE.match(s)
        if m_chap:
            current_chapter = m_chap.group(1).strip()
            pending_chapter_desc = True
            if in_body and not in_schedules:
                current_unit.lines.append((current_page, raw))
                i += 1
                continue
        elif pending_chapter_desc and s and not s.startswith(("#", "SECTIONS", "_")) and len(s) < 100:
            current_chapter = f"{current_chapter} — {s}"
            pending_chapter_desc = False
        elif s:
            pending_chapter_desc = False

        # Statement of Objects and Reasons (FEMA end)
        m_obj = OBJECTS_RE.match(s)
        if m_obj and in_body:
            if current_unit.lines:
                units.append(current_unit)
            in_schedules = True
            current_unit = Unit(
                "statement_of_objects",
                "Statement of Objects and Reasons",
                "Statement of Objects and Reasons",
                "",
                [(current_page, s)],
            )
            i += 1
            continue

        # Schedule headings (Companies Act)
        m_sched = SCHEDULE_RE.match(s)
        if m_sched and in_body and len(s) < 40:
            if current_unit.lines:
                units.append(current_unit)
            in_schedules = True
            sched_name = m_sched.group(1).title()
            current_unit = Unit("schedule", sched_name, sched_name, sched_name, [(current_page, s.lstrip("# "))])
            i += 1
            continue

        # Inside Schedules: split on TABLE –A..J or PART I..IV sub-units for cleaner granularity
        if in_schedules:
            m_sub = SUB_SCHEDULE_RE.match(s)
            if m_sub and len(s) < 60 and current_unit.unit_type == "schedule":
                base_sched = current_unit.chapter or current_unit.unit_number.split(" - ")[0]
                if len(current_unit.lines) > 3:
                    units.append(current_unit)
                    sub_label = re.sub(r"\s+", " ", m_sub.group(1).strip()).title()
                    current_unit = Unit(
                        "schedule",
                        f"{base_sched} - {sub_label}",
                        f"{base_sched}: {sub_label}",
                        base_sched,
                        [(current_page, s.lstrip("# "))],
                    )
                    i += 1
                    continue

        # Detect statutory sections inside Act body
        if in_body and not in_schedules and not FOOTNOTE_RE.match(s) and "|" not in s:
            matched_sec: tuple[str, str, str] | None = None
            consume_next = False

            m_in = INLINE_SEC_RE.match(s)
            if m_in:
                matched_sec = (m_in.group(1), m_in.group(2).strip(), s.lstrip("# "))
            else:
                m_om = OMITTED_SEC_RE.match(s)
                if m_om:
                    matched_sec = (m_om.group(1), m_om.group(2).strip().rstrip("."), s.lstrip("# "))
                else:
                    m_wr = WRAPPED_SEC_START_RE.match(s)
                    if m_wr:
                        nxt_i, nxt_s = next_non_marker_line(lines, i + 1)
                        if nxt_s and (".—" in nxt_s or "—(" in nxt_s) and len(nxt_s) < 180:
                            title_part = f"{m_wr.group(2).strip()} {nxt_s.split('—', 1)[0].rstrip('.').strip()}"
                            matched_sec = (m_wr.group(1), title_part, f"{s.lstrip('# ')} {nxt_s}")
                            consume_next = True

            if matched_sec is not None:
                sec_num, sec_title, first_line = matched_sec
                sec_i = section_int(sec_num)
                if sec_i >= last_sec_num and sec_i <= last_sec_num + 50:
                    if current_unit.lines:
                        units.append(current_unit)
                    last_sec_num = sec_i
                    current_unit = Unit("section", sec_num, sec_title.rstrip("."), current_chapter, [(current_page, first_line)])
                    i = (nxt_i + 1) if consume_next else (i + 1)
                    continue

        current_unit.lines.append((current_page, raw))
        i += 1

    if current_unit.lines:
        units.append(current_unit)
    return units


def split_ita_document(lines: list[str], start_idx: int) -> list[Unit]:
    """Split Income-tax Act 2025 into Front Matter, Sections (title-above), and Schedules."""
    units: list[Unit] = []
    current_page = 1
    current_chapter = ""
    pending_chapter_desc = False
    in_schedules = False
    last_sec_num = 0

    current_unit = Unit("front_matter", "Preamble", "Act Title & Enacting Formula", "")

    i = start_idx
    while i < len(lines):
        raw = lines[i]
        s = raw.strip()

        m_pg = PAGE_MARKER_RE.match(s)
        if m_pg:
            current_page = int(m_pg.group(1))
            i += 1
            continue
        if PAGE_LABEL_RE.match(s):
            i += 1
            continue

        # Track CHAPTER headings
        m_chap = CHAPTER_RE.match(s)
        if m_chap:
            current_chapter = m_chap.group(1).strip()
            pending_chapter_desc = True
            current_unit.lines.append((current_page, raw))
            i += 1
            continue
        elif pending_chapter_desc and s and not s.startswith(("#", "_")) and len(s) < 100 and not is_valid_ita_title(s):
            current_chapter = f"{current_chapter} — {s}"
            pending_chapter_desc = False
        elif s:
            pending_chapter_desc = False

        # Detect start of Schedules (e.g. "[SCHEDULE I", "### SCHEDULE II", ...)
        m_sched = SCHEDULE_RE.match(s)
        if m_sched and len(s) < 40 and not s.endswith(";"):
            sched_token = m_sched.group(1).upper()
            # Ensure Schedule I only triggers after the main Act sections (~500+)
            if sched_token != "SCHEDULE I" or last_sec_num >= 500:
                if current_unit.lines:
                    units.append(current_unit)
                in_schedules = True
                sched_name = sched_token.title()
                current_unit = Unit("schedule", sched_name, sched_name, sched_name, [(current_page, s.lstrip("# "))])
                i += 1
                continue

        # Detect ITA body section start (`N. ...` preceded by a title line)
        if not in_schedules and "|" not in s and not FOOTNOTE_RE.match(s):
            m_sec = ITA_SEC_START_RE.match(s)
            if m_sec:
                sec_num = m_sec.group(1)
                sec_i = section_int(sec_num)
                if sec_i > 0 and sec_i >= last_sec_num and sec_i <= last_sec_num + 20:
                    # Look backward inside current_unit.lines for the title line (skipping blank / superscript footnote digits)
                    title_idx = -1
                    for k in range(len(current_unit.lines) - 1, max(-1, len(current_unit.lines) - 5), -1):
                        cand = current_unit.lines[k][1].strip()
                        if not cand or FOOTNOTE_SUP_RE.match(cand):
                            continue
                        if is_valid_ita_title(cand):
                            title_idx = k
                        break

                    if title_idx != -1:
                        title_page, title_raw = current_unit.lines[title_idx]
                        sec_title = title_raw.strip().rstrip(".")
                        # Remove title (and trailing blanks/footnote superscripts) from previous unit
                        trailing_sup = [
                            ln for ln in current_unit.lines[title_idx + 1 :] if ln[1].strip()
                        ]
                        current_unit.lines = current_unit.lines[:title_idx]
                        if current_unit.lines:
                            units.append(current_unit)

                        new_lines: list[tuple[int, str]] = [(title_page, title_raw)]
                        new_lines.extend(trailing_sup)
                        new_lines.append((current_page, raw))
                        last_sec_num = sec_i
                        current_unit = Unit("section", sec_num, sec_title, current_chapter, new_lines)
                        i += 1
                        continue

        current_unit.lines.append((current_page, raw))
        i += 1

    if current_unit.lines:
        units.append(current_unit)
    return units


def trim_unit_lines(lines: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Strip leading/trailing blank lines from a unit while preserving page numbers."""
    start = 0
    while start < len(lines) and not lines[start][1].strip():
        start += 1
    end = len(lines)
    while end > start and not lines[end - 1][1].strip():
        end -= 1
    return lines[start:end]


def group_unit_blocks(lines: list[tuple[int, str]]) -> list[dict[str, Any]]:
    """Group a unit's lines into atomic blocks: paragraph, md_table, or borderless_table."""
    blocks: list[dict[str, Any]] = []
    i = 0
    n = len(lines)

    while i < n:
        pg, txt = lines[i]
        s = txt.strip()
        if not s:
            i += 1
            continue

        # 1. GitHub-flavored Markdown table (`| ... |` followed by `| --- | ... |`)
        if (
            s.startswith("|")
            and s.endswith("|")
            and i + 1 < n
            and re.match(r"^\|\s*---", lines[i + 1][1].strip())
        ):
            header_rows = [lines[i], lines[i + 1]]
            i += 2
            data_rows: list[tuple[int, str]] = []
            while i < n and lines[i][1].strip().startswith("|") and lines[i][1].strip().endswith("|"):
                data_rows.append(lines[i])
                i += 1
            blocks.append({"kind": "md_table", "header": header_rows, "rows": data_rows})
            continue

        # 2. Borderless multi-column table block (consecutive lines containing " | ")
        if " | " in s and not s.startswith("|"):
            tbl_lines: list[tuple[int, str]] = []
            while i < n and " | " in lines[i][1].strip() and not lines[i][1].strip().startswith("|"):
                tbl_lines.append(lines[i])
                i += 1

            # Detect leading header lines (e.g. "Sl. | ..." / "A | B | C")
            header_count = 0
            for j, (_, row_txt) in enumerate(tbl_lines[:3]):
                rs = row_txt.strip()
                if re.match(r"^[A-Z]\s*\|\s*[A-Z]", rs) or any(
                    kw in rs for kw in ("Sl.", "Serial", "Nature of", "Particulars", "Amount of", "Category")
                ):
                    header_count = j + 1
            if header_count == 0 and len(tbl_lines) >= 2:
                header_count = 1
            blocks.append(
                {
                    "kind": "borderless_table",
                    "header": tbl_lines[:header_count],
                    "rows": tbl_lines[header_count:],
                }
            )
            continue

        # 3. Standard narrative / heading / list paragraph block (split at blank lines or sub-sections)
        para_lines: list[tuple[int, str]] = []
        while i < n:
            cur_s = lines[i][1].strip()
            if not cur_s:
                break
            if cur_s.startswith("|") and i + 1 < n and re.match(r"^\|\s*---", lines[i + 1][1].strip()):
                break
            if " | " in cur_s and not cur_s.startswith("|"):
                break
            # Break paragraphs cleanly on sub-section start `(2)` / `(3)` if paragraph is already sizable
            if para_lines and re.match(r"^\(\d+\)\s+", cur_s) and sum(len(x[1]) for x in para_lines) > 250:
                break
            para_lines.append(lines[i])
            i += 1
        blocks.append({"kind": "text", "lines": para_lines})

    return blocks


def split_long_lines(lines: list[tuple[int, str]], max_tokens: int) -> list[list[tuple[int, str]]]:
    """Fallback splitter when a single narrative paragraph exceeds max_tokens."""
    chunks: list[list[tuple[int, str]]] = []
    cur: list[tuple[int, str]] = []
    for item in lines:
        candidate = cur + [item]
        if cur and estimate_tokens("\n".join(x[1] for x in candidate)) > max_tokens:
            chunks.append(cur)
            cur = [item]
        else:
            cur = candidate
    if cur:
        chunks.append(cur)

    # If a single line itself exceeds max_tokens, split on sentence boundaries.
    final_chunks: list[list[tuple[int, str]]] = []
    for ch in chunks:
        joined = "\n".join(x[1] for x in ch)
        if len(ch) == 1 and estimate_tokens(joined) > max_tokens:
            pg = ch[0][0]
            sentences = re.split(r"(?<=[.;])\s+", joined)
            buf = ""
            for sent in sentences:
                cand = f"{buf} {sent}".strip() if buf else sent
                if buf and estimate_tokens(cand) > max_tokens:
                    final_chunks.append([(pg, buf)])
                    buf = sent
                else:
                    buf = cand
            if buf:
                final_chunks.append([(pg, buf)])
        else:
            final_chunks.append(ch)
    return final_chunks


def chunk_unit(unit: Unit, doc_meta: DocMetadata, max_tokens: int) -> list[dict[str, Any]]:
    """Keep a natural unit intact unless it exceeds max_tokens; preserve table headers."""
    trimmed = trim_unit_lines(unit.lines)
    if not trimmed:
        return []

    full_text = "\n".join(x[1] for x in trimmed)
    if estimate_tokens(full_text) <= max_tokens:
        raw_parts = [trimmed]
    else:
        blocks = group_unit_blocks(trimmed)
        raw_parts = []
        cur_lines: list[tuple[int, str]] = []

        def flush_cur() -> None:
            nonlocal cur_lines
            if cur_lines:
                raw_parts.append(cur_lines)
                cur_lines = []

        for blk in blocks:
            if blk["kind"] == "text":
                blk_lines = blk["lines"]
                blk_text = "\n".join(x[1] for x in blk_lines)
                if estimate_tokens(blk_text) > max_tokens:
                    flush_cur()
                    for sub in split_long_lines(blk_lines, max_tokens):
                        raw_parts.append(sub)
                else:
                    cand = cur_lines + ([(cur_lines[-1][0], "")] if cur_lines else []) + blk_lines
                    if cur_lines and estimate_tokens("\n".join(x[1] for x in cand)) > max_tokens:
                        flush_cur()
                        cur_lines = list(blk_lines)
                    else:
                        cur_lines = cand
            else:
                # Table block (md_table or borderless_table): keep header rows with their rows!
                hdr: list[tuple[int, str]] = blk["header"]
                rows: list[tuple[int, str]] = blk["rows"]
                whole_tbl = hdr + rows
                whole_text = "\n".join(x[1] for x in whole_tbl)

                cand = cur_lines + ([(cur_lines[-1][0], "")] if cur_lines else []) + whole_tbl
                if cur_lines and estimate_tokens("\n".join(x[1] for x in cand)) <= max_tokens:
                    cur_lines = cand
                    continue

                if estimate_tokens(whole_text) <= max_tokens:
                    flush_cur()
                    cur_lines = list(whole_tbl)
                    continue

                # Oversized table: split along rows and repeat header on every continuation chunk.
                flush_cur()
                tbl_cur = list(hdr)
                for row in rows:
                    cand_tbl = tbl_cur + [row]
                    if len(tbl_cur) > len(hdr) and estimate_tokens("\n".join(x[1] for x in cand_tbl)) > max_tokens:
                        raw_parts.append(tbl_cur)
                        tbl_cur = list(hdr) + [row]
                    else:
                        tbl_cur = cand_tbl
                if tbl_cur:
                    raw_parts.append(tbl_cur)

        flush_cur()

    # Build final chunks with citation metadata
    total_parts = len(raw_parts)
    doc_slug = re.sub(r"[^a-z0-9]+", "_", doc_meta.document_name.lower()).strip("_")
    raw_unit_label = unit.unit_number if unit.unit_number.lower().startswith(unit.unit_type) else f"{unit.unit_type}_{unit.unit_number}"
    unit_slug = re.sub(r"[^a-z0-9]+", "_", raw_unit_label.lower()).strip("_")

    result_chunks: list[dict[str, Any]] = []
    for part_idx, part_lines in enumerate(raw_parts, start=1):
        clean_lines = trim_unit_lines(part_lines)
        if not clean_lines:
            continue
        text = "\n".join(x[1] for x in clean_lines)
        pages = sorted(set(x[0] for x in clean_lines))
        chunk_id = f"{doc_slug}__{unit_slug}__{part_idx:03d}"
        result_chunks.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "metadata": {
                    "document_name": doc_meta.document_name,
                    "source_file": doc_meta.source_file,
                    "version_or_year": doc_meta.version_or_year,
                    "chapter": unit.chapter or None,
                    "unit_type": unit.unit_type,
                    "unit_number": unit.unit_number,
                    "unit_title": unit.unit_title,
                    "page": pages[0],
                    "page_end": pages[-1],
                    "pages": pages,
                    "part": part_idx,
                    "total_parts": total_parts,
                    "token_estimate": estimate_tokens(text),
                    "char_count": len(text),
                },
            }
        )
    return result_chunks


def process_markdown_file(md_path: Path, max_tokens: int) -> tuple[DocMetadata, list[Unit], list[dict[str, Any]]]:
    """Detect natural units in one parsed Markdown file and chunk them."""
    lines = md_path.read_text(encoding="utf-8").splitlines()
    doc_meta, body_start = parse_frontmatter(lines, md_path)

    if doc_meta.style == "title_above":
        units = split_ita_document(lines, body_start)
    else:
        units = split_em_dash_document(lines, body_start)

    all_chunks: list[dict[str, Any]] = []
    seen_ids: dict[str, int] = {}
    for unit in units:
        for ch in chunk_unit(unit, doc_meta, max_tokens):
            base_id = ch["chunk_id"]
            if base_id in seen_ids:
                seen_ids[base_id] += 1
                stem, part_suffix = base_id.rsplit("__", 1)
                ch["chunk_id"] = f"{stem}_div{seen_ids[base_id]}__{part_suffix}"
            else:
                seen_ids[base_id] = 1
            all_chunks.append(ch)
    return doc_meta, units, all_chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk parsed legal Markdown files along natural statutory units with citation metadata."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("parsed_docs"),
        help="Directory containing parsed .md files (default: ./parsed_docs).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("parsed_docs/chunks.jsonl"),
        help="Output JSONL file containing all chunks (default: parsed_docs/chunks.jsonl).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=450,
        help="Maximum estimated tokens per chunk before sub-splitting a unit (default: 450).",
    )
    args = parser.parse_args()

    md_files = sorted(args.input_dir.glob("*.md"))
    if not md_files:
        print(f"No .md files found in {args.input_dir}")
        return

    all_chunks: list[dict[str, Any]] = []
    print(f"Chunking {len(md_files)} document(s) with max_tokens={args.max_tokens}...\n")

    for md_path in md_files:
        doc_meta, units, doc_chunks = process_markdown_file(md_path, args.max_tokens)
        sec_count = sum(1 for u in units if u.unit_type == "section")
        sched_count = sum(1 for u in units if u.unit_type == "schedule")
        split_units = sum(1 for c in doc_chunks if c["metadata"]["part"] == 1 and c["metadata"]["total_parts"] > 1)
        print(
            f"- {doc_meta.document_name} [{doc_meta.version_or_year}]: "
            f"{len(units)} units ({sec_count} sections, {sched_count} schedules/parts), "
            f"{split_units} oversized units split -> {len(doc_chunks)} chunks"
        )
        all_chunks.extend(doc_chunks)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for ch in all_chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")

    print(f"\nTotal chunks saved to {args.output}: {len(all_chunks)}")

    # Pick 3 representative sample chunks across documents (including a section, a split table chunk, etc.)
    samples: list[dict[str, Any]] = []
    for doc_prefix in ("companies_act_2013", "foreign_exchange_management_act_1999", "income_tax_act_2025"):
        match = next(
            (
                c
                for c in all_chunks
                if c["chunk_id"].startswith(doc_prefix)
                and (
                    (doc_prefix == "companies_act_2013" and "| " in c["text"] and c["metadata"]["total_parts"] > 1)
                    or (doc_prefix == "foreign_exchange_management_act_1999" and c["metadata"]["unit_type"] == "section" and c["metadata"]["unit_number"] == "6")
                    or (doc_prefix == "income_tax_act_2025" and c["metadata"]["unit_type"] == "section" and c["metadata"]["unit_number"] == "18")
                )
            ),
            None,
        )
        if match is None:
            match = next((c for c in all_chunks if c["chunk_id"].startswith(doc_prefix)), None)
        if match is not None:
            samples.append(match)

    print("\n=== THREE SAMPLE CHUNKS ===")
    for idx, ch in enumerate(samples[:3], start=1):
        print(f"\n--- Sample Chunk {idx} ({ch['chunk_id']}) ---")
        print("Metadata:", json.dumps(ch["metadata"], ensure_ascii=False, indent=2))
        preview = ch["text"][:700] + ("..." if len(ch["text"]) > 700 else "")
        print("Text:\n" + preview)


if __name__ == "__main__":
    main()
