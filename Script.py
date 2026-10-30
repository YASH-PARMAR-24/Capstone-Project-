#!/usr/bin/env python3
"""Extract ANCP telecom modelling inputs from one merged TRAI PDF.

This replaces the folder-of-PDFs extraction step. It writes:
  extracted.csv          one row per in-scope report month x 22 telecom circles
  report_inventory.csv   detected report ranges and extraction quality checks
  missing_months.csv     expected months not represented by a detected report
  extraction_summary.txt human-readable summary

The parser deliberately preserves failures as missing values and logs them; it
never guesses subscriber counts. PyMuPDF is used for searchable page text/word
positions; pdfplumber table extraction is used for MNP tables.

Install: pip install pymupdf pdfplumber pandas
Run from the folder containing the merged PDF and supplementary PDFs:
        python extract_ancp_merged_pdf.py "3rd_merged.pdf" --outdir ancp_extraction
Supplementary PDFs beside the merged PDF are auto-detected and combined by report month.
"""
from __future__ import annotations

import argparse
import calendar
import itertools
import re
import sys
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pandas as pd
import pdfplumber

CANON = [
    "Andhra Pradesh", "Assam", "Bihar", "Delhi", "Gujarat", "Haryana",
    "Himachal Pradesh", "J&K", "Karnataka", "Kerala", "Kolkata",
    "Madhya Pradesh", "Maharashtra", "Mumbai", "North East", "Odisha",
    "Punjab", "Rajasthan", "Tamil Nadu", "UP East", "UP West", "West Bengal",
]
METRO_PSU = {"Delhi", "Mumbai"}  # MTNL is in the wireless table for these two circles
MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTH_ABBR = {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}
MONTH_LOOKUP = {**MONTHS, **MONTH_ABBR, "sept": 9}
KEY = {re.sub(r"[^a-z]", "", c.lower()): c for c in CANON}
KEY.update({
    "jammukashmir": "J&K", "jammuandkashmir": "J&K", "jk": "J&K",
    "upe": "UP East", "upeast": "UP East", "upw": "UP West", "upwest": "UP West",
    "orissa": "Odisha", "uttarpradesheast": "UP East", "uttarpradeshwest": "UP West",
})

INT_RE = re.compile(r"^(?:\(?-?[\d,]+\)?|-)$")
DEC_RE = re.compile(r"^-?[\d,]+(?:\.\d+)?$")
TITLE_RE = re.compile(r"Highlights of Telecom Subscription Data", re.I)
PERIOD_RE = re.compile(
    r"Highlights of Telecom Subscription Data\s+(?:as\s+on|at\s+the\s+end\s+of)\s+([^\n\r]+)", re.I
)
MONTH_YEAR_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s*[, -]*\s*(20\d{2})\b",
    re.I,
)


def print_progress(current: int, total: int, label: str, width: int = 34) -> None:
    """Print a dependency-free terminal progress bar (works in VS Code on Windows)."""
    total = max(int(total), 1)
    current = min(max(int(current), 0), total)
    fraction = current / total
    filled = int(width * fraction)
    bar = "#" * filled + "-" * (width - filled)
    percent = fraction * 100
    print(f"\r{label:<23} [{bar}] {percent:6.2f}% ({current}/{total})", end="", flush=True)
    if current >= total:
        print(flush=True)


def compact(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def key_text(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def lookup_circle(name: str) -> str | None:
    k = key_text(name)
    if k in KEY:
        return KEY[k]
    # Match text with occasional leading OCR debris, but do not fuzzy-match arbitrary names.
    for kk, value in KEY.items():
        if len(kk) >= 5 and k.endswith(kk):
            return value
    return None


def parse_period_from_title(text: str) -> tuple[str | None, str]:
    m = PERIOD_RE.search(text)
    raw = compact(m.group(1)) if m else ""
    # The title may use "as on 30th April, 2024" or "at the end of July 2026".
    mm = MONTH_YEAR_RE.search(raw)
    if not mm:
        # Slightly wider fallback, limited to the 200 chars after the title.
        pos = TITLE_RE.search(text)
        snippet = text[pos.start():pos.start() + 250] if pos else text[:250]
        mm = MONTH_YEAR_RE.search(snippet)
    if not mm:
        return None, raw
    month_no = MONTHS[mm.group(1).lower()]
    return f"{int(mm.group(2)):04d}-{month_no:02d}", raw


def filename_month_override(path: Path) -> str | None:
    """Known single-report supplements whose PDF text layer omits the title."""
    normalized = re.sub(r"[^a-z0-9]", "", path.stem.lower())
    known = {
        "prno12of2022": "2021-12",  # TRAI PR 12/2022: data as on 31 Dec 2021
        "prno19of2026": "2025-12",  # TRAI PR 19/2026: data as on 31 Dec 2025
        "prno46of2026": "2026-02",  # TRAI PR 46/2026: data at the end of Feb 2026
        "prno125of2026": "2026-08", # TRAI PR 125/2026: data as on 31 Aug 2026
    }
    for prefix, month in known.items():
        # Downloaded filenames may append suffixes such as "_0" or a copy number.
        if normalized.startswith(prefix):
            return month
    # Fallback for the Jan/Feb 2025 files if their embedded heading is unreadable.
    if "tsdjan25english" in normalized:
        return "2025-01"
    if "tsdfeb25english" in normalized:
        return "2025-02"
    return None


def parse_release_date(text: str) -> str | None:
    # Release date is an audit field only. Some releases contain OCR/printing noise;
    # in those cases return None rather than inventing a date.
    head = text[:1800]
    m = re.search(
        r"New Delhi,?\s*(\d{1,2})(?:st|nd|rd|th)?\s+"
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s*,?\s*(20\d{2})",
        head, re.I,
    )
    if not m:
        return None
    try:
        dt = pd.Timestamp(year=int(m.group(3)), month=MONTHS[m.group(2).lower()], day=int(m.group(1)))
        return dt.date().isoformat()
    except Exception:
        return None


def clean_token(token: str) -> str:
    t = str(token).strip()
    # Strip stray delimiters that sometimes appear around OCRed numeric table cells.
    t = re.sub(r"^[\[\]{}()|]+|[\[\]{}()|]+$", "", t)
    return t.strip()


def is_int_token(token: str) -> bool:
    return bool(INT_RE.fullmatch(clean_token(token)))


def to_int(token: str) -> int:
    t = clean_token(token)
    if t in ("", "-"):
        return 0
    neg = t.startswith("(") or t.startswith("-")
    v = int(t.strip("()-").replace(",", ""))
    return -v if neg else v


def page_lines(page: fitz.Page) -> list[list[tuple]]:
    """Group PyMuPDF words into visual lines, retaining word coordinates."""
    words = sorted(page.get_text("words"), key=lambda w: (w[1], w[0]))
    groups: list[list[tuple]] = []
    for w in words:
        if groups and abs(float(w[1]) - float(groups[-1][0][1])) < 2.7:
            groups[-1].append(w)
        else:
            groups.append([w])
    return [sorted(g, key=lambda w: w[0]) for g in groups]


def line_tokens(words: list[tuple]) -> list[str]:
    toks: list[str] = []
    last_x1 = None
    for w in words:
        raw = clean_token(w[4])
        if not raw:
            continue
        # Glue numerical fragments if the PDF positioned them with no visible gap.
        if toks and last_x1 is not None and float(w[0]) - last_x1 < 0.8:
            prev = toks[-1]
            if re.fullmatch(r"[\d,.-]+", prev) and re.fullmatch(r"[\d,.-]+", raw):
                toks[-1] = prev + raw
                last_x1 = float(w[2])
                continue
        toks.append(raw)
        last_x1 = float(w[2])
    return toks


def assign_operators(circle: str, pairs: list[tuple[int, int]], total_prev: int, total_curr: int):
    """Map observed pair sequence to operator columns using reported total reconciliation."""
    metro = circle in METRO_PSU
    exact, implied = [], []
    for r, b, v, m in itertools.product((0, 1), repeat=4):
        ops = ["airtel"] + (["rcom"] if r else []) + ["vi"] + (["bsnl"] if b else []) \
              + (["vno"] if v else []) + (["mtnl"] if m else []) + ["jio"]
        if len(ops) != len(pairs):
            continue
        if ("mtnl" in ops) != metro or ("bsnl" in ops) == metro:
            continue
        if "rcom" in ops and max(pairs[ops.index("rcom")]) > 200_000:
            continue
        if min(pairs[ops.index("vi")]) < 100_000:
            continue
        d_prev = total_prev - sum(p[0] for p in pairs)
        d_curr = total_curr - sum(p[1] for p in pairs)
        if d_prev == 0 and d_curr == 0:
            exact.append(ops)
        elif "rcom" not in ops and 0 < d_prev <= 5000 and 0 <= d_curr <= 5000:
            implied.append((ops, (d_prev, d_curr)))
    if exact:
        return exact[0], len(exact) > 1, None
    if implied:
        return implied[0][0], len(implied) > 1, implied[0][1]
    return None, False, None


def parse_subscriber_line(words: list[tuple]) -> dict | None:
    toks = line_tokens(words)
    if len(toks) < 10:
        return None
    start_i = None
    circle = None
    # Try each numeric token as start of data, allowing names like U.P.(E) or Jammu & Kashmir.
    for i, tok in enumerate(toks):
        if is_int_token(tok):
            possible = lookup_circle(" ".join(toks[:i]))
            if possible:
                start_i, circle = i, possible
                break
    if start_i is None or circle is None:
        return None
    num_toks = toks[start_i:]
    if not all(is_int_token(t) for t in num_toks):
        return None
    if len(num_toks) % 2 == 0 or len(num_toks) < 9:
        return None
    nums = [to_int(t) for t in num_toks]
    net, (tot_prev, tot_curr), body = nums[-1], nums[-3:-1], nums[:-3]
    pairs = [(body[k], body[k + 1]) for k in range(0, len(body), 2)]
    ops, ambiguous, implied = assign_operators(circle, pairs, tot_prev, tot_curr)
    row = {
        "circle": circle, "total_prev": tot_prev, "total_curr": tot_curr,
        "total_net_add": net, "row_check_ok": bool(ops is not None and net == tot_curr - tot_prev),
        "column_ambiguous": bool(ambiguous), "assigned_operators": "|".join(ops) if ops else "",
    }
    for name in ("airtel", "rcom", "vi", "bsnl", "vno", "mtnl", "jio"):
        if ops is None:
            pair = (float("nan"), float("nan"))
        else:
            pair = pairs[ops.index(name)] if name in ops else (0, 0)
        if name == "rcom" and implied is not None:
            pair = implied
        row[f"{name}_prev"], row[f"{name}_curr"] = pair
    return row


def extract_wireless_table(doc: fitz.Document, texts: list[str], first_page: int, last_page: int, report_month: str):
    best_rows: list[dict] = []
    best_page = None
    best_score = (-1, -1, -1)
    all_page_scores = []
    for pidx in range(first_page, last_page + 1):  # zero-based inclusive
        # Fast pre-filter: only pages with many circle labels and many number-like tokens
        # can plausibly be the per-circle subscriber table. This avoids laying out every page.
        txt = texts[pidx]
        if len(re.findall(r"(?i)\b(?:Andhra Pradesh|Assam|Bihar|Delhi|Gujarat|Haryana|Himachal Pradesh|Karnataka|Kerala|Kolkata|Madhya Pradesh|Maharashtra|Mumbai|North East|Odisha|Punjab|Rajasthan|Tamil Nadu|West Bengal|J\s*&\s*K|Jammu & Kashmir|U\.?P\.?\s*\(?[EW]\)?)\b", txt)) < 10:
            continue
        if len(re.findall(r"(?<![A-Za-z])\d[\d,]*(?:\.\d+)?", txt)) < 80:
            continue
        page = doc[pidx]
        parsed = []
        for line in page_lines(page):
            row = parse_subscriber_line(line)
            if row:
                parsed.append(row)
        good = [r for r in parsed if r["row_check_ok"]]
        # Prefer more verified rows, then more distinct circles, then more parsed rows.
        score = (len(good), len(set(r["circle"] for r in good)), len(parsed))
        if len(parsed) >= 3:
            all_page_scores.append((pidx + 1, score, len(parsed)))
        if score > best_score:
            best_score, best_rows, best_page = score, parsed, pidx + 1
    # A wireless page should be reconcilable for most of the 22 circles. Prevent accidentally
    # selecting a different table with coincidental circle names.
    best_good = [r for r in best_rows if r["row_check_ok"]]
    if len(best_good) < 10:
        return [], None, best_score, all_page_scores
    # Prefer valid duplicate rows, de-duplicate by circle, preserve failures for audit if needed.
    dedup: dict[str, dict] = {}
    for r in sorted(best_rows, key=lambda r: (not r["row_check_ok"], r["column_ambiguous"])):
        dedup.setdefault(r["circle"], r)
    return list(dedup.values()), best_page, best_score, all_page_scores


def parse_mnp_table(table: list[list[Any]], report_month: str):
    # Keep source semantics explicit. Some older tables label the values only "in Million",
    # but the per-circle cumulative totals are clearly >5 million; those must be differenced.
    joined = " ".join(compact(c) for row in table for c in row if c)
    low = joined.lower()
    monthly_explicit = bool(re.search(r"requests?\s+(?:received\s+)?in\s+the\s+month|number\s+of\s+porting\s+requests\s+in\s+the\s+month", low))
    cumulative_explicit = "cumulative" in low or "till date" in low or "till (in million)" in low
    num_re = re.compile(r"^-?[\d,]+(?:\.\d+)?$")

    def cells(row, a, b):
        return [compact(c) for c in row[a:b] if c is not None and compact(c)]

    sides = {"left": [], "right": []}
    for orig in table:
        row = list(orig) + [None] * max(0, 18 - len(orig))
        for side, name_rng, num_rng in (
            ("left", (0, 3), (3, 9)), ("right", (9, 12), (12, 18)),
        ):
            name_parts = cells(row, *name_rng)
            nums = [c for c in cells(row, *num_rng) if num_re.fullmatch(c)]
            if len(nums) == 2:
                name = " ".join(name_parts)
                circle = lookup_circle(name)
                sides[side].append({"circle": circle, "name": name, "nums": nums})
            elif name_parts and sides[side]:
                # Multi-line service area names such as Andhra / Pradesh or Madhya / Pradesh.
                sides[side][-1]["name"] += " " + " ".join(name_parts)
                sides[side][-1]["circle"] = lookup_circle(sides[side][-1]["name"])
    records = {}
    kind = "monthly" if monthly_explicit else "cumulative" if cumulative_explicit else "undetermined"
    for rec in sides["left"] + sides["right"]:
        circle = rec.get("circle") or lookup_circle(rec.get("name", ""))
        if not circle:
            continue
        try:
            a = float(rec["nums"][0].replace(",", ""))
            b = float(rec["nums"][1].replace(",", ""))
        except (ValueError, IndexError):
            continue
        # Older reports call the table "Number of Porting Requests (in Million)" but list
        # cumulative totals; >5 million is a strong table-wide discriminator. In that case,
        # per-circle requests = current cumulative minus previous cumulative.
        has_million_unit = "million" in low
        if kind == "undetermined" and has_million_unit and max(a, b) > 5:
            effective_kind = "cumulative"
        else:
            effective_kind = kind
        if effective_kind == "cumulative":
            value = round((b - a) * 1_000_000)
        elif effective_kind == "monthly":
            # Newer monthly tables are either counts, or decimal millions.
            value = round(b * 1_000_000) if abs(b) < 5 and has_million_unit else round(b)
        else:
            value = None
        records[circle] = {"mnp_month": value, "mnp_kind": effective_kind, "mnp_prev_value": a, "mnp_curr_value": b}
    return records, kind, len(records)


def extract_mnp_for_report(plumber_pdf, texts: list[str], start_idx: int, end_idx: int, report_month: str):
    best_records, best_kind, best_page, best_count = {}, "missing", None, 0
    for pidx in range(start_idx, end_idx + 1):
        page = plumber_pdf.pages[pidx]
        txt = texts[pidx] or ""
        lo = txt.lower()
        if not ("mnp status" in lo and ("service area" in lo or "licensed service area" in lo)):
            continue
        try:
            tables = page.extract_tables()
        except Exception:
            tables = []
        for table in tables:
            if not table:
                continue
            records, kind, count = parse_mnp_table(table, report_month)
            if count > best_count:
                best_records, best_kind, best_page, best_count = records, kind, pidx + 1, count
    return best_records, best_kind, best_page, best_count


def parse_vlr_page(page: fitz.Page):
    words = page.get_text("words")
    if not words:
        return {}
    lowtext = (page.get_text("text") or "").lower()
    if not any(term in lowtext for term in ("proportion of vlr", "percentage of vlr", "vlr on the date of peak")):
        return {}
    vod_words = [w for w in words if "vodafone" in w[4].lower()]
    idea_words = [w for w in words if w[4].lower() == "idea"]
    if not vod_words:
        return {}
    vod = max(vod_words, key=lambda w: w[2] - w[0])
    idea = min((w for w in idea_words if abs(w[1] - vod[1]) < 35 and w[0] >= vod[0] - 5),
               key=lambda w: abs(w[1] - vod[1]) + abs(w[0] - vod[2]), default=None)
    vi_center = (float(vod[0]) + float(idea[2])) / 2 if idea else (float(vod[0]) + float(vod[2])) / 2
    out = {}
    for line in page_lines(page):
        toks = line_tokens(line)
        if not toks:
            continue
        first_num = None
        for i, tok in enumerate(toks):
            clean = clean_token(tok)
            if DEC_RE.fullmatch(clean) or clean == "-":
                first_num = i
                break
        if first_num is None:
            continue
        circle = lookup_circle(" ".join(toks[:first_num]))
        if not circle:
            continue
        nums = []
        for w in line:
            token = clean_token(w[4])
            if DEC_RE.fullmatch(token) or token == "-":
                nums.append((float(w[0] + w[2]) / 2, token))
        if not nums:
            continue
        nearest = min(nums, key=lambda pair: abs(pair[0] - vi_center))
        try:
            out[circle] = None if nearest[1] == "-" else float(nearest[1].replace(",", ""))
        except ValueError:
            pass
    return out


def extract_vlr_for_report(doc: fitz.Document, texts: list[str], start_idx: int, end_idx: int):
    best: dict[str, float | None] = {}
    best_page = None
    for pidx in range(start_idx, end_idx + 1):
        txt = texts[pidx] or ""
        if not re.search(r"(proportion|percentage) of VLR|VLR on the date of Peak VLR", txt, re.I):
            continue
        vals = parse_vlr_page(doc[pidx])
        nonnull = sum(v is not None for v in vals.values())
        if nonnull > sum(v is not None for v in best.values()):
            best, best_page = vals, pidx + 1
    return best, best_page


def month_range(start: str, end: str) -> list[str]:
    a = pd.Period(start, freq="M")
    b = pd.Period(end, freq="M")
    if b < a:
        raise ValueError("--end must be >= --start")
    return [str(p) for p in pd.period_range(a, b, freq="M")]


def process_pdf(input_pdf: Path, args) -> tuple[list[dict], list[dict], int, int]:
    """Extract each detected TRAI report from one source PDF.

    A source may be the large merged PDF or a supplementary single-report PDF.
    Returns extracted rows, report inventory, page count, and detected headings.
    """
    input_pdf = input_pdf.resolve()
    print(f"\n=== Source PDF: {input_pdf.name} ===", flush=True)
    doc = fitz.open(input_pdf)
    texts = []
    print(f"Reading searchable text from {len(doc)} pages...", flush=True)
    update_every = max(1, len(doc) // 100)
    for page_number, page in enumerate(doc, 1):
        texts.append(page.get_text("text") or "")
        if page_number % update_every == 0 or page_number == len(doc):
            print_progress(page_number, len(doc), f"Pages: {input_pdf.name[:13]}")

    starts = []
    print("Detecting report headings...", flush=True)
    for idx, txt in enumerate(texts):
        if TITLE_RE.search(txt):
            period, raw_period = parse_period_from_title(txt)
            starts.append({
                "start_idx": idx,
                "start_page": idx + 1,
                "report_month": period,
                "heading_date_text": raw_period,
                "release_date": parse_release_date(txt),
                "title_page_text": txt[:1500],
            })

    # Some TRAI supplemental PDFs have a broken/missing searchable text layer on
    # the cover, so the heading detector sees zero headings even though the
    # annexure tables are still extractable. For known single-report filenames,
    # treat the whole PDF as one report and assign only the month documented by
    # the official report number. This is not used for arbitrary filenames.
    fallback_month = filename_month_override(input_pdf)
    if fallback_month and not any(rec.get("report_month") for rec in starts):
        first_text = "\n".join(texts[:3])
        starts = [{
            "start_idx": 0,
            "start_page": 1,
            "report_month": fallback_month,
            "heading_date_text": f"filename fallback; report month {fallback_month}",
            "release_date": parse_release_date(first_text),
            "title_page_text": first_text[:1500],
        }]
        print(
            f"Heading text was not detected; using verified filename mapping for {input_pdf.name}: "
            f"{fallback_month}",
            flush=True,
        )

    for i, rec in enumerate(starts):
        rec["end_idx"] = starts[i + 1]["start_idx"] - 1 if i + 1 < len(starts) else len(doc) - 1
        rec["end_page"] = rec["end_idx"] + 1
        rec["report_id"] = f"{input_pdf.name}:p{rec['start_page']}-{rec['end_page']}"
        rec["source_pdf"] = input_pdf.name

    print(
        f"Prepared {len(starts)} report block(s) in {input_pdf.name} "
        "(detected heading or verified filename fallback).",
        flush=True,
    )
    extracted_records: list[dict] = []
    inventory: list[dict] = []

    # PDF table layout extraction is used for MNP values.
    with pdfplumber.open(input_pdf) as plumber_pdf:
        for n, report in enumerate(starts, 1):
            month = report["report_month"]
            in_scope = bool(month and args.start <= month <= args.end)
            print_progress(n - 1, len(starts), f"Reports: {input_pdf.name[:13]}")
            print(
                f"\n{input_pdf.name}: report {n}/{len(starts)}: "
                f"{month or 'UNKNOWN'} | pages {report['start_page']}-{report['end_page']}",
                flush=True,
            )

            rows, wireless_page, score, candidates = extract_wireless_table(
                doc, texts, report["start_idx"], report["end_idx"], month or ""
            )
            rowmap: dict[str, dict] = {}
            for row in rows:
                current = rowmap.get(row["circle"])
                if current is None or (not current["row_check_ok"] and row["row_check_ok"]):
                    rowmap[row["circle"]] = row

            vlr, vlr_page = extract_vlr_for_report(
                doc, texts, report["start_idx"], report["end_idx"]
            )
            if month:
                mnp, mnp_kind, mnp_page, mnp_count = extract_mnp_for_report(
                    plumber_pdf, texts, report["start_idx"], report["end_idx"], month
                )
            else:
                mnp, mnp_kind, mnp_page, mnp_count = {}, "missing", None, 0

            header_pair = []
            if wireless_page:
                table_txt = texts[wireless_page - 1] or ""
                monrx = re.compile(
                    r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
                    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
                    r"Dec(?:ember)?)[\s-]+(\d{2}|20\d{2})\b", re.I
                )
                seen = set()
                for mm in monrx.finditer(table_txt):
                    mn = MONTH_LOOKUP.get(mm.group(1).lower())
                    yy = int(mm.group(2))
                    yy = yy if yy >= 100 else 2000 + yy
                    if mn and 1900 <= yy <= 2100:
                        val = f"{yy:04d}-{mn:02d}"
                        if val not in seen:
                            seen.add(val)
                            header_pair.append(val)
                    if len(header_pair) >= 2:
                        break

            found_good = [r for r in rowmap.values() if r.get("row_check_ok")]
            missing_circles = [
                c for c in CANON if c not in rowmap or not rowmap[c].get("row_check_ok")
            ]
            vlr_found = sum(v is not None for v in vlr.values())
            header_matches = bool(month and len(header_pair) >= 2 and header_pair[-1] == month)

            inv = {
                "report_id": report["report_id"],
                "source_pdf": input_pdf.name,
                "report_month": month,
                "heading_date_text": report["heading_date_text"],
                "release_date": report["release_date"],
                "start_page": report["start_page"],
                "end_page": report["end_page"],
                "in_requested_scope": in_scope,
                "wireless_table_page": wireless_page,
                "wireless_rows_found": len(rowmap),
                "wireless_rows_reconciled": len(found_good),
                "missing_or_unverified_circles": "; ".join(missing_circles),
                "wireless_page_score": str(score),
                "wireless_header_months": ";".join(header_pair),
                "wireless_header_matches_report_month": header_matches,
                "vlr_table_page": vlr_page,
                "vlr_circle_values_found": vlr_found,
                "mnp_table_page": mnp_page,
                "mnp_circle_values_found": len(mnp),
                "mnp_kind": mnp_kind,
                "status": "ok" if wireless_page and len(found_good) >= 18 else "REVIEW_WIRELESS_TABLE",
                "selected_for_output": False,
            }
            inventory.append(inv)

            if not month or not (args.keep_out_of_scope or in_scope):
                print_progress(n, len(starts), f"Reports: {input_pdf.name[:13]}")
                continue

            prev_month = str(pd.Period(month, freq="M") - 1)
            for circle in CANON:
                base = {
                    "circle": circle,
                    "prev_month": prev_month,
                    "report_month": month,
                    "total_prev": float("nan"),
                    "total_curr": float("nan"),
                    "total_net_add": float("nan"),
                    "row_check_ok": False,
                    "column_ambiguous": False,
                    "assigned_operators": "",
                    "vi_vlr_pct": vlr.get(circle),
                    "mnp_month": mnp.get(circle, {}).get("mnp_month", float("nan")),
                    "mnp_kind": mnp.get(circle, {}).get("mnp_kind", mnp_kind if mnp else "missing"),
                    "mnp_prev_value": mnp.get(circle, {}).get("mnp_prev_value", float("nan")),
                    "mnp_curr_value": mnp.get(circle, {}).get("mnp_curr_value", float("nan")),
                    "release_date": report["release_date"],
                    "source_file": report["report_id"],
                    "source_pdf": input_pdf.name,
                    "source_start_page": report["start_page"],
                    "source_end_page": report["end_page"],
                    "wireless_table_page": wireless_page,
                    "vlr_table_page": vlr_page,
                    "mnp_table_page": mnp_page,
                    "extraction_status": "circle_row_not_recovered",
                }
                for op in ("airtel", "rcom", "vi", "bsnl", "vno", "mtnl", "jio"):
                    base[f"{op}_prev"] = float("nan")
                    base[f"{op}_curr"] = float("nan")
                if circle in rowmap:
                    base.update(rowmap[circle])
                    base["extraction_status"] = (
                        "ok" if rowmap[circle]["row_check_ok"]
                        else "row_found_but_reconciliation_failed"
                    )
                    # `row.update()` does not override the identifying fields, but set them again
                    # defensively so the month/circle keys always stay correct.
                    base["circle"] = circle
                    base["report_month"] = month
                    base["source_file"] = report["report_id"]
                    base["source_pdf"] = input_pdf.name
                extracted_records.append(base)

            print_progress(n, len(starts), f"Reports: {input_pdf.name[:13]}")

    page_count = len(doc)
    heading_count = len(starts)
    doc.close()
    return extracted_records, inventory, page_count, heading_count


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf", help="Path to the main merged searchable TRAI PDF")
    ap.add_argument("--outdir", default="ancp_extraction", help="Output directory")
    ap.add_argument("--start", default="2021-09", help="First in-scope report month")
    ap.add_argument("--end", default="2026-08", help="Last in-scope report month")
    ap.add_argument(
        "--supplemental-dir", default=None,
        help="Folder containing extra single-report PDFs; defaults to the main PDF's folder",
    )
    ap.add_argument(
        "--keep-out-of-scope", action="store_true",
        help="Also output detected reports outside the requested period",
    )
    ap.add_argument(
        "--no-auto-supplementals", action="store_true",
        help="Only process the main PDF, ignoring other PDFs in its folder",
    )
    args = ap.parse_args()

    main_pdf = Path(args.pdf).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    if not main_pdf.exists():
        raise SystemExit(f"PDF not found: {main_pdf}")
    if pd.Period(args.end, freq="M") < pd.Period(args.start, freq="M"):
        raise SystemExit("--end must be the same as or later than --start")
    outdir.mkdir(parents=True, exist_ok=True)

    # Automatically include supplemental monthly PDFs stored beside 3rd_merged.pdf.
    # This matches the user's folder setup and avoids manual filename mapping.
    supplemental_dir = Path(args.supplemental_dir).expanduser().resolve() if args.supplemental_dir else main_pdf.parent
    sources = [main_pdf]
    if not args.no_auto_supplementals and supplemental_dir.exists():
        for candidate in sorted(supplemental_dir.glob("*.pdf"), key=lambda p: p.name.lower()):
            if candidate.resolve() == main_pdf:
                continue
            # Avoid accidentally re-processing prior generated combined PDFs / output products.
            if candidate.name.lower() in {"3rd_merged_all_60.pdf", "merged_all_60.pdf"}:
                continue
            sources.append(candidate.resolve())

    print(f"Main PDF: {main_pdf}")
    print(f"Supplemental PDFs to check: {max(0, len(sources) - 1)}")
    print(f"Requested window: {args.start} through {args.end} ({len(month_range(args.start, args.end))} months)")

    all_records: list[dict] = []
    all_inventory: list[dict] = []
    source_summaries: list[dict] = []
    for source in sources:
        try:
            records, inventory, page_count, heading_count = process_pdf(source, args)
            # Skip unrelated PDFs cleanly when auto-scanning the supplement folder.
            if heading_count == 0:
                print(f"No TRAI report headings found in {source.name}; skipped.")
                continue
            all_records.extend(records)
            all_inventory.extend(inventory)
            source_summaries.append({
                "source_pdf": source.name,
                "pages": page_count,
                "report_headings": heading_count,
                "in_scope_reports": sum(bool(r.get("in_requested_scope")) for r in inventory),
            })
        except Exception as exc:
            if source == main_pdf:
                raise
            print(f"WARNING: Could not process supplemental {source.name}: {exc}")
            source_summaries.append({
                "source_pdf": source.name, "pages": None, "report_headings": 0,
                "in_scope_reports": 0, "error": str(exc),
            })

    if not all_inventory:
        raise SystemExit("No TRAI report headings were found in the supplied PDFs.")

    # Select one source report per month. If duplicates exist, use the report with the most
    # reconciled circle rows, then the most VLR/MNP values. This avoids double-counting months.
    inv_df = pd.DataFrame(all_inventory)
    inv_df["report_month_sort"] = inv_df["report_month"].fillna("9999-99")
    inv_df["quality_score"] = (
        pd.to_numeric(inv_df["wireless_rows_reconciled"], errors="coerce").fillna(0) * 10000
        + pd.to_numeric(inv_df["vlr_circle_values_found"], errors="coerce").fillna(0) * 100
        + pd.to_numeric(inv_df["mnp_circle_values_found"], errors="coerce").fillna(0)
    )
    # Tie-break: prefer larger source reports (often the merged PDF containing the full annexures),
    # then stable filename/page ordering.
    inv_df["source_pdf_tiebreak"] = inv_df["source_pdf"].eq(main_pdf.name).astype(int)
    candidate_inv = inv_df[
        inv_df["report_month"].notna()
        & inv_df["in_requested_scope"].astype(bool)
    ].copy()
    candidate_inv = candidate_inv.sort_values(
        ["report_month", "quality_score", "source_pdf_tiebreak", "source_pdf", "start_page"],
        ascending=[True, False, False, True, True],
        kind="stable",
    )
    selected_ids = set()
    for _, group in candidate_inv.groupby("report_month", sort=False):
        chosen_id = group.iloc[0]["report_id"]
        selected_ids.add(chosen_id)
    inv_df["selected_for_output"] = inv_df["report_id"].isin(selected_ids)
    duplicate_ids = set(
        inv_df.loc[
            inv_df["report_month"].notna()
            & inv_df["in_requested_scope"].astype(bool)
            & ~inv_df["selected_for_output"],
            "report_id",
        ]
    )
    inv_df.loc[inv_df["report_id"].isin(duplicate_ids), "status"] = "duplicate_month_not_selected"

    raw_df = pd.DataFrame(all_records)
    if raw_df.empty:
        extracted_df = raw_df
    else:
        extracted_df = raw_df[raw_df["source_file"].isin(selected_ids)].copy()
        # Required chronology: first month, then the 22 canonical circle names per month.
        circle_order = {circle: i for i, circle in enumerate(CANON)}
        extracted_df["_circle_order"] = extracted_df["circle"].map(circle_order).fillna(999).astype(int)
        extracted_df = extracted_df.sort_values(
            ["report_month", "_circle_order"], kind="stable"
        ).drop(columns="_circle_order").reset_index(drop=True)

    expected = month_range(args.start, args.end)
    selected_months = set(inv_df.loc[inv_df["selected_for_output"], "report_month"].dropna())
    missing_months_rows = [
        {
            "missing_report_month": month,
            "status": "No source report was detected for this month. Obtain/verify the report before modelling.",
        }
        for month in expected if month not in selected_months
    ]

    # Keep report inventory in chronological month order and source page order.
    inv_df = inv_df.sort_values(
        ["report_month_sort", "selected_for_output", "source_pdf", "start_page"],
        ascending=[True, False, True, True], kind="stable"
    ).drop(columns=["report_month_sort", "quality_score", "source_pdf_tiebreak"])
    missing_df = pd.DataFrame(missing_months_rows, columns=["missing_report_month", "status"])
    source_summary_df = pd.DataFrame(source_summaries)

    extracted_path = outdir / "extracted.csv"
    inventory_path = outdir / "report_inventory.csv"
    missing_path = outdir / "missing_months.csv"
    source_summary_path = outdir / "source_pdf_summary.csv"
    extracted_df.to_csv(extracted_path, index=False)
    inv_df.to_csv(inventory_path, index=False)
    missing_df.to_csv(missing_path, index=False)
    source_summary_df.to_csv(source_summary_path, index=False)

    in_scope_inventory = inv_df[
        inv_df["in_requested_scope"].fillna(False).astype(bool)
        & inv_df["selected_for_output"].fillna(False).astype(bool)
    ]
    reconciled_rows = int(extracted_df["row_check_ok"].fillna(False).sum()) if len(extracted_df) else 0
    summary_lines = [
        "ANCP MERGED + SUPPLEMENTAL PDF EXTRACTION SUMMARY",
        f"Main input PDF: {main_pdf.name}",
        f"PDF files recognized (heading or verified filename fallback): {len(source_summaries)}",
        f"Requested period: {args.start} through {args.end} ({len(expected)} months)",
        f"Unique in-scope report months selected: {len(selected_months)}",
        f"Missing expected months: {', '.join(r['missing_report_month'] for r in missing_months_rows) or 'none'}",
        f"Extracted circle-month rows: {len(extracted_df)}",
        f"Rows whose main subscriber table reconciles: {reconciled_rows}",
        f"Rows with a Vi VLR value: {int(extracted_df['vi_vlr_pct'].notna().sum()) if len(extracted_df) else 0}",
        f"Rows with MNP values: {int(extracted_df['mnp_month'].notna().sum()) if len(extracted_df) else 0}",
        f"Selected reports flagged for wireless-table review: {int((in_scope_inventory['status'] != 'ok').sum())}",
        f"Duplicate in-scope reports not selected: {len(duplicate_ids)}",
        "Rows are sorted by report_month ascending, then by the 22-circle canonical order.",
        "Missing source data is left missing, never replaced with zero or invented.",
        "Check report_inventory.csv for report-level quality and selected-source decisions.",
    ]
    (outdir / "extraction_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n" + "\n".join(summary_lines))
    print(
        f"\nSaved: {extracted_path}\n       {inventory_path}\n       {missing_path}\n"
        f"       {source_summary_path}\n       {outdir / 'extraction_summary.txt'}"
    )


if __name__ == "__main__":
    main()
