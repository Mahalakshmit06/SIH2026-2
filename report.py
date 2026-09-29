"""
MnBrain - Decision Support Report Generator
============================================

Builds a decision-support report from a completed blend optimisation run.
Three export formats are available, all dependency-free:

  * Plain-text report (.txt)  - printable, copy-paste friendly
  * CSV package               - allocation, scenario summary and
                                constraint-verification tables
  * PDF report (.pdf)         - minimal pure-standard-library PDF writer
                                (no reportlab / fpdf / latex required)

A report carries the exact scenario inputs, the selected stockpile
allocations, the blended quality metrics and the independent constraint
verification (PASS/FAIL) - no signatures are faked or implied.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def _fmt_pct(value) -> str:
    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_rs(value) -> str:
    try:
        return f"Rs {float(value):,.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_tons(value) -> str:
    try:
        return f"{float(value):,.0f} t"
    except (TypeError, ValueError):
        return "n/a"


def decision_report_text(blend: dict, ctx: dict[str, Any] | None = None) -> str:
    """Return the full decision-support report as a printable string."""
    ctx = ctx or {}
    scenario = ctx.get("scenario", {})
    block_model = ctx.get("block_model", {})
    verification = ctx.get("verification", {})
    generated_at = ctx.get(
        "generated_at",
        datetime.now().strftime("%d %b %Y, %H:%M:%S"))

    L: list[str] = []
    L.append("=" * 62)
    L.append("MnBrain - Decision Support Report")
    L.append("Geospatial AI & Grade Blending Intelligence")
    L.append("=" * 62)
    L.append(f"Generated              : {generated_at}")
    L.append(f"Data source            : {ctx.get('data_source', 'n/a')}")

    if block_model:
        L.append("-" * 62)
        L.append("3D Block Model Summary")
        L.append("-" * 62)
        L.append(f"  Kriged blocks                : {block_model.get('n_blocks', 'n/a')}")
        L.append(f"  Average Mn grade            : {_fmt_pct(block_model.get('avg_mn'))}")
        L.append(f"  Average Fe grade            : {_fmt_pct(block_model.get('avg_fe'))}")
        L.append(f"  Screening (UNFC 111) blocks : {block_model.get('unfc_111', 'n/a')}")

    L.append("-" * 62)
    L.append("Optimisation Scenario")
    L.append("-" * 62)
    L.append(f"  Required tonnage        : {_fmt_tons(scenario.get('target_tonnage'))}")
    L.append(f"  Minimum blended Mn      : >= {_fmt_pct(scenario.get('target_mn'))}")
    L.append(f"  Maximum blended Fe      : <= {_fmt_pct(scenario.get('max_fe'))}")
    if scenario.get("max_cost_per_ton") is not None:
        L.append(f"  Maximum cost per ton    : <= {_fmt_rs(scenario.get('max_cost_per_ton'))}")
    else:
        L.append("  Maximum cost per ton    : not applied (unconstrained)")

    L.append("-" * 62)
    L.append("Optimal Stockpile Allocation")
    L.append("-" * 62)
    df = blend.get("result_df")
    alloc = blend.get("allocations")
    if isinstance(df, pd.DataFrame) and alloc is not None:
        L.append(f"{'Stockpile':<16}{'Avail (t)':>12}{'Use (t)':>12}"
                 f"{'Mn%':>8}{'Fe%':>7}{'Cost/t':>12}")
        L.append("-" * 67)
        for row, tonnes in zip(df.itertuples(index=False), alloc):
            L.append(f"{str(row[0]):<16}{float(row[1]):>12,.0f}"
                     f"{float(tonnes):>12,.1f}{float(row[2]):>8.2f}"
                     f"{float(row[3]):>7.2f}{float(row[4]):>12,.2f}")
    else:
        L.append("  (no allocation table available)")

    L.append("-" * 62)
    L.append("Blended Result")
    L.append("-" * 62)
    L.append(f"  Allocated tonnage       : {_fmt_tons(blend.get('total_tonnage'))}")
    L.append(f"  Blended Mn grade        : {_fmt_pct(blend.get('blended_mn'))}")
    L.append(f"  Blended Fe grade        : {_fmt_pct(blend.get('blended_fe'))}")
    L.append(f"  Total cost              : {_fmt_rs(blend.get('total_cost'))}")
    L.append(f"  Cost per ton            : {_fmt_rs(blend.get('cost_per_ton'))}")
    L.append(f"  Stockpile utilisation   : {blend.get('utilization_pct', 0):.1f}%")
    L.append(f"  Cost difference vs ref. : {_fmt_rs(blend.get('savings'))} "
             f"({_fmt_rs(blend.get('savings_per_ton'))})")

    L.append("-" * 62)
    L.append("Constraint Validation")
    L.append("-" * 62)
    checks = verification.get("checks", [])
    if checks:
        for c in checks:
            mark = "PASS" if c["status"] == "PASS" else "FAIL"
            L.append(f"  [{mark}] {c['constraint']:<26} "
                     f"req {c['required']:<14} got {c['achieved']}")
        verdict = "ALL CONSTRAINTS SATISFIED" if verification.get("all_pass") \
            else "CONSTRAINT VIOLATION DETECTED"
        L.append("-" * 62)
        L.append(f"  VERDICT : {verdict}")
    elif verification.get("reason"):
        L.append(f"  {verification['reason']}")

    L.append("=" * 62)
    L.append("Decision context: resulting blend must always be re-checked "
             "against site-specific")
    L.append("quality and operational limits before shipment. Model outputs "
             "are planning aids, not")
    L.append("regulatory certifications.")
    L.append("=" * 62)
    return "\n".join(L)


# ---------------------------------------------------------------------------
# CSV package
# ---------------------------------------------------------------------------

def decision_report_csv_frames(blend: dict,
                               ctx: dict[str, Any] | None = None) -> dict[str, pd.DataFrame]:
    """Return {filename: DataFrame} for the report's CSV package."""
    ctx = ctx or {}
    scenario = ctx.get("scenario", {})
    frames: dict[str, pd.DataFrame] = {}

    df = blend.get("result_df")
    alloc = blend.get("allocations")
    if isinstance(df, pd.DataFrame) and alloc is not None:
        out = df.copy()
        out["Allocated_Tons"] = alloc
        frames["allocation_plan.csv"] = out

    summary = {
        "Metric": ["Status", "Required tonnage (t)", "Minimum Mn (%)",
                   "Maximum Fe (%)",
                   "Maximum cost/ton (Rs/t)" if scenario.get("max_cost_per_ton") is not None
                   else "Maximum cost/ton (Rs/t)",
                   "Allocated tonnage (t)", "Blended Mn (%)", "Blended Fe (%)",
                   "Total cost (Rs)", "Cost per ton (Rs/t)",
                   "Stockpile utilisation (%)",
                   "Cost difference vs reference (Rs)"],
        "Value": [blend.get("status", ""), scenario.get("target_tonnage"),
                  scenario.get("target_mn"), scenario.get("max_fe"),
                  scenario.get("max_cost_per_ton",
                               "not applied") if scenario.get("max_cost_per_ton") is not None
                  else "not applied",
                  blend.get("total_tonnage"), blend.get("blended_mn"),
                  blend.get("blended_fe"), blend.get("total_cost"),
                  blend.get("cost_per_ton"), blend.get("utilization_pct"),
                  blend.get("savings")],
    }
    frames["scenario_summary.csv"] = pd.DataFrame(summary)

    checks = (ctx.get("verification") or {}).get("checks", [])
    if checks:
        frames["constraint_validation.csv"] = pd.DataFrame({
            "Constraint": [c["constraint"] for c in checks],
            "Required": [c["required"] for c in checks],
            "Achieved": [c["achieved"] for c in checks],
            "Status": [c["status"] for c in checks],
        })
    return frames


# ---------------------------------------------------------------------------
# Minimal dependency-free PDF writer
# ---------------------------------------------------------------------------
# A4 page, Helvetica base-14 font, header + footer, auto page breaks.
# ASCII-safe: currency is written as "Rs", non-latin characters replaced.

_PAGE_W = 595.0
_PAGE_H = 842.0
_MARGIN = 40.0
_FONT_SIZE = 8.5
_LINE_H = 11.0
_CHAR_W_FACTOR = 0.5
_MAX_COLS = 114


def _pdf_escape(text: str) -> bytes:
    cleaned = text.encode("latin-1", "replace").decode("latin-1")
    cleaned = cleaned.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return cleaned.encode("latin-1")


def _wrap(line: str, width: int = _MAX_COLS) -> list[str]:
    if len(line) <= width:
        return [line]
    return [line[i:i + width] for i in range(0, len(line), width)]


def _render_pdf_page(lines: list[bytes], top: float) -> bytes:
    stream = bytearray()
    stream += b"BT\n/F1 " + str(_FONT_SIZE).encode() + b" Tf\n"
    stream += str(_MARGIN).encode() + b" " + str(top).encode() + b" Td\n"
    y = 0.0
    for ln in lines:
        stream += b"(" + ln + b") Tj\n"
        y += _LINE_H
        stream += b"0 -" + str(_LINE_H).encode() + b" Td\n"
    stream += b"ET\n"
    return bytes(stream)


def _pdf_header_footer(page_no: int, total: int, title: str) -> list[bytes]:
    return [
        f"MnBrain - Decision Support Report".encode("latin-1"),
        f"{title}  |  Page {page_no} of {total}".encode("latin-1"),
    ]


def _pdf_escape_title(text: str) -> str:
    return text.encode("latin-1", "replace").decode("latin-1")


def decision_report_pdf(blend: dict, output_path: str,
                        ctx: dict[str, Any] | None = None) -> str:
    """Write the decision report to ``output_path`` as a PDF.

    Uses only the standard library. Returns the destination path.
    """
    ctx = ctx or {}
    generated_at = ctx.get(
        "generated_at",
        datetime.now().strftime("%d %b %Y, %H:%M:%S"))
    title = _pdf_escape_title(ctx.get("data_source", "Blend Optimisation"))

    text_lines = decision_report_text(blend, ctx).splitlines()
    all_lines: list[bytes] = []
    for raw in text_lines:
        for piece in _wrap(raw):
            all_lines.append(_pdf_escape(piece))

    body_height = _PAGE_H - _MARGIN * 2 - _LINE_H * 4.5
    per_page = max(1, int(body_height // (_LINE_H * 1.2)))

    pages: list[list[bytes]] = []
    current: list[bytes] = []
    for ln in all_lines:
        if len(current) >= per_page:
            pages.append(current)
            current = []
        current.append(ln)
    if current:
        pages.append(current)

    objects: list[bytes] = []

    def add_obj(data: bytes) -> int:
        objects.append(data)
        return len(objects)

    # Object 1: catalog, Object 2: pages tree, Object 3: font.
    add_obj(b"<< /Type /Catalog /Pages 2 0 R >>")
    pages_obj_id = add_obj(b"<< /Type /Pages /Kids [] /Count 0 >>")
    font_id = add_obj(b"<< /Type /Font /Subtype /Type1 "
                      b"/BaseFont /Helvetica >>")

    # One content stream + page object per page (with header line).
    page_obj_ids: list[int] = []
    for page_no, page_lines in enumerate(pages, start=1):
        top = _PAGE_H - _MARGIN - _LINE_H * 2.0 - 6.0
        hf = _pdf_header_footer(page_no, len(pages), title)
        full_lines = hf + page_lines
        contents = _render_pdf_page(full_lines, top)
        footer_stream = (
            b"BT /F1 " + str(_FONT_SIZE).encode() + b" Tf " +
            str(_MARGIN).encode() + b" " + str(_MARGIN - 6).encode() +
            b" Td (" + _pdf_escape(
                f"MnBrain - Decision Support Report | {title} | generated "
                f"{generated_at}") + b") Tj ET")
        merged = contents + b"\n" + footer_stream
        content_id = add_obj(b"<< /Length " + str(len(merged)).encode() +
                             b" >>\nstream\n" + merged + b"\nendstream")
        page_obj_ids.append(add_obj(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 " +
            str(_PAGE_W).encode() + b" " + str(_PAGE_H).encode() +
            b"] /Resources << /Font << /F1 " +
            str(font_id).encode() + b" 0 R >> >> /Contents " +
            str(content_id).encode() + b" 0 R >>"))

    kids = b" ".join(str(p).encode() + b" 0 R" for p in page_obj_ids)
    objects[pages_obj_id - 1] = (
        b"<< /Type /Pages /Kids [" + kids + b"] /Count " +
        str(len(pages)).encode() + b" >>")

    # Assemble the file body and xref table (offset 0 is the body start;
    # the %PDF header written before it shifts the absolute offsets).
    header = b"%PDF-1.4\n"
    body = bytearray()
    body_offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        body_offsets.append(len(body))
        body += str(idx).encode() + b" 0 obj\n" + obj + b"\nendobj\n"

    xref_offset = len(header) + len(body)
    xref = bytearray(b"xref\n0 " + str(len(objects) + 1).encode() + b"\n")
    xref += b"0000000000 65535 f \n"
    for off in body_offsets[1:]:
        xref += f"{len(header) + off:010d} 00000 n \n".encode()
    trailer = (b"trailer\n<< /Size " + str(len(objects) + 1).encode() +
               b" /Root 1 0 R >>\nstartxref\n" + str(xref_offset).encode() +
               b"\n%%EOF")

    with open(output_path, "wb") as fh:
        fh.write(header)
        fh.write(bytes(body))
        fh.write(bytes(xref))
        fh.write(bytes(trailer))
    return output_path


def export_decision_report(blend: dict, output_dir: str,
                           ctx: dict[str, Any] | None = None,
                           formats=("txt", "csv", "pdf")) -> list[str]:
    """Export the report in the requested formats.

    Returns the list of files written.
    """
    ctx = ctx or {}
    os.makedirs(output_dir, exist_ok=True)
    stamp = ctx.get("stamp") or datetime.now().strftime("%Y%m%d_%H%M%S")
    written: list[str] = []

    if "txt" in formats:
        txt = decision_report_text(blend, ctx)
        path = os.path.join(output_dir, f"mnbrain_decision_report_{stamp}.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(txt)
        written.append(path)

    if "csv" in formats:
        for name, frame in decision_report_csv_frames(blend, ctx).items():
            path = os.path.join(output_dir, f"{stamp}_{name}")
            frame.to_csv(path, index=False, encoding="utf-8")
            written.append(path)

    if "pdf" in formats:
        path = os.path.join(output_dir, f"mnbrain_decision_report_{stamp}.pdf")
        decision_report_pdf(blend, path, ctx)
        written.append(path)

    return written


if __name__ == "__main__":
    from optimization_engine import (calculate_optimal_blend, verify_blend,
                                     load_sample_stockpiles)

    stock = load_sample_stockpiles()
    blended = calculate_optimal_blend(stock, 5000, 33.0, 20.0)
    scenario = {"target_tonnage": 5000, "target_mn": 33.0, "max_fe": 20.0,
                "max_cost_per_ton": None}
    verification = verify_blend(blended, 5000, 33.0, 20.0, None)
    ctx = {"data_source": "Demonstration Dataset",
           "scenario": scenario,
           "block_model": {"n_blocks": 1280, "avg_mn": 33.07, "avg_fe": 23.12,
                           "unfc_111": 529},
           "verification": verification}

    print(decision_report_text(blended, ctx)[:1800])
    print("...")
    written = export_decision_report(blended, "reports_demo", ctx)
    for w in written:
        print("wrote", w, os.path.getsize(w), "bytes")