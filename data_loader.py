"""
MnBrain - Data Ingestion & Validation Layer
============================================

Abstraction layer that lets the analytical engines consume data from two
sources without changing the core algorithms:

  1. The bundled Demonstration Dataset (synthetic), or
  2. User-uploaded geological / stockpile files (CSV or Excel).

Uploaded data is normalised to the standard internal schema and validated
before it is passed to the kriging or optimization engine. Validation
results are machine-readable so callers can decide whether to proceed.

Standard geological sample schema:
    Borehole_ID, X, Y, Depth, Mn_Grade, Fe_Grade

Standard stockpile schema:
    Stockpile_ID, Available_Tons, Mn_Grade, Fe_Grade, Cost_per_Ton

Reasonable column aliases are accepted and normalised internally
(e.g. Easting -> X, Mn% -> Mn_Grade, Tonnes -> Available_Tons).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd


class DataLoadError(Exception):
    """Raised when an uploaded file cannot be read as a table."""


GEOLOGICAL_COLUMNS = ["Borehole_ID", "X", "Y", "Depth", "Mn_Grade", "Fe_Grade"]
STOCKPILE_COLUMNS = ["Stockpile_ID", "Available_Tons", "Mn_Grade", "Fe_Grade",
                     "Cost_per_Ton"]

GEOLOGICAL_SOURCE_OPTIONS = ["Demonstration Dataset", "Upload Geological Data"]
STOCKPILE_SOURCE_OPTIONS = ["Demonstration Dataset", "Upload Stockpile Data"]

_GEOLOGICAL_ALIASES = {
    "borehole_id": "Borehole_ID", "hole_id": "Borehole_ID",
    "bhid": "Borehole_ID", "sample_id": "Borehole_ID", "holeid": "Borehole_ID",
    "x": "X", "easting": "X", "east": "X", "x_coord": "X", "xcoord": "X",
    "y": "Y", "northing": "Y", "north": "Y", "y_coord": "Y", "ycoord": "Y",
    "depth": "Depth", "z": "Depth", "depth_m": "Depth", "depthm": "Depth",
    "mn_grade": "Mn_Grade", "mn": "Mn_Grade", "mn%": "Mn_Grade",
    "mn_pct": "Mn_Grade", "mn_%": "Mn_Grade",
    "fe_grade": "Fe_Grade", "fe": "Fe_Grade", "fe%": "Fe_Grade",
    "fe_pct": "Fe_Grade", "fe_%": "Fe_Grade",
}

_STOCKPILE_ALIASES = {
    "stockpile_id": "Stockpile_ID", "stockpile": "Stockpile_ID",
    "id": "Stockpile_ID", "pile_id": "Stockpile_ID",
    "available_tons": "Available_Tons", "available_tonnes": "Available_Tons",
    "tons": "Available_Tons", "tonnes": "Available_Tons",
    "available": "Available_Tons", "avail_tons": "Available_Tons",
    "mn_grade": "Mn_Grade", "mn": "Mn_Grade", "mn%": "Mn_Grade",
    "mn_pct": "Mn_Grade",
    "fe_grade": "Fe_Grade", "fe": "Fe_Grade", "fe%": "Fe_Grade",
    "fe_pct": "Fe_Grade",
    "cost_per_ton": "Cost_per_Ton", "cost_per_tonne": "Cost_per_Ton",
    "cost": "Cost_per_Ton", "costperton": "Cost_per_Ton",
    "cost_per_t": "Cost_per_Ton",
}


def _normalise_columns(df: pd.DataFrame, aliases: dict,
                       required: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Rename accepted aliases to canonical names. Returns (frame, missing)."""
    lower = {}
    for col in df.columns:
        lower.setdefault(str(col).strip().lower(), col)

    found = {}
    for alias, canonical in aliases.items():
        if alias in lower and canonical not in found:
            found[canonical] = lower[alias]

    missing = [c for c in required if c not in found]
    rename = {src: canon for canon, src in found.items() if src != canon}
    return df.rename(columns=rename), missing


_GRADE_COLS = ["X", "Y", "Depth", "Mn_Grade", "Fe_Grade"]


# ---------------------------------------------------------------------------
# Geological data
# ---------------------------------------------------------------------------

_GRADE_COLS = ["X", "Y", "Depth", "Mn_Grade", "Fe_Grade"]


def _to_clean_numeric(series: pd.Series) -> pd.Series:
    """Coerce to numeric, mapping non-finite values (incl. inf) to NaN."""
    return (pd.to_numeric(series, errors="coerce")
            .replace([np.inf, -np.inf], np.nan))


def _is_blank(value: pd.Series) -> pd.Series:
    """True for NaN, empty or na-like string identifiers."""
    stripped = value.astype(str).str.strip().str.lower()
    return (value.isna() | stripped.eq("") |
            stripped.isin(["nan", "none", "null", "na", "<na>", "<nat>"]))


def validate_geological_data(df: pd.DataFrame | None) -> dict:
    """Validate a geological sample table against the standard schema.

    Returns
    -------
    dict with keys:
        ok, errors, warnings, cleaned, summary

    ``cleaned`` holds the normalised, numeric, de-duplicated rows that are
    safe to pass to the kriging engine. ``summary`` contains data-quality
    counters and ranges for display.
    """
    result = {
        "ok": False, "errors": [], "warnings": [], "cleaned": None,
        "summary": {"n_total": 0, "n_valid": 0, "n_missing_values": 0,
                    "n_duplicates": 0, "mn_min": None, "mn_max": None,
                    "fe_min": None, "fe_max": None, "depth_min": None,
                    "depth_max": None, "coord_span": None},
    }

    if not isinstance(df, pd.DataFrame):
        result["errors"].append("Uploaded geological data is not a table.")
        return result
    if df.empty:
        result["errors"].append("Uploaded geological data contains no rows.")
        return result

    clean, missing = _normalise_columns(df, _GEOLOGICAL_ALIASES,
                                        GEOLOGICAL_COLUMNS)
    if missing:
        result["errors"].append(
            "Missing required column(s): "
            + ", ".join(missing)
            + ". Accepted names/aliases include Borehole_ID, X/Easting, "
            "Y/Northing, Depth and Mn_Grade / Fe_Grade.")
        return result

    work = clean[GEOLOGICAL_COLUMNS].copy()
    for col in _GRADE_COLS:
        work[col] = _to_clean_numeric(work[col])

    n_total = len(work)
    blank_ids = int(_is_blank(work["Borehole_ID"]).sum())
    bad_numeric_mask = work[["X", "Y", "Depth", "Mn_Grade", "Fe_Grade"]].isna()
    n_bad_numeric = int(bad_numeric_mask.any(axis=1).sum())

    if blank_ids:
        result["warnings"].append(f"{blank_ids} row(s) have an empty "
                                  "Borehole_ID; sample records without an ID "
                                  "are dropped.")

    valid = work.dropna(subset=["X", "Y", "Depth", "Mn_Grade", "Fe_Grade"])
    valid = valid[~_is_blank(valid["Borehole_ID"])]

    if valid.empty:
        result["errors"].append("No valid numeric geological samples remain "
                                "after cleaning (check values and IDs).")
        return result

    if n_bad_numeric:
        result["warnings"].append(
            f"{n_bad_numeric} row(s) with missing/invalid numeric values "
            "were dropped; check source data for commas or text cells.")

    if (valid["Depth"] < 0).any():
        result["errors"].append("Geological samples contain negative depths; "
                                "depth must be zero or greater.")
    if ((valid["Mn_Grade"] < 0) | (valid["Mn_Grade"] > 100)).any():
        result["errors"].append("Mn grade values must lie between 0 and 100%.")
    if ((valid["Fe_Grade"] < 0) | (valid["Fe_Grade"] > 100)).any():
        result["errors"].append("Fe grade values must lie between 0 and 100%.")

    dup_cols = ["Borehole_ID", "X", "Y", "Depth"]
    n_duplicates = int(valid.duplicated(subset=dup_cols, keep="first").sum())
    if n_duplicates:
        result["warnings"].append(
            f"{n_duplicates} duplicate sample record(s) removed "
            "(same borehole/coordinate/depth).")
        valid = valid.drop_duplicates(subset=dup_cols, keep="first")

    span_x = float(valid["X"].max() - valid["X"].min())
    span_y = float(valid["Y"].max() - valid["Y"].min())

    result["summary"] = {
        "n_total": n_total,
        "n_valid": len(valid),
        "n_missing_values": int(n_bad_numeric),
        "n_duplicates": n_duplicates,
        "mn_min": float(valid["Mn_Grade"].min()),
        "mn_max": float(valid["Mn_Grade"].max()),
        "fe_min": float(valid["Fe_Grade"].min()),
        "fe_max": float(valid["Fe_Grade"].max()),
        "depth_min": float(valid["Depth"].min()),
        "depth_max": float(valid["Depth"].max()),
        "coord_span": max(span_x, span_y),
    }

    if span_x <= 1e-9 and span_y <= 1e-9:
        result["warnings"].append("All samples share a single coordinate "
                                  "point; kriging will be based on depth "
                                  "variation only.")
    elif max(span_x, span_y) > 1.0e6:
        result["warnings"].append("Coordinate span is very large (> 1e6 m); "
                                  "verify X/Y units (metres expected).")

    result["cleaned"] = valid.reset_index(drop=True)
    result["ok"] = not result["errors"]
    return result


# ---------------------------------------------------------------------------
# Stockpile data
# ---------------------------------------------------------------------------

def validate_stockpile_data(df: pd.DataFrame | None) -> dict:
    """Validate a stockpile table against the standard schema.

    Returns the same structure as :func:`validate_geological_data`.
    """
    result = {
        "ok": False, "errors": [], "warnings": [], "cleaned": None,
        "summary": {"n_total": 0, "n_valid": 0, "n_missing_values": 0,
                    "n_duplicates": 0, "total_tonnage": 0.0,
                    "mn_min": None, "mn_max": None, "fe_min": None,
                    "fe_max": None, "cost_min": None, "cost_max": None},
    }

    if not isinstance(df, pd.DataFrame):
        result["errors"].append("Uploaded stockpile data is not a table.")
        return result
    if df.empty:
        result["errors"].append("Uploaded stockpile data contains no rows.")
        return result

    clean, missing = _normalise_columns(df, _STOCKPILE_ALIASES,
                                        STOCKPILE_COLUMNS)
    if missing:
        result["errors"].append(
            "Missing required column(s): "
            + ", ".join(missing)
            + ". Accepted names/aliases include Stockpile_ID, "
            "Available_Tons/Tonnes, Mn_Grade, Fe_Grade and Cost_per_Ton.")
        return result

    work = clean[STOCKPILE_COLUMNS].copy()
    for col in ["Available_Tons", "Mn_Grade", "Fe_Grade", "Cost_per_Ton"]:
        work[col] = _to_clean_numeric(work[col])

    n_total = len(work)
    n_bad = int(work[["Available_Tons", "Mn_Grade", "Fe_Grade",
                      "Cost_per_Ton"]].isna().any(axis=1).sum())
    blank_ids = int(_is_blank(work["Stockpile_ID"]).sum())
    work = work.dropna(subset=["Available_Tons", "Mn_Grade", "Fe_Grade",
                               "Cost_per_Ton"])
    work = work[~_is_blank(work["Stockpile_ID"])]
    if blank_ids:
        result["warnings"].append(f"{blank_ids} row(s) have an empty "
                                  "Stockpile_ID and were dropped.")
    if work.empty:
        result["errors"].append("No valid numeric stockpile rows remain "
                                "after cleaning.")
        return result

    if n_bad:
        result["warnings"].append(
            f"{n_bad} row(s) with missing/invalid numeric values were dropped.")

    if (work["Available_Tons"] < 0).any():
        result["errors"].append("Stockpile availability cannot be negative.")
    if (work["Cost_per_Ton"] <= 0).any():
        result["errors"].append("Stockpile cost per ton must be positive.")
    if ((work["Mn_Grade"] < 0) | (work["Mn_Grade"] > 100)).any():
        result["errors"].append("Mn grade values must lie between 0 and 100%.")
    if ((work["Fe_Grade"] < 0) | (work["Fe_Grade"] > 100)).any():
        result["errors"].append("Fe grade values must lie between 0 and 100%.")

    n_duplicates = int(work.duplicated(subset=["Stockpile_ID"], keep="first").sum())
    if n_duplicates:
        result["warnings"].append(
            f"{n_duplicates} duplicate Stockpile_ID(s) removed; only the "
            "first occurrence is used.")
        work = work.drop_duplicates(subset=["Stockpile_ID"], keep="first")

    if float(work["Available_Tons"].sum()) <= 0:
        result["errors"].append("Total available stockpile tonnage is zero.")

    result["summary"] = {
        "n_total": n_total,
        "n_valid": len(work),
        "n_missing_values": int(n_bad),
        "n_duplicates": n_duplicates,
        "total_tonnage": float(work["Available_Tons"].sum()),
        "mn_min": float(work["Mn_Grade"].min()),
        "mn_max": float(work["Mn_Grade"].max()),
        "fe_min": float(work["Fe_Grade"].min()),
        "fe_max": float(work["Fe_Grade"].max()),
        "cost_min": float(work["Cost_per_Ton"].min()),
        "cost_max": float(work["Cost_per_Ton"].max()),
    }

    result["cleaned"] = work.reset_index(drop=True)
    result["ok"] = not result["errors"]
    return result


# ---------------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------------

def parse_upload(uploaded_file, file_name: str | None = None) -> pd.DataFrame:
    """Parse an uploaded CSV/Excel file into a DataFrame.

    Raises :class:`DataLoadError` with a friendly message when the file
    cannot be read.
    """
    name = (file_name or getattr(uploaded_file, "name", None) or "").strip()
    low = name.lower()
    try:
        if low.endswith((".xlsx", ".xls", ".xlsm")):
            return pd.read_excel(uploaded_file)
        if low.endswith((".csv", ".txt")):
            return _read_csv(uploaded_file.getvalue())
        if not name:
            # Bare uploaded bytes: attempt CSV, then Excel.
            try:
                return _read_csv(uploaded_file.getvalue())
            except Exception:
                return pd.read_excel(uploaded_file)
        raise DataLoadError(
            f"Unsupported file type '{name}'. Please upload a CSV or Excel "
            "(.xlsx) file.")
    except DataLoadError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface a friendly message
        raise DataLoadError(
            f"Could not read '{name or 'file'}'. Ensure it is a valid CSV or "
            f"Excel file. ({type(exc).__name__})") from exc


def _read_csv(data: bytes) -> pd.DataFrame:
    text = data.decode("utf-8-sig", errors="replace")
    try:
        df = pd.read_csv(io.StringIO(text))
    except pd.errors.ParserError:
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    if df.shape[1] <= 1 and ";" in text.splitlines()[0]:
        df = pd.read_csv(io.StringIO(text), sep=";")
    if df.shape[1] <= 1 and "\t" in text.splitlines()[0]:
        df = pd.read_csv(io.StringIO(text), sep="\t")
    return df


# ---------------------------------------------------------------------------
# Demonstration data
# ---------------------------------------------------------------------------

def load_demonstration_geological_samples(pit_name: str) -> pd.DataFrame:
    """Return the deterministic synthetic borehole sample frame for a pit."""
    from geospatial_model import get_demonstration_samples  # late import
    return get_demonstration_samples(pit_name)


def load_demonstration_stockpiles() -> pd.DataFrame:
    """Return the bundled synthetic stockpile dataset."""
    from optimization_engine import load_sample_stockpiles  # late import
    return load_sample_stockpiles()


if __name__ == "__main__":
    geo = load_demonstration_geological_samples("Balaghat")
    vr = validate_geological_data(geo)
    print("Geological validation ok:", vr["ok"])
    print("Summary:", vr["summary"])

    stock = load_demonstration_stockpiles()
    sr = validate_stockpile_data(stock)
    print("Stockpile validation ok:", sr["ok"])
    print("Summary:", sr["summary"])