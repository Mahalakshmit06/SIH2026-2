"""
MnBrain - Module 2
Prescriptive Grade Blending Optimization (Linear Programming)
=============================================================

Linear-programming engine that finds the cost-minimising allocation of
tonnage from a set of stockpiles such that the blended product meets:

  * a target total tonnage                 (equality)
  * a minimum blended Mn grade             (inequality)
  * a maximum blended Fe grade             (inequality)
  * per-stockpile availability limits      (box constraints)

Objective:  minimize  SUM(c_j * w_j)
  c_j : cost per ton of stockpile j
  w_j : tonnes allocated from stockpile j

Main entry point:
    calculate_optimal_blend(stockpile_df, target_tonnage,
                            target_mn_percent, max_fe_percent=20.0,
                            max_cost_per_ton=None)

Final results can be re-verified against the original constraints with:

    verify_blend(result, target_tonnage, target_mn_percent, max_fe_percent,
                 max_cost_per_ton)

All stockpile values are SYNTHETIC DATA. Financial figures are labelled as
"Cost Difference vs Reference" and are comparison values computed on the
synthetic dataset, not actual MOIL savings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

REQUIRED_COLUMNS = [
    "Stockpile_ID", "Available_Tons", "Mn_Grade", "Fe_Grade", "Cost_per_Ton",
]

SAMPLE_STOCKPILES_PATH = Path(__file__).resolve().parent / "sample_stockpiles.csv"


def load_sample_stockpiles(path: str | Path | None = None) -> pd.DataFrame:
    """Load the bundled synthetic stockpile dataset."""
    file_path = Path(path) if path is not None else SAMPLE_STOCKPILES_PATH
    return pd.read_csv(file_path)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_stockpile_frame(stockpile_df: pd.DataFrame) -> tuple[bool, str, pd.DataFrame]:
    """Validate stockpile data. Returns (ok, message, clean_df)."""
    if stockpile_df is None:
        return False, "Stockpile data is missing.", stockpile_df

    if not isinstance(stockpile_df, pd.DataFrame):
        return False, "Stockpile data must be a pandas DataFrame.", stockpile_df

    if stockpile_df.empty:
        return False, "Stockpile data is empty - no stockpiles available.", stockpile_df

    missing = [c for c in REQUIRED_COLUMNS if c not in stockpile_df.columns]
    if missing:
        return False, (
            "Stockpile data is missing required column(s): "
            + ", ".join(missing) + "."
        ), stockpile_df

    clean = stockpile_df.copy()
    numeric_cols = ["Available_Tons", "Mn_Grade", "Fe_Grade", "Cost_per_Ton"]

    try:
        for col in numeric_cols:
            clean[col] = pd.to_numeric(clean[col], errors="raise")
    except (TypeError, ValueError):
        return False, (
            "Stockpile grade / cost / tonnage columns must be numeric."
        ), stockpile_df

    if clean[numeric_cols].isna().any().any():
        return False, (
            "Stockpile data contains missing numeric values; please fill or "
            "remove them."
        ), stockpile_df

    if (clean["Available_Tons"] < 0).any():
        return False, "Stockpile availability cannot be negative.", stockpile_df
    if (clean["Cost_per_Ton"] <= 0).any():
        return False, "Stockpile cost per ton must be positive.", stockpile_df
    if ((clean["Mn_Grade"] < 0) | (clean["Mn_Grade"] > 100)).any():
        return False, "Mn grade values must be between 0 and 100%.", stockpile_df
    if ((clean["Fe_Grade"] < 0) | (clean["Fe_Grade"] > 100)).any():
        return False, "Fe grade values must be between 0 and 100%.", stockpile_df

    if clean["Available_Tons"].sum() <= 0:
        return False, "Total available stockpile tonnage is zero.", stockpile_df

    return True, "", clean


def _error_result(status: str, message: str) -> dict:
    """Uniform result dict for failed / infeasible runs."""
    return {
        "success": False,
        "status": status,
        "message": message,
        "result_df": None,
        "allocations": None,
        "total_tonnage": 0.0,
        "blended_mn": 0.0,
        "blended_fe": 0.0,
        "total_cost": 0.0,
        "cost_per_ton": 0.0,
        "utilization_pct": 0.0,
        "savings": 0.0,
        "savings_per_ton": 0.0,
    }


def _max_possible_mn(clean: pd.DataFrame, target_tonnage: float) -> float:
    """Maximum Mn content actually achievable for a given tonnage."""
    unit = clean["Mn_Grade"].values
    cap = clean["Available_Tons"].values
    order = np.argsort(-unit)
    need = float(target_tonnage)
    total = 0.0
    for j in order:
        take = min(cap[j], need)
        total += take * unit[j]
        need -= take
        if need <= 1e-9:
            break
    return total if need <= 1e-9 else -np.inf


def _min_possible_fe(clean: pd.DataFrame, target_tonnage: float) -> float:
    """Minimum Fe content actually achievable for a given tonnage."""
    unit = clean["Fe_Grade"].values
    cap = clean["Available_Tons"].values
    order = np.argsort(unit)
    need = float(target_tonnage)
    total = 0.0
    for j in order:
        take = min(cap[j], need)
        total += take * unit[j]
        need -= take
        if need <= 1e-9:
            break
    return total if need <= 1e-9 else np.inf


def _grade_priority_reference(clean: pd.DataFrame, target_tonnage: float,
                              max_fe_percent: float) -> tuple[float, float]:
    """Reference blend cost-per-ton: grade-priority (cost-agnostic) strategy.

    Tonnage is drawn from the highest-Mn stockpiles first. Used only as a
    transparent baseline for the "Cost Difference vs Reference" figure.
    """
    unit = clean["Mn_Grade"].values
    cap = clean["Available_Tons"].values
    fe = clean["Fe_Grade"].values
    cost = clean["Cost_per_Ton"].values
    order = np.argsort(-unit)
    need = float(target_tonnage)
    ref_cost = 0.0
    ref_fe = 0.0
    for j in order:
        take = min(cap[j], need)
        ref_cost += take * cost[j]
        ref_fe += take * fe[j]
        need -= take
        if need <= 1e-9:
            break
    if need > 1e-9 or (target_tonnage > 0 and ref_fe / target_tonnage > max_fe_percent):
        # Fallback: worst-cost feasible benchmark solved via LP.
        n = len(clean)
        bounds = [(0.0, cap[j]) for j in range(n)]
        result = linprog(
            c=-cost,
            A_eq=np.ones((1, n)), b_eq=[target_tonnage],
            bounds=bounds, method="highs",
        )
        if result.success:
            return float(result.fun * -1.0), target_tonnage
        return float(np.inf), float("nan")
    return ref_cost, ref_fe


# ---------------------------------------------------------------------------
# Optimisation engine
# ---------------------------------------------------------------------------

def calculate_optimal_blend(stockpile_df: pd.DataFrame,
                            target_tonnage: float,
                            target_mn_percent: float,
                            max_fe_percent: float = 20.0,
                            max_cost_per_ton: float | None = None) -> dict:
    """Compute the cost-minimising stockpile blend via linear programming.

    Parameters
    ----------
    stockpile_df : DataFrame with columns
        Stockpile_ID, Available_Tons, Mn_Grade, Fe_Grade, Cost_per_Ton
    target_tonnage : total tonnes required (1,000 - 30,000)
    target_mn_percent : minimum blended Mn grade %
    max_fe_percent : maximum blended Fe grade %
    max_cost_per_ton : optional maximum allowable cost per ton (Rs/t).
        Pass None to disable the budget constraint.

    Returns
    -------
    dict : status, allocations and blended-quality metrics. On failure the
        dict carries success=False and a human-readable message.
    """
    ok, message, clean = _validate_stockpile_frame(stockpile_df)
    if not ok:
        return _error_result("Invalid Input", message)

    try:
        target = float(target_tonnage)
        target_mn = float(target_mn_percent)
        max_fe = float(max_fe_percent)
        max_cost = None if max_cost_per_ton is None else float(max_cost_per_ton)
    except (TypeError, ValueError) as exc:
        return _error_result("Invalid Input",
                             "Target tonnage, Mn and Fe values must be numeric.")
    finite_ok = (np.isfinite(target) and np.isfinite(target_mn)
                 and np.isfinite(max_fe)
                 and (max_cost is None or np.isfinite(max_cost)))
    if not finite_ok:
        return _error_result("Invalid Input",
                             "Target tonnage, Mn and Fe values must be finite numbers.")
    if target <= 0:
        return _error_result("Invalid Input", "Target tonnage must be greater than 0.")
    if not 0 < target_mn <= 100:
        return _error_result(
            "Invalid Input", "Target Mn grade must be between 0 and 100%.")
    if not 0 <= max_fe <= 100:
        return _error_result(
            "Invalid Input", "Maximum Fe grade must be between 0 and 100%.")
    if max_cost is not None and max_cost <= 0:
        return _error_result(
            "Invalid Input", "Maximum cost per ton must be greater than 0.")

    # ------------------------------------------------------------------ checks
    available = clean["Available_Tons"].values
    unit_mn = clean["Mn_Grade"].values
    unit_fe = clean["Fe_Grade"].values
    unit_cost = clean["Cost_per_Ton"].values

    if available.sum() < target - 1e-6:
        return _error_result(
            "Infeasible",
            f"Optimization infeasible: total available stockpile tonnage "
            f"({available.sum():,.0f} t) is less than the requested target "
            f"({target:,.0f} t).",
        )

    max_mn = _max_possible_mn(clean, target)
    if max_mn < target_mn * target - 1e-6:
        return _error_result(
            "Infeasible",
            "Optimization infeasible: available stockpiles cannot satisfy "
            f"the requested Mn target ({target_mn:.1f}% Mn). "
            f"The best achievable blend is {max_mn / target:.1f}% Mn.",
        )

    min_fe = _min_possible_fe(clean, target)
    if min_fe > max_fe * target + 1e-6:
        return _error_result(
            "Infeasible",
            "Optimization infeasible: available stockpiles cannot meet the "
            f"requested Fe limit ({max_fe:.1f}% max Fe). "
            f"The lowest achievable blend is {min_fe / target:.1f}% Fe.",
        )

    # ------------------------------------------------------------------- LP
    n = len(clean)
    objective = unit_cost.tolist()
    A_ub_rows = [-unit_mn, unit_fe]
    b_ub_rows = [-target_mn * target, max_fe * target]
    if max_cost is not None:
        # Budget constraint: blended cost per ton <= max_cost_per_ton.
        A_ub_rows.append(unit_cost)
        b_ub_rows.append(max_cost * target)

    A_eq = np.ones((1, n))
    b_eq = [target]
    bounds = [(0.0, float(cap)) for cap in available]

    try:
        result = linprog(c=objective,
                         A_ub=np.asarray(A_ub_rows, dtype=float),
                         b_ub=np.asarray(b_ub_rows, dtype=float),
                         A_eq=A_eq, b_eq=b_eq,
                         bounds=bounds, method="highs")
    except Exception as exc:  # pragma: no cover - solver level failure
        return _error_result(
            "Solver Error",
            "The optimization solver could not run. "
            f"({exc.__class__.__name__})",
        )

    if result.status == 2:
        return _error_result(
            "Infeasible",
            "Optimization infeasible: the combination of tonnage, Mn, Fe and "
            "budget constraints cannot be satisfied by the available "
            "stockpiles.",
        )
    if result.status != 0:
        return _error_result(
            "Solver Error",
            "The optimization solver did not converge "
            f"(status code {result.status}).",
        )

    # Keep the exact solver values: only zero out floating-point noise below
    # a tiny numerical tolerance. Any meaningful allocation (however small)
    # is preserved so tonnage, Mn, Fe, cost and constraint validation all use
    # the actual solver result. Values are rounded only for display below.
    allocations = np.where(result.x < 1e-8, 0.0, result.x)

    blended_mn = float(np.dot(allocations, unit_mn)) / target
    blended_fe = float(np.dot(allocations, unit_fe)) / target
    total_cost = float(np.dot(allocations, unit_cost))
    cost_per_ton = total_cost / target
    utilization = float(allocations.sum()) / float(available.sum()) * 100.0

    # Cost difference vs grade-priority reference strategy.
    ref_cost, _ = _grade_priority_reference(clean, target, max_fe)
    if np.isfinite(ref_cost):
        savings = max(ref_cost - total_cost, 0.0)
    else:
        savings = 0.0
    savings_per_ton = savings / target if target else 0.0

    df_out = clean[["Stockpile_ID", "Available_Tons", "Mn_Grade",
                    "Fe_Grade", "Cost_per_Ton"]].copy()
    df_out["Allocated_Tons"] = np.round(allocations, 2)
    df_out["Utilization_Pct"] = np.round(
        np.where(df_out["Available_Tons"] > 0,
                 allocations / df_out["Available_Tons"] * 100.0, 0.0), 1)
    df_out["In_Blend"] = np.where(allocations > 0.5, "In Blend", "-")

    return {
        "success": True,
        "status": "Optimal",
        "message": "Optimal blend found successfully.",
        "result_df": df_out,
        "allocations": allocations,
        "total_tonnage": target,
        "blended_mn": blended_mn,
        "blended_fe": blended_fe,
        "total_cost": total_cost,
        "cost_per_ton": cost_per_ton,
        "utilization_pct": utilization,
        "savings": savings,
        "savings_per_ton": savings_per_ton,
    }


def verify_blend(result: dict,
                 target_tonnage: float,
                 target_mn_percent: float,
                 max_fe_percent: float,
                 max_cost_per_ton: float | None = None,
                 tol_tons: float = 0.5, tol_grade: float = 0.05,
                 tol_cost: float = 0.01) -> dict:
    """Independently re-check an optimisation result against every constraint.

    Returns
    -------
    dict with keys:
        checks : list of dicts (constraint, required, achieved, unit, status)
        all_pass : bool
        reason : str when the result cannot be verified

    This does not trust solver output - each constraint is recomputed from
    the returned allocations and stockpile data.
    """
    checks: list[dict] = []

    if not isinstance(result, dict) or not result.get("success"):
        return {"checks": [], "all_pass": False,
                "reason": "Optimization did not succeed; no blend to verify."}

    alloc = result.get("allocations")
    df_out = result.get("result_df")
    if alloc is None or df_out is None:
        return {"checks": [], "all_pass": False,
                "reason": "Result contains no allocations to verify."}

    try:
        target = float(target_tonnage)
        target_mn = float(target_mn_percent)
        max_fe = float(max_fe_percent)
        max_cost = None if max_cost_per_ton is None else float(max_cost_per_ton)
    except (TypeError, ValueError):
        return {"checks": [], "all_pass": False,
                "reason": "Verification targets are not numeric."}

    # 1 - Total tonnage
    achieved_tons = float(np.sum(alloc))
    checks.append({
        "constraint": "Total tonnage",
        "required": f"= {target:,.0f} t",
        "achieved": f"{achieved_tons:,.0f} t",
        "unit": "t",
        "status": "PASS" if abs(achieved_tons - target) <= tol_tons else "FAIL",
    })

    # 2 - Minimum blended Mn
    unit_mn = df_out["Mn_Grade"].to_numpy(dtype=float)
    blended_mn = float(np.dot(alloc, unit_mn)) / achieved_tons if achieved_tons else 0.0
    checks.append({
        "constraint": "Minimum blended Mn",
        "required": f">= {target_mn:.1f}%",
        "achieved": f"{blended_mn:.2f}%",
        "unit": "%",
        "status": "PASS" if blended_mn >= target_mn - tol_grade else "FAIL",
    })

    # 3 - Maximum blended Fe
    unit_fe = df_out["Fe_Grade"].to_numpy(dtype=float)
    blended_fe = float(np.dot(alloc, unit_fe)) / achieved_tons if achieved_tons else 0.0
    checks.append({
        "constraint": "Maximum blended Fe",
        "required": f"<= {max_fe:.1f}%",
        "achieved": f"{blended_fe:.2f}%",
        "unit": "%",
        "status": "PASS" if blended_fe <= max_fe + tol_grade else "FAIL",
    })

    # 4 - Stockpile availability (per stockpile, recomputed)
    avail = df_out["Available_Tons"].to_numpy(dtype=float)
    within_avail = bool(np.all(alloc <= avail + tol_tons))
    checks.append({
        "constraint": "Stockpile availability",
        "required": "allocated <= available",
        "achieved": ("all stockpiles within limits" if within_avail
                     else "an allocation exceeds its stockpile"),
        "unit": "t",
        "status": "PASS" if within_avail else "FAIL",
    })

    # 5 - Non-negative allocations
    non_neg = bool(np.all(alloc >= -1e-6))
    checks.append({
        "constraint": "Non-negative allocations",
        "required": "allocations >= 0",
        "achieved": "all allocations >= 0" if non_neg else "negative value found",
        "unit": "t",
        "status": "PASS" if non_neg else "FAIL",
    })

    # 6 - Optional cost-per-ton budget
    if max_cost is not None:
        unit_cost = df_out["Cost_per_Ton"].to_numpy(dtype=float)
        blended_cost = (float(np.dot(alloc, unit_cost)) / achieved_tons
                        if achieved_tons else 0.0)
        checks.append({
            "constraint": "Maximum cost per ton",
            "required": f"<= Rs {max_cost:,.2f}",
            "achieved": f"Rs {blended_cost:,.2f}",
            "unit": "Rs/t",
            "status": "PASS" if blended_cost <= max_cost + tol_cost else "FAIL",
        })

    all_pass = all(c["status"] == "PASS" for c in checks)
    return {"checks": checks, "all_pass": all_pass, "reason": ""}


if __name__ == "__main__":
    data = load_sample_stockpiles()
    out = calculate_optimal_blend(data, 10000, 42.0, 20.0)
    print("Status:", out["status"])
    print("Blended Mn:", round(out["blended_mn"], 2), "%")
    print("Blended Fe:", round(out["blended_fe"], 2), "%")
    print("Total cost: Rs", round(out["total_cost"], 0))
    print("Cost / ton: Rs", round(out["cost_per_ton"], 2))
    print(out["result_df"])