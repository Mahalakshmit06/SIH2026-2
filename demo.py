"""
MnBrain - Command-line Demonstration Script
=============================================

Walks the full decision-support pipeline without Streamlit:

    Data ingestion & validation -> Geological model -> Ordinary Kriging
    -> 3D block model -> UNFC screening -> Stockpile validation
    -> LP optimization -> Constraint validation -> Decision report export

Run with:  python demo.py

Everything uses the bundled Demonstration Dataset (synthetic). A small
decision report package (TXT / CSV / PDF) is written to ./demo_reports.
"""

from __future__ import annotations

import io
import os

from data_loader import (load_demonstration_geological_samples,
                         load_demonstration_stockpiles, parse_upload,
                         validate_geological_data, validate_stockpile_data)
from geospatial_model import MAX_GRID_CELLS, get_demonstration_samples, \
    get_kriged_block_model, krige_uploaded_samples
from optimization_engine import (calculate_optimal_blend, verify_blend,
                                 load_sample_stockpiles)
from report import export_decision_report

PIT = "Balaghat"
DEPTH = 50
CONFIDENCE = 3.0

TARGET_TONNAGE = 10000
TARGET_MN = 42.0
MAX_FE = 20.0
MAX_COST_PER_TON = 1150.0

REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "demo_reports")


def main() -> None:
    print("=" * 66)
    print("MnBrain - Geospatial AI & Grade Blending Intelligence")
    print("Full decision-support pipeline (Demonstration Dataset)")
    print("=" * 66)

    # ------------------------------------------------ 1. Data ingestion
    print("\n## 1. DATA INGESTION & VALIDATION")
    geo = load_demonstration_geological_samples(PIT)
    vg = validate_geological_data(geo)
    print(f"Geological rows       : {vg['summary']['n_total']:,} loaded, "
          f"{vg['summary']['n_valid']:,} valid")
    print(f"QC  - Mn range        : {vg['summary']['mn_min']:.1f}% - "
          f"{vg['summary']['mn_max']:.1f}%")
    print(f"QC  - Fe range        : {vg['summary']['fe_min']:.1f}% - "
          f"{vg['summary']['fe_max']:.1f}%")
    print(f"QC  - Depth range     : {vg['summary']['depth_min']:.0f} m - "
          f"{vg['summary']['depth_max']:.0f} m")
    print(f"QC  - missing/dupes   : {vg['summary']['n_missing_values']} / "
          f"{vg['summary']['n_duplicates']}")
    print(f"Validation OK         : {vg['ok']}")

    stock = load_demonstration_stockpiles()
    vs = validate_stockpile_data(stock)
    print(f"Stockpile rows        : {vs['summary']['n_valid']} valid, "
          f"total {vs['summary']['total_tonnage']:,.0f} t, "
          f"cost {vs['summary']['cost_min']:.0f}-"
          f"{vs['summary']['cost_max']:.0f} Rs/t")
    print(f"Validation OK         : {vs['ok']}")

    # --------------------------------------- 2. Geological model (kriging)
    print("\n## 2. GEOLOGICAL MODEL + ORDINARY KRIGING")
    print("(via Demonstration Dataset path)")
    model = get_kriged_block_model(PIT, DEPTH, CONFIDENCE)
    screened = int((model["UNFC_Code"] == "111").sum())
    print(f"Pit / Depth / Thresh  : {PIT} / {DEPTH} m / {CONFIDENCE}%")
    print(f"3D block model        : {len(model):,} blocks")
    print(f"Average Mn / Fe       : {model['Mn_Grade'].mean():.2f}% / "
          f"{model['Fe_Grade'].mean():.2f}%")
    print(f"Avg kriging variance  : {model['Kriging_Variance'].mean():.2f}%")
    print(f"UNFC 111 (screening)  : {screened:,} "
          f"({screened / len(model) * 100:.1f}%)")

    print("\n(identical samples passed through the uploaded-data path)")
    up = krige_uploaded_samples(get_demonstration_samples(PIT), DEPTH,
                                CONFIDENCE, nugget=0.3, partial_sill=32.0,
                                range_m=1000.0, anisotropy_z=2.0)
    up_scr = int((up["UNFC_Code"] == "111").sum())
    print(f"Uploaded-path model   : {len(up):,} blocks, avg Mn "
          f"{up['Mn_Grade'].mean():.2f}%, UNFC 111 {up_scr:,}")

    print("\n(sample upload: sample_geological_data.csv)")
    sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "sample_geological_data.csv")
    with open(sample_path, "rb") as fh:
        sample = parse_upload(io.BytesIO(fh.read()), "sample_geological_data.csv")
    vsample = validate_geological_data(sample)
    print(f"CSV rows loaded       : {vsample['summary']['n_total']} validated "
          f"{'OK' if vsample['ok'] else 'FAIL'}")
    up_csv = krige_uploaded_samples(vsample["cleaned"], DEPTH, CONFIDENCE)
    grid_x = up_csv["X"].nunique()
    grid_y = up_csv["Y"].nunique()
    print(f"Uploaded block model  : {len(up_csv):,} blocks, avg Mn "
          f"{up_csv['Mn_Grade'].mean():.2f}%, avg Fe "
          f"{up_csv['Fe_Grade'].mean():.2f}%")
    print(f"Horizontal grid       : {grid_x} x {grid_y} cells "
          f"(cap {MAX_GRID_CELLS} per axis), no NaN: "
          f"{bool(not up_csv.isna().any().any())}")

    # ------------------------------------------------ 3. LP optimization
    print("\n## 3. LP OPTIMIZATION")
    out = calculate_optimal_blend(stock, TARGET_TONNAGE, TARGET_MN, MAX_FE)
    print(f"Status                : {out['status']}")
    print(f"Target tonnage / Mn   : {TARGET_TONNAGE:,.0f} t / "
          f">= {TARGET_MN:.1f}%")
    print(f"Max Fe / max cost/t   : {MAX_FE:.1f}% / Rs {MAX_COST_PER_TON:,.0f}")
    print(f"Allocated tonnage     : {out['total_tonnage']:,.0f} t")
    print(f"Blended Mn / Fe       : {out['blended_mn']:.2f}% / "
          f"{out['blended_fe']:.2f}%")
    print(f"Total cost / cost-per-ton : Rs {out['total_cost']:,.0f} / "
          f"Rs {out['cost_per_ton']:,.2f}")
    print(f"Cost diff vs reference: Rs {out['savings']:,.0f}")
    print("\nAllocation detail:")
    cols = ["Stockpile_ID", "Available_Tons", "Allocated_Tons",
            "Mn_Grade", "Fe_Grade", "Cost_per_Ton", "Utilization_Pct"]
    print(out["result_df"][cols].to_string(index=False))

    budgeted = calculate_optimal_blend(stock, TARGET_TONNAGE, TARGET_MN,
                                       MAX_FE, max_cost_per_ton=MAX_COST_PER_TON)
    print("\nBudget-constrained run (max Rs 1,150/t):",
          budgeted["status"],
          f"- cost/t Rs {budgeted['cost_per_ton']:,.2f}")

    # ------------------------------------------ 4. Constraint validation
    print("\n## 4. CONSTRAINT VALIDATION (verify_blend)")
    verification = verify_blend(out, TARGET_TONNAGE, TARGET_MN, MAX_FE)
    for c in verification["checks"]:
        print(f"  [{c['status']}] {c['constraint']:<26} required {c['required']}"
              f"  achieved {c['achieved']}")
    print("  VERDICT:", "ALL CONSTRAINTS SATISFIED"
          if verification["all_pass"] else "CONSTRAINT VIOLATION")
    budget_verify = verify_blend(budgeted, TARGET_TONNAGE, TARGET_MN, MAX_FE,
                                 max_cost_per_ton=MAX_COST_PER_TON)
    print("  Budget check:", "PASS"
          if budget_verify["all_pass"] else "FAIL")

    # ------------------------------------------------- 5. Decision report
    print("\n## 5. DECISION SUPPORT REPORT")
    ctx = {
        "data_source": "Demonstration Dataset",
        "scenario": {"target_tonnage": TARGET_TONNAGE, "target_mn": TARGET_MN,
                     "max_fe": MAX_FE, "max_cost_per_ton": None},
        "block_model": {"n_blocks": len(model),
                        "avg_mn": float(model["Mn_Grade"].mean()),
                        "avg_fe": float(model["Fe_Grade"].mean()),
                        "unfc_111": screened},
        "verification": verification,
    }
    written = export_decision_report(out, REPORT_DIR, ctx)
    for path in written:
        print(f"  wrote {path} ({os.path.getsize(path):,} bytes)")

    print("\nNOTE: All values above are from the synthetic Demonstration "
          "Dataset.")
    print("UNFC 111 entries are screening statuses from the kriging-variance")
    print("criterion, not regulatory resource certifications.")


if __name__ == "__main__":
    main()