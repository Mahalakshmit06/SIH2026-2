# MnBrain - Geospatial AI & Grade Blending Intelligence

**3D Subsurface Modeling + Prescriptive Grade Blending**

A standalone, fully-offline decision-support application for the
SIH 2026 / Ministry of Steel **Problem Statement ID: 26009** (MnBrain).
It runs on the bundled **Demonstration Dataset** (synthetic) or on
user-uploaded CSV / Excel geological and stockpile files.

---

## Project Overview

MnBrain delivers two integrated modules wrapped in a single Streamlit app:

| Module | Purpose | Technology |
| ------ | ------- | ---------- |
| **1. 3D Subsurface Grade Modeling (Geospatial AI)** | Estimate Mn / Fe grades across a 3D block model using an Ordinary-Kriging interpolator with a spherical variogram. | Python, NumPy, Pandas, SciPy, Plotly |
| **2. Prescriptive Grade Blending Optimization (LP)** | Compute the cost-minimising stockpile blend that meets tonnage, Mn, Fe and (optionally) cost-per-ton constraints. | Python, Pandas, SciPy `linprog`, Plotly |

Around these, a data-ingestion layer validates and normalises user data, a
constraint-verifier re-checks solver output (PASS/FAIL), and a
dependency-free report generator exports decision reports (TXT / CSV / PDF).

---

## Processing Pipeline

```
Data Ingestion ──► Validation & QA ──► Geological Model
      │                                          │
      │  demo dataset or uploaded CSV/Excel      │  spherical variogram
      ▼                                          ▼
Spherical Variogram                         Ordinary Kriging
      │                                          │
      ▼                                          ▼
3D Block Model ◄────────────────────  Grade + Uncertainty
      │
      ▼
UNFC 111 Screening ──► Stockpile Data ──► LP Optimization
                                                  │
                                                  ▼
                                     Constraint Validation (PASS/FAIL)
                                                  │
                                                  ▼
                              Decision Support & Report (TXT/CSV/PDF)
```

---

## Features

**Data ingestion (`data_loader.py`)**
- Loads geological samples and stockpile data from the **Demonstration
  Dataset** or from user-uploaded **CSV / Excel** files.
- Normalises accepted column aliases (`Easting → X`, `Northing → Y`,
  `Mn% / Mn_pct → Mn_Grade`, `Tonnes → Available_Tons`, etc.).
- Validates before modelling: required columns, numeric values, grade
  ranges (0-100%), non-negative depth/tonnage, positive cost, duplicate
  rows/IDs, non-finite values.
- Renders a data-quality summary (total / valid / missing / duplicates,
  Mn / Fe / depth / cost ranges). Invalid data is **never** passed to the
  modelling engines.

**Module 1 - Geospatial AI (`geospatial_model.py`)**
- Three selectable pits (Dongri Buzurg, Balaghat, Generic Pit) with
  distinct deterministic spatial patterns.
- Depth control (10-100 m) that rebuilds the block model.
- Ordinary Kriging with a **spherical variogram model**. For uploaded
  data the nugget / partial sill / range / z-anisotropy are explicit
  **model parameters** (there is no automatic variogram fitting).
- Per-block kriging variance and a **UNFC 111 / UNVERIFIED** screening
  driven by a confidence threshold.
- The shared kriging core (`krige_samples`) serves both the demonstration
  path and validated uploads (`krige_uploaded_samples`), so the algorithm
  is identical whatever the source.
- Interactive Plotly 3D block model, grade distributions, Mn-Fe scatter,
  depth profile and screening charts.

**Module 2 - Grade Blending Optimization (`optimization_engine.py`)**
- Cost-minimising linear program solved with `scipy.optimize.linprog`
  (HiGHS). Constraints: tonnage (equality), minimum Mn, maximum Fe,
  per-stockpile availability, and an **optional maximum cost per ton**.
- **Constraint validation (`verify_blend`)** independently re-computes every
  constraint from the allocations and reports PASS/FAIL - the solver output
  is not trusted blindly.
- Infeasible / invalid input handling with clear, crash-free messages.

**Decision support (`report.py`)**
- Generates a decision report in **TXT**, **CSV** (allocation plan, scenario
  summary, constraint validation) and **PDF** - the PDF writer uses only the
  Python standard library (no reportlab / fpdf / LaTeX required).
- Each report carries inputs, allocations, blended metrics, the PASS/FAIL
  verdict, and a timestamp. No signatures are faked or implied.

**Session audit trail**
- Every model run, optimization and report export is recorded in an
  in-memory audit trail (timestamp, stage, summary) shown in the app.

---

## Architecture / Project layout

```
MnBrain_SIH_Submission/
├── app.py                  # Streamlit application (3 pages)
├── data_loader.py          # Ingestion + validation (geological & stockpile)
├── geospatial_model.py     # Module 1 - OK kriging, variogram, block model
├── optimization_engine.py  # Module 2 - LP blend optimizer + verify_blend
├── report.py               # Decision report TXT / CSV / PDF (std-lib only)
├── sample_stockpiles.csv   # Demonstration stockpile dataset
├── sample_geological_data.csv  # Synthetic demonstration geological upload sample (169 boreholes)
├── demo.py                 # CLI full-pipeline demo
├── test_modules.py         # 84 unit tests across 9 suites
├── requirements.txt
├── README.md
├── .streamlit/config.toml  # Industrial Slate theme
└── data/
    └── README.md           # notes on the data folder
```

---

## Installation

**Requirements:** Python 3.10+ (tested on 3.13).

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

`requirements.txt` pins the exact, tested versions (streamlit, pandas, numpy,
scipy, plotly, openpyxl) so the environment is reproducible. The app works
**offline** - no internet, APIs, databases, GIS servers or commercial solvers.
`openpyxl` is included solely to read uploaded Excel files.

---

## Run Commands

```bash
# Run the Streamlit application
streamlit run app.py            # (python -m streamlit run app.py if not on PATH)

# Run the test suite (84 tests)
python test_modules.py

# Run the CLI demo (full pipeline, no browser required)
python demo.py
```

---

## Expected input schemas

**Geological samples** (required columns; aliases accepted):

| Column | Alias examples |
| ------ | -------------- |
| `Borehole_ID` | `Hole_ID`, `BHID`, `Sample_ID` |
| `X` | `Easting`, `East`, `X_coord` |
| `Y` | `Northing`, `North`, `Y_coord` |
| `Depth` | `Z`, `Depth_m` |
| `Mn_Grade` | `Mn`, `Mn%`, `Mn_pct` |
| `Fe_Grade` | `Fe`, `Fe%`, `Fe_pct` |

`sample_geological_data.csv` is a small (40-row) **synthetic demonstration**
table matching this schema - upload it in the app to try the uploaded-data
workflow instantly. As with the built-in dataset, it is generated
deterministically and is **not** real MOIL data.

**Stockpiles**:

| Column | Alias examples |
| ------ | -------------- |
| `Stockpile_ID` | `Stockpile`, `Pile_ID` |
| `Available_Tons` | `Available_Tonnes`, `Tons`, `Tonnes` |
| `Mn_Grade`, `Fe_Grade` | `Mn`/`Mn%`, `Fe`/`Fe%` |
| `Cost_per_Ton` | `Cost`, `Cost_per_Tonne` |

---

## Module 1 - Explanation (Geospatial AI)

**`geospatial_model.py`** exposes:

```python
get_demonstration_samples(pit_name)                       # sample table
krige_samples(sx, sy, sz, mn, fe, depth_m, cutoff,        # shared core
              nugget, partial_sill, range_m, anisotropy_z,
              extent, ..., code, clip_mn, clip_fe) -> DataFrame
krige_uploaded_samples(sample_df, depth_m, cutoff, ...)   # validated uploads
get_kriged_block_model(pit_name, depth_m, cutoff)         # demonstration
list_pits() -> tuple[str, ...]
get_variogram_parameters(pit_name) -> dict
```

### Workflow
1. Samples come from the Demonstration Dataset or a validated upload.
2. A **spherical variogram** models spatial continuity (explicit model
   parameters for uploads):
   - γ(h) = C0 + C·[1.5(h/a) − 0.5(h/a)³]  for 0 < h ≤ a
   - γ(h) = C0 + C                          for h > a
3. **Ordinary Kriging** estimates the grade at every block centre:
   - Ẑ(x₀) = Σ λᵢ·Z(xᵢ)  with  Σ λᵢ = 1
4. Kriging variance is reported relative to the sill and converted to a
   confidence score.
5. Blocks with relative variance below the threshold are screened as
   **UNFC 111**, otherwise **UNVERIFIED**.

### Output schema

| Column | Description |
| ------ | ----------- |
| `Block_ID` | unique block identifier |
| `X`, `Y`, `Z` | block centre coordinates / depth (m) |
| `Mn_Grade`, `Fe_Grade` | kriged manganese / iron grade (%) |
| `Kriging_Variance` | relative prediction variance (%) |
| `Confidence` | 0-100 confidence score |
| `UNFC_Code` | `111` (screening) or `UNVERIFIED` |

> ℹ️ UNFC code `111` here is a **screening / status result** from the
> kriging-variance criterion. It is **not** a regulatory resource
> classification. Uploaded grids are derived from the observed sample extent
> and capped (`MAX_GRID_CELLS`) so the dashboard stays interactive.

---

## Module 2 - Explanation (Grade Blending Optimization)

**`optimization_engine.py`** exposes:

```python
calculate_optimal_blend(stockpile_df, target_tonnage, target_mn_percent,
                        max_fe_percent=20.0,
                        max_cost_per_ton=None) -> dict
verify_blend(result, target_tonnage, target_mn_percent, max_fe_percent,
             max_cost_per_ton=None) -> {"checks": [...], "all_pass": bool}
load_sample_stockpiles() -> pd.DataFrame
```

### Mathematical Formulation (Linear Program)

```
Minimize        Σ ( c_j · w_j )                          cost per ton × tons

subject to      Σ w_j             = target_tonnage        (equality)
                Σ (Mn_j · w_j)    ≥ target_mn · tons      (min Mn)
                Σ (Fe_j · w_j)    ≤ max_fe   · tons       (max Fe)
                Σ (c_j · w_j)     ≤ cost_max · tons       (optional budget)
                0 ≤ w_j           ≤ Available_Tons_j      (availability)
```

### Result keys

`status`, `message`, `result_df`, `allocations`, `total_tonnage`,
`blended_mn`, `blended_fe`, `total_cost`, `cost_per_ton`,
`utilization_pct`, `savings`, `savings_per_ton`.

The engine never raises on bad input: infeasible or invalid requests return
a result dict with `success=False` and a human-readable message.

`verify_blend` re-checks total tonnage, blended Mn, blended Fe, stockpile
availability, non-negative allocations and (when configured) the blended
cost per ton, returning a `PASS`/`FAIL` verdict independent of the solver.

> ℹ️ `savings` is displayed as **Cost Difference vs Reference** (compared
> with a grade-priority reference blend) and is a computed comparison, not
> audited MOIL savings.

---

## Decision Support Report

`report.py` builds a report from a successful blend, the scenario inputs,
block-model summary and the `verify_blend` verdict:

- **`.txt`** - printable report.
- **`.csv`** - `allocation_plan.csv`, `scenario_summary.csv`,
  `constraint_validation.csv`.
- **`.pdf`** - A4 report built with a minimal standard-library PDF writer
  (valid xref table, single Helvetica font, header/footer, pagination).

---

## Testing

```bash
python test_modules.py
```

84 tests across nine suites:

- **Geospatial model** - valid model, schema, NaN-free, reproducibility,
  differing pits, grade ranges, screening effect, invalid depth /
  confidence / pit.
- **Optimization engine** - columns, success, all constraints, infeasible
  cases, invalid inputs, crash-free behaviour.
- **Data loader** - demo validation, alias normalisation, missing columns,
  negative depth, out-of-range grades, duplicates, non-finite values,
  CSV/Excel parsing.
- **Uploaded kriging** - standard schema, reproducibility, alignment with
  the demo grade field, term rejection for invalid inputs, degenerate
  extents.
- **Constraint validation** - feasible PASS, detected Mn / tonnage /
  budget violations, infeasible results not verifiable, budget handling,
  tiny allocations preserved, display/internal allocation consistency.
- **Upload workflow** - `sample_geological_data.csv` loads, schema detected,
  validates, kriges to valid NaN-free blocks within the grid cap.
- **Grid cap regression** - a 3200 m x 100 m strip must never exceed
  `MAX_GRID_CELLS` (32) cells on either axis while keeping full coverage.
- **Sample model quality** - the bundled sample is substantial (100+
  boreholes, 5+ depth levels), kriges to a meaningful 3D block model with
  valid grades and no NaNs, and its screening output contains both UNFC 111
  and UNVERIFIED blocks.
- **KPI state consistency** - after running an uploaded geological dataset
  every KPI card reflects the uploaded model (never stale demonstration
  numbers), and switching back restores the demonstration values.

---

## Demo

```bash
python demo.py
```

Walks the full pipeline (ingestion → validation → kriging → UNFC screening →
optimization → verify_blend → report export) headlessly and writes a sample
report package to `./demo_reports`.

---

## Limitations

- **Synthetic dataset:** bundled grades and costs resemble realistic ranges
  but are **not** real MOIL measurements.
- **OK-style estimator:** a working geostatistical demonstration, not a
  certified resource estimator; no regulatory UNFC classification is claimed.
- **Uploaded data quality** is the user's responsibility (units are assumed
  to be metres, tonnes, % and Rs/t).
- **Audit trail** is in-memory only and is not persisted.
- **Performance:** kriging recomputes on demand; very large extents are
  capped to keep the app interactive.

---

## Future Integration Possibilities

- Alternate variogram models (exponential / gaussian) and automatic
  variogram fitting with cross-validation.
- Persist audit trails and model runs to `data/` for review.
- Connect to real assay databases (subject to approvals).
- Add cut-off-grade analysis and schedule-aware blending.

---

*Prepared for SIH 2026 • Ministry of Steel • PS ID 26009 • MnBrain.*