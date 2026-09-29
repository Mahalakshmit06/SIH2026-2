"""
MnBrain - Geospatial AI & Grade Blending Intelligence
=======================================================

Standalone Streamlit application for the SIH 2026 MnBrain project
(Ministry of Steel, Problem Statement ID: 26009).

Module 1 : 3D Subsurface Grade Modeling (Ordinary Kriging + spherical variogram)
Module 2 : Prescriptive Ore Grade Blending Optimization (Linear Programming)

Data can come from the bundled Demonstration Dataset or from user-uploaded
CSV/Excel files. Uploaded data is validated before use and is never kriged
or optimised unless it passes its data-quality checks. All bundled data is
synthetic - no real MOIL geological or financial data is used or claimed.
Run with:  streamlit run app.py
"""

from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="MnBrain - Geospatial AI & Grade Blending Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

from data_loader import (  # noqa: E402
    GEOLOGICAL_SOURCE_OPTIONS,
    STOCKPILE_SOURCE_OPTIONS,
    DataLoadError,
    load_demonstration_geological_samples,
    load_demonstration_stockpiles,
    parse_upload,
    validate_geological_data,
    validate_stockpile_data,
)
from geospatial_model import (  # noqa: E402
    get_kriged_block_model,
    get_variogram_parameters,
    krige_uploaded_samples,
    list_pits,
)
from optimization_engine import (  # noqa: E402
    calculate_optimal_blend,
    load_sample_stockpiles,
    verify_blend,
)
from report import export_decision_report  # noqa: E402

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

BG = "#0F172A"
PANEL = "#1E293B"
BORDER = "#334155"
PRIMARY = "#06B6D4"
SECONDARY = "#10B981"
WARNING = "#F59E0B"
DANGER = "#EF4444"
TEXT = "#F8FAFC"
TEXT_MUTED = "#94A3B8"

DEFAULT_PIT = "Balaghat"
DEFAULT_DEPTH = 50
DEFAULT_CUTOFF = 3.0
DEFAULT_TARGET_TONNAGE = 10000
DEFAULT_TARGET_MN = 42.0
DEFAULT_MAX_FE = 20.0

CSS = f"""
<style>
:root {{
  --bg: {BG}; --panel: {PANEL}; --border: {BORDER};
  --primary: {PRIMARY}; --secondary: {SECONDARY};
  --warning: {WARNING}; --danger: {DANGER};
  --text: {TEXT}; --muted: {TEXT_MUTED};
}}
html, body, [data-testid="stAppViewContainer"] {{
  background-color: {BG};
}}
[data-testid="stSidebar"] {{
  background-color: {PANEL};
  border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] * {{ color: {TEXT_MUTED}; }}
h1, h2, h3, h4 {{ color: {TEXT}; }}
p, label, .stMarkdown {{ color: {TEXT_MUTED}; }}

.mb-brand {{
  padding: 0.4rem 0 0.6rem 0;
}}
.mb-brand .title {{
  font-size: 2.1rem; font-weight: 800; letter-spacing: 1px;
  color: {TEXT}; line-height: 1.1;
}}
.mb-brand .title span {{ color: {PRIMARY}; }}
.mb-brand .tagline {{
  color: {TEXT_MUTED}; font-size: 0.95rem; margin-top: 0.15rem;
}}
.badge {{
  display: inline-block; background: rgba(245,158,11,0.12);
  border: 1px solid rgba(245,158,11,0.35); color: {WARNING};
  border-radius: 4px; padding: 0.15rem 0.6rem;
  font-size: 0.75rem; letter-spacing: 0.5px; margin-top: 0.45rem;
}}
.badge.synth {{
  background: rgba(16,185,129,0.10); border-color: rgba(16,185,129,0.35);
  color: {SECONDARY};
}}
.badge.upload {{
  background: rgba(129,140,248,0.12); border-color: rgba(129,140,248,0.35);
  color: #818CF8;
}}

.kpi-card {{
  background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px;
  padding: 0.85rem 1rem; height: 100%; min-width: 150px;
  border-top: 3px solid var(--kpi-color, {PRIMARY});
}}
.kpi-label {{
  color: {TEXT_MUTED}; font-size: 0.72rem; text-transform: uppercase;
  letter-spacing: 0.8px; margin-bottom: 0.3rem;
}}
.kpi-value {{
  color: var(--kpi-color, {PRIMARY}); font-size: 1.3rem;
  font-weight: 700; line-height: 1.15; white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis;
}}
.kpi-sub {{ color: {TEXT_MUTED}; font-size: 0.78rem; margin-top: 0.2rem; }}

.panel {{
  background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px;
  padding: 1.1rem 1.2rem; margin-bottom: 1rem;
}}
.panel h3 {{ color: {TEXT}; font-size: 1.05rem; margin: 0 0 0.8rem 0; }}

.section-title {{
  border-left: 3px solid {PRIMARY}; padding-left: 0.7rem;
  color: {TEXT}; font-size: 1.15rem; font-weight: 600;
  margin: 1.4rem 0 0.8rem 0;
}}
.note {{
  background: rgba(6,182,212,0.08); border: 1px solid rgba(6,182,212,0.25);
  border-radius: 6px; padding: 0.55rem 0.8rem; color: {TEXT_MUTED};
  font-size: 0.85rem;
}}
.warn {{
  background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.3);
  border-radius: 6px; padding: 0.55rem 0.8rem; color: {WARNING};
  font-size: 0.85rem;
}}
.err {{
  background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.35);
  border-radius: 6px; padding: 0.55rem 0.8rem; color: {DANGER};
  font-size: 0.85rem;
}}
.qc-chip {{
  display: inline-block; background: {PANEL}; border: 1px solid {BORDER};
  border-radius: 6px; padding: 0.3rem 0.65rem; margin: 0.15rem 0.25rem;
  font-size: 0.78rem; color: {TEXT_MUTED};
}}
.qc-chip b {{ color: {TEXT}; }}
.badge-pass {{
  display: inline-block; background: rgba(16,185,129,0.10);
  border: 1px solid rgba(16,185,129,0.45); color: {SECONDARY};
  border-radius: 5px; padding: 0.1rem 0.55rem; font-size: 0.72rem;
  font-weight: 700; letter-spacing: 0.5px;
}}
.badge-fail {{
  display: inline-block; background: rgba(239,68,68,0.10);
  border: 1px solid rgba(239,68,68,0.45); color: {DANGER};
  border-radius: 5px; padding: 0.1rem 0.55rem; font-size: 0.72rem;
  font-weight: 700; letter-spacing: 0.5px;
}}

.flow-box {{
  background: {PANEL}; border: 1px solid {BORDER}; border-radius: 8px;
  padding: 0.55rem 1rem; text-align: center; color: {TEXT};
  font-weight: 600; font-size: 0.9rem; margin: 0 auto;
  width: min(340px, 92%);
}}
.flow-arrow {{ text-align: center; color: {PRIMARY}; margin: 0.15rem 0;
  font-size: 1.1rem; }}

.stButton > button,
[data-testid="stButton"] > button,
[data-testid="stBaseButton-primary"] button,
[data-testid="stBaseButton-primaryFormSubmit"] button,
[data-testid="stDownloadButton"] button {{
  background: {PRIMARY}; color: #062028; font-weight: 800;
  font-size: 0.95rem; letter-spacing: 0.3px;
  border: 1px solid {PRIMARY}; border-radius: 6px;
}}
.stButton > button:hover,
[data-testid="stButton"] > button:hover,
[data-testid="stBaseButton-primary"] button:hover,
[data-testid="stBaseButton-primaryFormSubmit"] button:hover,
[data-testid="stDownloadButton"] button:hover {{
  background: #22d3ee; color: #062028; border-color: #22d3ee;
}}
.stButton p,
.stButton span,
[data-testid="stButton"] p,
[data-testid="stDownloadButton"] p,
[data-testid="stDownloadButton"] span,
[data-testid="stBaseButton-primary"] p,
[data-testid="stBaseButton-primaryFormSubmit"] p {{
  color: #062028;
}}
[data-testid="stSlider"] [data-testid="stThumbValue"] {{ color: {PRIMARY}; }}
[data-baseweb="select"] > div {{ background: {PANEL}; border-color: {BORDER}; }}
.stExpander {{ border: 1px solid {BORDER}; background: {PANEL}; border-radius: 8px; }}
div[data-testid="stDataFrame"] {{ border-radius: 8px; overflow: hidden; }}
hr {{ border-color: {BORDER}; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def brand_header(status_text: str = "Demonstration Dataset") -> None:
    badge_cls = "upload" if status_text not in (
        "Demonstration Dataset", "Synthetic Dataset") else "synth"
    st.markdown(
        f"""
        <div class="mb-brand">
          <div class="title">Mn<span>Brain</span></div>
          <div class="tagline">Mining Intelligence &amp; Decision Support</div>
          <span class="badge {badge_cls}">Data Source: {status_text}</span>
          <span class="badge synth">3D Subsurface Modeling</span>
          <span class="badge synth">Prescriptive Grade Blending</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, color: str = PRIMARY,
             sub: str | None = None) -> str:
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="kpi-card" style="--kpi-color:{color}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>{sub_html}</div>'
    )


def render_kpis(items: list[tuple[str, str, str, str | None]]) -> None:
    """items = [(label, value, color, sub)]."""
    cols = st.columns(len(items))
    for col, (label, value, color, sub) in zip(cols, items):
        col.markdown(kpi_card(label, value, color, sub), unsafe_allow_html=True)


def style_fig(fig, height: int = 420) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.0)",
        font=dict(color=TEXT_MUTED, family="'Segoe UI', sans-serif", size=13),
        height=height,
        margin=dict(l=20, r=20, t=46, b=20),
        title_font=dict(color=TEXT, size=15),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5, font=dict(size=12)),
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER,
                     tickfont=dict(color=TEXT_MUTED))
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER,
                     tickfont=dict(color=TEXT_MUTED))
    return fig


def scene_style(fig: go.Figure) -> go.Figure:
    fig.update_scenes(
        bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER,
                   title=dict(font=dict(color=TEXT_MUTED))),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER,
                   title=dict(font=dict(color=TEXT_MUTED))),
        zaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER,
                   title=dict(font=dict(color=TEXT_MUTED))),
    )
    return fig


def render_qc_chips(summary: dict, kind: str) -> None:
    """Render data-quality summary chips in a single HTML line."""
    if kind == "geo":
        def rng(lo, hi):
            return "n/a" if lo is None else f"{lo:.1f}-{hi:.1f}%"
        chips = [
            ("Total rows", f'{summary.get("n_total", 0):,}'),
            ("Valid rows", f'{summary.get("n_valid", 0):,}'),
            ("Missing values", f'{summary.get("n_missing_values", 0):,}'),
            ("Duplicates", f'{summary.get("n_duplicates", 0):,}'),
            ("Mn range", rng(summary.get("mn_min"), summary.get("mn_max"))),
            ("Fe range", rng(summary.get("fe_min"), summary.get("fe_max"))),
            ("Depth range",
             (f"{summary['depth_min']:.1f}-{summary['depth_max']:.1f} m"
              if summary.get("depth_min") is not None else "n/a")),
        ]
    else:
        def rng(lo, hi):
            return "n/a" if lo is None else f"{lo:.1f}-{hi:.1f}%"
        chips = [
            ("Total rows", f'{summary.get("n_total", 0):,}'),
            ("Valid rows", f'{summary.get("n_valid", 0):,}'),
            ("Missing values", f'{summary.get("n_missing_values", 0):,}'),
            ("Duplicates", f'{summary.get("n_duplicates", 0):,}'),
            ("Total tonnage", f'{summary.get("total_tonnage", 0):,.0f} t'),
            ("Mn range", rng(summary.get("mn_min"), summary.get("mn_max"))),
            ("Fe range", rng(summary.get("fe_min"), summary.get("fe_max"))),
            ("Cost range",
             (f"{summary['cost_min']:.0f}-{summary['cost_max']:.0f} Rs/t"
              if summary.get("cost_min") is not None else "n/a")),
        ]
    html = "".join(
        f'<span class="qc-chip"><b>{label}:</b> {value}</span>'
        for label, value in chips)
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state + audit trail
# ---------------------------------------------------------------------------

def now_stamp() -> str:
    return datetime.now().strftime("%d %b %Y %H:%M:%S")


def append_audit(kind: str, summary: str) -> None:
    trail = st.session_state.audit_trail
    trail.append({"timestamp": now_stamp(), "kind": kind, "summary": summary})
    if len(trail) > 50:
        st.session_state.audit_trail = trail[-50:]


def render_audit_trail() -> None:
    with st.expander("Session Audit Trail", expanded=False):
        trail = st.session_state.audit_trail
        if not trail:
            st.caption("No modelling runs recorded in this session yet.")
            return
        st.dataframe(
            pd.DataFrame(trail),
            width="stretch", hide_index=True,
            column_config={
                "timestamp": st.column_config.TextColumn("Timestamp"),
                "kind": st.column_config.TextColumn("Stage"),
                "summary": st.column_config.TextColumn("Details"),
            },
        )
        st.caption("Audit trail records the key events of this session. It "
                   "is in-memory only and is never persisted to disk.")


def init_state() -> None:
    st.session_state.setdefault("audit_trail", [])

    if "geo_df" not in st.session_state:
        st.session_state.geo_df = get_kriged_block_model(
            DEFAULT_PIT, DEFAULT_DEPTH, DEFAULT_CUTOFF)
    st.session_state.setdefault(
        "geo_run", {
            "label": f"{DEFAULT_PIT} &middot; Depth {DEFAULT_DEPTH} m",
            "source": "Demonstration Dataset",
            "depth": DEFAULT_DEPTH,
            "cutoff": DEFAULT_CUTOFF,
        })

    if "opt_result" not in st.session_state:
        st.session_state.opt_result = calculate_optimal_blend(
            load_sample_stockpiles(), DEFAULT_TARGET_TONNAGE,
            DEFAULT_TARGET_MN, DEFAULT_MAX_FE)
        st.session_state.opt_verification = verify_blend(
            st.session_state.opt_result, DEFAULT_TARGET_TONNAGE,
            DEFAULT_TARGET_MN, DEFAULT_MAX_FE, None)
    st.session_state.setdefault(
        "opt_params", (DEFAULT_TARGET_TONNAGE, DEFAULT_TARGET_MN,
                       DEFAULT_MAX_FE, None))
    st.session_state.setdefault("opt_source", "Demonstration Dataset")
    st.session_state.setdefault("opt_verification", None)


def current_block_summary() -> dict:
    df = st.session_state.geo_df
    return {
        "n_blocks": len(df),
        "avg_mn": float(df["Mn_Grade"].mean()),
        "avg_fe": float(df["Fe_Grade"].mean()),
        "unfc_111": int((df["UNFC_Code"] == "111").sum()),
    }


# ===========================================================================
# PAGE 1 : GEOSPATIAL AI
# ===========================================================================

def page_geospatial() -> None:
    model_df = st.session_state.geo_df
    run_info = st.session_state.geo_run

    total = len(model_df)
    avg_mn = model_df["Mn_Grade"].mean()
    avg_fe = model_df["Fe_Grade"].mean()
    screened = int((model_df["UNFC_Code"] == "111").sum())
    screen_pct = screened / total * 100 if total else 0.0

    render_kpis([
        ("Total Blocks", f"{total:,}", PRIMARY, run_info["label"]),
        ("Avg Mn Grade", f"{avg_mn:.2f}%", PRIMARY, "block mean"),
        ("Avg Fe Grade", f"{avg_fe:.2f}%", SECONDARY, "block mean"),
        ("UNFC 111 Screening", f"{screened:,}", SECONDARY,
         f"{screen_pct:.1f}% of blocks"),
    ])

    st.markdown('<div class="section-title">3D Subsurface Grade Modeling</div>',
                unsafe_allow_html=True)

    controls, viz = st.columns([4, 9], gap="large")

    with controls:
        st.markdown("<h3>Model Controls</h3>", unsafe_allow_html=True)
        source = st.radio(
            "Geological Data Source", GEOLOGICAL_SOURCE_OPTIONS, index=0,
            horizontal=True, label_visibility="collapsed")

        geo_clean = None
        geo_file_label = None

        if source == "Demonstration Dataset":
            pit = st.selectbox("Pit Selection", list_pits(),
                               index=list_pits().index(DEFAULT_PIT))
            depth = st.slider("Modelled Depth (m)", 10, 100, DEFAULT_DEPTH,
                              step=10)
            cutoff = st.slider(
                "Confidence / Variance Threshold (%)",
                1.0, 50.0, float(DEFAULT_CUTOFF), step=0.5,
                help=("Blocks with relative kriging variance below this "
                      "threshold are screened as UNFC 111. Concept baseline: "
                      "prediction variance < 3%."))
            run = st.button("Run Kriging Model", width="stretch",
                            type="primary", disabled=False)

            params = get_variogram_parameters(pit)
            st.markdown(
                f"""
                <div class="note" style="margin-top:0.6rem">
                  <b>Spherical Variogram Model</b><br>
                  Nugget C0: {params['nugget']:.1f} &nbsp;|&nbsp;
                  Sill C: {params['partial_sill']:.1f}<br>
                  Range a: {params['range_m']:.0f} m &nbsp;|&nbsp;
                  Z anisotropy: {params['anisotropy_z']:.1f}x
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption("Variogram parameters are configurable; there is no "
                       "automatic fitting. UNFC 111 is a screening status "
                       "from the kriging-variance criterion.")
        else:
            up = st.file_uploader(
                "Upload geological samples (CSV / Excel)",
                type=["csv", "xlsx"],
                help=("Columns: Borehole_ID, X (or Easting), Y (or Northing), "
                      "Depth, Mn_Grade, Fe_Grade. "
                      "Try the sample file sample_geological_data.csv "
                      "from the project folder."))
            depth = st.slider("Modelled Depth (m)", 10, 100, DEFAULT_DEPTH,
                              step=10)
            cutoff = st.slider(
                "Confidence / Variance Threshold (%)",
                1.0, 50.0, float(DEFAULT_CUTOFF), step=0.5,
                help=("Blocks with relative kriging variance below this "
                      "threshold are screened as UNFC 111."))

            st.markdown("<b>Variogram model parameters</b>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                v_nugget = st.number_input("Nugget (C0)", min_value=0.0,
                                           max_value=50.0, value=0.3, step=0.1)
                v_range = st.number_input("Range (a, m)", min_value=50.0,
                                          max_value=5000.0, value=1000.0,
                                          step=50.0)
            with c2:
                v_sill = st.number_input("Partial sill (C)", min_value=1.0,
                                         max_value=200.0, value=32.0, step=1.0)
                v_aniso = st.number_input("Z anisotropy", min_value=0.5,
                                          max_value=10.0, value=2.0, step=0.5)
            st.caption("Spherical variogram model with configurable nugget, sill and "
                   "range parameters. There is no automatic fitting.")

            geo_ok = False
            if up is not None:
                geo_file_label = up.name
                try:
                    raw = parse_upload(up)
                    qc = validate_geological_data(raw)
                except DataLoadError as exc:
                    st.markdown(f'<div class="err">{exc}</div>',
                                unsafe_allow_html=True)
                else:
                    when_fail = ""
                    if not qc["ok"]:
                        when_fail = " The uploaded samples are not usable, so "
                        "the model will not be updated."
                        st.markdown(
                            f'<div class="err">{" ".join(qc["errors"])}'
                            f"{when_fail}</div>",
                            unsafe_allow_html=True)
                    for warn_for in qc["warnings"]:
                        st.markdown(f'<div class="warn">{warn_for}</div>',
                                    unsafe_allow_html=True)
                    render_qc_chips(qc["summary"], "geo")
                    geo_ok = qc["ok"]
                    if geo_ok:
                        geo_clean = qc["cleaned"]
                        st.caption(f"Validation OK - {geo_file_label}")
            else:
                st.caption("Upload a CSV or Excel file to enable the "
                           "model run.")

            run = st.button("Run Kriging Model", width="stretch",
                            type="primary", disabled=not geo_ok)

    with viz:
        source_label = "Demonstration Dataset"
        if run:
            with st.spinner("Running Ordinary Kriging over block grid..."):
                try:
                    if source == "Demonstration Dataset":
                        model_df = get_kriged_block_model(pit, depth, cutoff)
                        run_info = {
                            "label": f"{pit} &middot; Depth {depth} m",
                            "source": source_label,
                            "depth": depth, "cutoff": cutoff}
                    else:
                        model_df = krige_uploaded_samples(
                            geo_clean, depth, cutoff, nugget=v_nugget,
                            partial_sill=v_sill, range_m=v_range,
                            anisotropy_z=v_aniso)
                        source_label = f"Upload - {geo_file_label}"
                        run_info = {
                            "label": f"Depth {depth} m",
                            "source": source_label,
                            "depth": depth, "cutoff": cutoff}
                    st.session_state.geo_df = model_df
                    st.session_state.geo_run = run_info
                    append_audit(
                        "Geological model",
                        f"SOURCE: {source_label} | {len(model_df)} blocks | "
                        f"avg Mn {model_df['Mn_Grade'].mean():.2f}% | "
                        f"avg Fe {model_df['Fe_Grade'].mean():.2f}% | "
                        f"UNFC 111 {(model_df['UNFC_Code'] == '111').sum()}.")
                    # Re-render so every KPI, chart and statistic on the page
                    # reflects the freshly built model (never a mix of the
                    # previous demonstration state and the uploaded model).
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))

        st.markdown(
            f"<div class='panel'><h3>3D Block Model "
            f"&mdash; {run_info['label']} &middot; Threshold "
            f"{run_info['cutoff']:g}%</h3></div>",
            unsafe_allow_html=True)

        fig3d = go.Figure(go.Scatter3d(
            x=model_df["X"], y=model_df["Y"], z=model_df["Z"],
            mode="markers",
            marker=dict(size=3.2, color=model_df["Mn_Grade"],
                        colorscale="Viridis",
                        colorbar=dict(title="Mn %", thickness=14,
                                      tickfont=dict(color=TEXT_MUTED)),
                        showscale=True),
            hovertemplate=(
                "X %{customdata[0]:.0f}  Y %{customdata[1]:.0f}  "
                "Z %{customdata[2]:.0f}m<br>" +
                "<extra></extra>"
            ),
            customdata=np.column_stack([
                model_df["X"], model_df["Y"], model_df["Z"],
                model_df["Mn_Grade"],
                model_df["Fe_Grade"],
                model_df["UNFC_Code"],
                model_df["Kriging_Variance"],
            ]),
        ))
        fig3d.update_traces(hovertemplate=(
            "X %{customdata[0]:.0f}  Y %{customdata[1]:.0f}  "
            "Z %{customdata[2]:.0f} m<br>"
            "Mn Grade %{customdata[3]:.2f}%<br>"
            "Fe Grade %{customdata[4]:.2f}%<br>"
            "UNFC %{customdata[5]}<br>"
            "Kriging Variance %{customdata[6]:.2f}%<extra></extra>"
        ))
        fig3d.update_layout(
            title=dict(text="Block Model coloured by Mn Grade (%)",
                       font=dict(color=TEXT, size=15)),
            scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)",
                       zaxis_title="Depth (m)"),
            height=560,
        )
        fig3d = scene_style(fig3d)
        st.plotly_chart(fig3d, width="stretch",
                        config={"displayModeBar": False})

    # ----------------------------------------------------------------- charts
    st.markdown('<div class="section-title">Geological Analytics</div>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig1 = px.histogram(model_df, x="Mn_Grade", nbins=24,
                            title="Mn Grade Distribution",
                            color_discrete_sequence=[PRIMARY])
        fig1.update_traces(marker=dict(line=dict(width=0)))
        fig1.add_vline(x=avg_mn, line_color=SECONDARY, line_dash="dash",
                       annotation_text=f"mean {avg_mn:.1f}%")
        fig1.update_xaxes(title="Mn Grade (%)")
        fig1.update_yaxes(title="Blocks")
        st.plotly_chart(style_fig(fig1, 330), width="stretch",
                        config={"displayModeBar": False})

        fig3 = px.scatter(model_df.sample(min(1500, len(model_df)),
                                          random_state=7),
                          x="Mn_Grade", y="Fe_Grade",
                          color="Mn_Grade", color_continuous_scale="Viridis",
                          title="Mn vs Fe Relationship",
                          labels={"Mn_Grade": "Mn (%)", "Fe_Grade": "Fe (%)"})
        fig3.update_traces(marker=dict(size=4, opacity=0.7))
        st.plotly_chart(style_fig(fig3, 330), width="stretch",
                        config={"displayModeBar": False})

    with col_b:
        fig2 = px.histogram(model_df, x="Fe_Grade", nbins=24,
                            title="Fe Grade Distribution",
                            color_discrete_sequence=[SECONDARY])
        fig2.update_traces(marker=dict(line=dict(width=0)))
        fig2.add_vline(x=avg_fe, line_color=PRIMARY, line_dash="dash",
                       annotation_text=f"mean {avg_fe:.1f}%")
        fig2.update_xaxes(title="Fe Grade (%)")
        fig2.update_yaxes(title="Blocks")
        st.plotly_chart(style_fig(fig2, 330), width="stretch",
                        config={"displayModeBar": False})

        depth_profile = (model_df.groupby("Z")["Mn_Grade"]
                         .agg(["mean", "min", "max"]).reset_index())
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=depth_profile["mean"],
                                  y=depth_profile["Z"], mode="lines+markers",
                                  name="Mean Mn", line=dict(color=PRIMARY),
                                  marker=dict(size=6)))
        fig4.add_trace(go.Scatter(
            x=depth_profile["max"], y=depth_profile["Z"],
            mode="lines", name="Max Mn", line=dict(color="#22d3ee", dash="dot")))
        fig4.add_trace(go.Scatter(
            x=depth_profile["min"], y=depth_profile["Z"],
            mode="lines", name="Min Mn", line=dict(color=WARNING, dash="dot")))
        fig4.update_layout(title="Depth-wise Mn Grade Profile",
                           xaxis_title="Mn Grade (%)",
                           yaxis_title="Depth (m)",
                           yaxis=dict(autorange="reversed"))
        st.plotly_chart(style_fig(fig4, 330), width="stretch",
                        config={"displayModeBar": False})

    counts = model_df["UNFC_Code"].value_counts().rename_axis("UNFC_Code")
    fig5 = go.Figure(go.Pie(
        labels=counts.index.tolist(),
        values=counts.values.tolist(),
        hole=0.45,
        marker=dict(colors=[SECONDARY, WARNING]),
        textinfo="label+percent",
    ))
    fig5.update_layout(title="UNFC Screening Distribution",
                       showlegend=False)
    st.plotly_chart(style_fig(fig5, 300), width="stretch",
                    config={"displayModeBar": False})

    # ------------------------------------------------------------ summary
    st.markdown('<div class="section-title">Model Summary</div>',
                unsafe_allow_html=True)
    avg_var = model_df["Kriging_Variance"].mean()
    render_kpis([
        ("Data Source", run_info["source"], PRIMARY, None),
        ("Modelled Depth", f"{run_info['depth']:g} m", PRIMARY, None),
        ("Number of Blocks", f"{total:,}", PRIMARY, None),
        ("Avg Kriging Variance", f"{avg_var:.2f}%", WARNING, "relative to sill"),
        ("Average Mn Grade", f"{avg_mn:.2f}%", PRIMARY, None),
        ("Average Fe Grade", f"{avg_fe:.2f}%", SECONDARY, None),
        ("Minimum Mn", f"{model_df['Mn_Grade'].min():.2f}%", WARNING, None),
        ("Maximum Mn", f"{model_df['Mn_Grade'].max():.2f}%", WARNING, None),
        ("UNFC 111 Screening", f"{screened:,}", SECONDARY, None),
        ("Screening Share", f"{screen_pct:.1f}%", SECONDARY, None),
    ])

    st.markdown(
        '<div class="note">UNFC 111 indicates the block satisfies the '
        "implemented screening criterion (relative kriging variance below "
        "the threshold). It is a calculated status result, not a regulatory "
        "certification.</div>",
        unsafe_allow_html=True,
    )

    render_audit_trail()


# ===========================================================================
# PAGE 2 : GRADE BLENDING
# ===========================================================================

def page_blending() -> None:
    st.markdown('<div class="section-title">Prescriptive Ore Grade Blending</div>',
                unsafe_allow_html=True)

    controls, table = st.columns([4, 9], gap="large")

    with controls:
        st.markdown("<h3>Optimization Controls</h3>", unsafe_allow_html=True)
        source = st.radio(
            "Stockpile Data Source", STOCKPILE_SOURCE_OPTIONS, index=0,
            horizontal=True, label_visibility="collapsed")

        stockpiles = None
        stock_source = "Demonstration Dataset"
        stock_ready = False

        if source == "Demonstration Dataset":
            stockpiles = load_demonstration_stockpiles()
            stock_source = "Demonstration Dataset"
            stock_ready = True
        else:
            up = st.file_uploader(
                "Upload stockpile data (CSV / Excel)",
                type=["csv", "xlsx"],
                help=("Columns: Stockpile_ID, Available_Tons, Mn_Grade, "
                      "Fe_Grade, Cost_per_Ton."))
            if up is not None:
                stock_source = f"Upload - {up.name}"
                try:
                    raw = parse_upload(up)
                    sqc = validate_stockpile_data(raw)
                except DataLoadError as exc:
                    st.markdown(f'<div class="err">{exc}</div>',
                                unsafe_allow_html=True)
                else:
                    if not sqc["ok"]:
                        st.markdown(
                            f'<div class="err">{" ".join(sqc["errors"])}'
                            " The uploaded stockpiles are not usable.</div>",
                            unsafe_allow_html=True)
                    for warn_for in sqc["warnings"]:
                        st.markdown(f'<div class="warn">{warn_for}</div>',
                                    unsafe_allow_html=True)
                    render_qc_chips(sqc["summary"], "stock")
                    stock_ready = sqc["ok"]
                    if stock_ready:
                        stockpiles = sqc["cleaned"]
                        st.caption(f"Validation OK - {up.name}")
            else:
                st.caption("Upload a CSV or Excel file to enable the "
                           "optimization run.")

        target_tons = st.number_input(
            "Target Tonnage (tons)", min_value=1000, max_value=30000,
            value=DEFAULT_TARGET_TONNAGE, step=500)
        target_mn = st.number_input(
            "Target Minimum Mn Grade (%)", min_value=10.0, max_value=90.0,
            value=DEFAULT_TARGET_MN, step=0.5)
        max_fe = st.number_input(
            "Maximum Fe Grade (%)", min_value=5.0, max_value=60.0,
            value=DEFAULT_MAX_FE, step=0.5)
        max_cost = st.number_input(
            "Maximum Cost per Ton (Rs/t) - optional", min_value=0.0,
            max_value=5000.0, value=0.0, step=100.0,
            help=("Leave at 0 to run without a budget constraint. When set, "
                  "the solver must also keep the blended cost per ton at or "
                  "below this value."))
        max_cost = None if max_cost <= 0 else float(max_cost)
        run = st.button("Run Optimization", width="stretch",
                        type="primary", disabled=not stock_ready)
        st.caption("Cost Difference vs Reference compares the solution "
                   "against a grade-priority reference blend.")

    with table:
        st.markdown(
            f"<div class='panel'><h3>Stockpile Availability "
            f"({stock_source})</h3></div>",
            unsafe_allow_html=True)
        if stockpiles is not None:
            st.dataframe(stockpiles, width="stretch", hide_index=True)
        else:
            st.caption("No stockpile data loaded yet.")

    result = st.session_state.opt_result
    params_running = st.session_state.opt_params

    if run:
        try:
            result = calculate_optimal_blend(
                stockpiles, target_tons, target_mn, max_fe,
                max_cost_per_ton=max_cost)
            st.session_state.opt_result = result
            st.session_state.opt_params = (target_tons, target_mn, max_fe,
                                           max_cost)
            st.session_state.opt_source = stock_source
            params_running = (target_tons, target_mn, max_fe, max_cost)
            if result["success"]:
                verification = verify_blend(result, target_tons, target_mn,
                                            max_fe, max_cost)
                st.session_state.opt_verification = verification
                append_audit(
                    "Optimization",
                    f"SOURCE: {stock_source} | {result['status']} | "
                    f"{result['total_tonnage']:,.0f} t | Mn "
                    f"{result['blended_mn']:.2f}% | Fe "
                    f"{result['blended_fe']:.2f}% | cost/t Rs "
                    f"{result['cost_per_ton']:,.2f}.")
                # Re-render so the data-source badge, KPIs and tables all
                # reflect the uploaded dataset in this same render pass.
                st.rerun()
            else:
                st.session_state.opt_verification = None
                append_audit("Optimization",
                             f"SOURCE: {stock_source} | {result['status']} | "
                             f"{result['message']}")
        except Exception as exc:  # pragma: no cover - defence in depth
            st.error(result["message"] if isinstance(result, dict)
                     and "message" in result else f"Optimization error: {exc}")

    if not isinstance(result, dict) or not result.get("success"):
        st.markdown(f'<div class="err">{result.get("message", "")}</div>',
                    unsafe_allow_html=True)
        render_audit_trail()
        return

    t_tons, t_mn, t_fe, t_cost = params_running
    st.markdown(
        f"<div class='panel'><h3>Optimal Blend &mdash; Target "
        f"{t_tons:,.0f} t &middot; Mn ≥ {t_mn:.1f}% &middot; Fe ≤ {t_fe:.1f}%"
        + (f" &middot; Cost ≤ Rs {t_cost:,.0f}/t" if t_cost else "")
        + "</h3></div>",
        unsafe_allow_html=True)

    verif = st.session_state.opt_verification
    if verif:
        all_pass = verif.get("all_pass", False)
        verdict_html = ('<span class="badge-pass">ALL CONSTRAINTS SATISFIED'
                        '</span>' if all_pass else
                        '<span class="badge-fail">CONSTRAINT VIOLATION '
                        'DETECTED</span>')
        rows = "".join(
            f'<div style="padding:0.18rem 0">'
            f'<span class="badge-{"pass" if c["status"]=="PASS" else "fail"}">'
            f'{c["status"]}</span> '
            f'<span style="color:{TEXT}">{c["constraint"]}</span> '
            f'<span style="color:{TEXT_MUTED}">required {c["required"]} '
            f'&middot; achieved {c["achieved"]}</span></div>'
            for c in verif["checks"])
        st.markdown(
            f'<div class="panel"><h3>Constraint Validation {verdict_html}</h3>'
            f"{rows}</div>",
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="warn">Blend not re-verified (constraint validation '
            "skipped).</div>",
            unsafe_allow_html=True)

    render_kpis([
        ("Optimization Status", result["status"], SECONDARY, "LP solver"),
        ("Total Blended Tonnage", f"{result['total_tonnage']:,.0f} t",
         PRIMARY, None),
        ("Blended Mn Grade", f"{result['blended_mn']:.2f}%", PRIMARY,
         f"target ≥ {t_mn:.1f}%"),
        ("Blended Fe Grade", f"{result['blended_fe']:.2f}%", SECONDARY,
         f"limit ≤ {t_fe:.1f}%"),
        ("Total Cost", f"₹{result['total_cost']:,.0f}", WARNING,
         "blended total"),
        ("Cost / Ton", f"₹{result['cost_per_ton']:,.0f}/t", WARNING,
         "blended average"),
        ("Stockpile Utilization", f"{result['utilization_pct']:.1f}%",
         SECONDARY, "of total available"),
        ("Cost Difference vs Reference", f"₹{result['savings']:,.0f}",
         SECONDARY, "vs grade-priority reference blend"),
    ])

    # -------------------------------------------------------------- charts
    df_out = result["result_df"]

    col1, col2 = st.columns(2)
    with col1:
        figa = px.bar(
            df_out, x="Stockpile_ID", y="Allocated_Tons",
            title="Stockpile Allocation (tons)",
            color="Stockpile_ID",
            color_discrete_sequence=[PRIMARY, SECONDARY, WARNING,
                                     "#818CF8", "#F472B6"],
        )
        figa.update_layout(showlegend=False)
        st.plotly_chart(style_fig(figa, 340), width="stretch",
                        config={"displayModeBar": False})

        figc = go.Figure()
        figc.add_trace(go.Bar(name="Available (t)", x=df_out["Stockpile_ID"],
                              y=df_out["Available_Tons"],
                              marker_color="#475569"))
        figc.add_trace(go.Bar(name="Allocated (t)", x=df_out["Stockpile_ID"],
                              y=df_out["Allocated_Tons"], marker_color=SECONDARY))
        figc.update_layout(title="Allocation vs Availability",
                           barmode="group", showlegend=True)
        st.plotly_chart(style_fig(figc, 340), width="stretch",
                        config={"displayModeBar": False})

    with col2:
        figb = go.Figure()
        figb.add_trace(go.Bar(name="Mn Grade (%)", x=df_out["Stockpile_ID"],
                              y=df_out["Mn_Grade"], marker_color=PRIMARY))
        figb.add_trace(go.Bar(name="Fe Grade (%)", x=df_out["Stockpile_ID"],
                              y=df_out["Fe_Grade"], marker_color=WARNING))
        figb.update_layout(title="Stockpile Grade Comparison (Mn vs Fe)",
                           barmode="group")
        st.plotly_chart(style_fig(figb, 712), width="stretch",
                        config={"displayModeBar": False})

    # ----------------------------------------------------------- result table
    st.markdown('<div class="section-title">Blend Allocation Detail</div>',
                unsafe_allow_html=True)
    display = df_out[["Stockpile_ID", "Available_Tons", "Allocated_Tons",
                      "Mn_Grade", "Fe_Grade", "Cost_per_Ton",
                      "Utilization_Pct", "In_Blend"]].copy()
    display.columns = ["Stockpile ID", "Available (t)", "Allocated (t)",
                       "Mn %", "Fe %", "Cost/Ton", "Utilization %",
                       "Status"]

    st.dataframe(
        display,
        width="stretch", hide_index=True,
        column_config={
            "Available (t)": st.column_config.NumberColumn(format="%d"),
            "Allocated (t)": st.column_config.NumberColumn(format="%.0f"),
            "Mn %": st.column_config.NumberColumn(format="%.1f"),
            "Fe %": st.column_config.NumberColumn(format="%.1f"),
            "Cost/Ton": st.column_config.NumberColumn(format="%d"),
            "Utilization %": st.column_config.NumberColumn(format="%.1f"),
        },
    )

    st.markdown(
        '<div class="note">All stockpiles are listed. The Status column '
        "identifies stockpiles selected for the final blend. Grade and cost "
        "figures reflect the data source in use.</div>",
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------ export
    st.markdown('<div class="section-title">Export Decision Report</div>',
                unsafe_allow_html=True)
    ctx = {
        "data_source": st.session_state.opt_source,
        "generated_at": now_stamp(),
        "scenario": {"target_tonnage": t_tons, "target_mn": t_mn,
                     "max_fe": t_fe, "max_cost_per_ton": t_cost},
        "block_model": current_block_summary(),
        "verification": verif or {},
    }
    tmp_dir = tempfile.mkdtemp(prefix="mnbrain_report_")

    def _materialise() -> dict[str, bytes]:
        files = export_decision_report(result, tmp_dir, ctx)
        payload = {}
        for path in files:
            with open(path, "rb") as fh:
                payload[os.path.basename(path)] = fh.read()
        return payload

    try:
        payload = _materialise()
        pdf_path = next((k for k in payload if k.endswith(".pdf")), None)
        txt_path = next((k for k in payload if k.endswith(".txt")), None)
        clicked = False
        if pdf_path:
            if st.download_button("Download PDF Report",
                                  data=payload[pdf_path], file_name=pdf_path,
                                  mime="application/pdf", type="primary"):
                clicked = True
        if txt_path:
            if st.download_button("Download Text Report",
                                  data=payload[txt_path], file_name=txt_path,
                                  mime="text/plain"):
                clicked = True
        csv_names = [k for k in payload if k.endswith(".csv")]
        if csv_names:
            buf = tempfile.TemporaryFile()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for name in csv_names:
                    zf.writestr(name.split("_", 1)[1], payload[name])
            buf.seek(0)
            zdata = buf.read()
            buf.close()
            if st.download_button("Download CSVs (ZIP)",
                                  data=zdata, file_name="mnbrain_csv_package.zip",
                                  mime="application/zip"):
                clicked = True
        if clicked:
            append_audit("Report", f"SOURCE: {st.session_state.opt_source} | "
                                   f"Decision report exported "
                                   f"(PDF + TXT + CSV) for target "
                                   f"{t_tons:,.0f} t.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    render_audit_trail()


# ===========================================================================
# PAGE 3 : ABOUT / METHODOLOGY
# ===========================================================================

DATA_FLOW = [
    "Data Ingestion", "Validation & QA", "Geological Model",
    "Spherical Variogram", "Ordinary Kriging", "3D Block Model",
    "Grade + Uncertainty", "UNFC 111 Screening", "Stockpile Data",
    "LP Optimization", "Constraint Validation", "Decision Support & Report",
]

METHOD_STEPS = [
    ("STEP 1 - Data Ingestion",
     "Data can be loaded from the bundled Demonstration Dataset or from "
     "user-uploaded CSV/Excel files. Uploaded geological and stockpile "
     "tables are normalised (aliases such as Easting -> X are accepted) "
     "into the standard internal schema."),
    ("STEP 2 - Data Validation",
     "Uploaded data is validated before any modelling: required columns, "
     "numeric values, grade ranges (0-100%), non-negative depths and "
     "tonnage, duplicate records and ID integrity. Data that fails "
     "validation is never passed to the kriging or optimization engines, "
     "and a quality summary is shown to the user."),
    ("STEP 3 - Spherical Variogram",
     "Spatial correlation is modelled with a spherical variogram model "
     "defined by nugget (C0), partial sill (C), range (a) and a vertical "
     "anisotropy factor. These are configurable parameters; there is no "
     "automatic fitting."),
    ("STEP 4 - Ordinary Kriging",
     "For every block centre, kriging weights lambda are solved from the "
     "kriging system so that sum(lambda) = 1, and the block grade is "
     "estimated as Z_hat = sum(lambda_i * Z_i)."),
    ("STEP 5 - 3D Block Modeling",
     "Estimated grades are written to a regular 3D block model (X, Y, Z, "
     "Mn_Grade, Fe_Grade, Kriging_Variance, Confidence) for visualisation "
     "and aggregation."),
    ("STEP 6 - Uncertainty Estimation",
     "Kriging variance is computed and reported relative to the global "
     "sill, giving a transparent per-block uncertainty metric."),
    ("STEP 7 - UNFC 111 Screening",
     "Blocks with relative kriging variance below the configured threshold "
     "receive UNFC code 111 - a screening indicator based on prediction "
     "variance, not a regulatory certification. All other blocks are "
     "flagged UNVERIFIED."),
    ("STEP 8 - Stockpile Optimization",
     "Stockpile records (grade, tonnage, cost) are validated for negative "
     "tonnage, empty IDs and out-of-range grades. A cost-minimising linear "
     "program is then solved with scipy linprog: tonnage, Mn, Fe and "
     "(optionally) cost-per-ton constraints are linear, and each stockpile "
     "is bounded by its availability."),
    ("STEP 9 - Constraint Validation",
     "The returned blend is re-checked independently against every "
     "constraint - tonnage, Mn, Fe, stockpile availability and the "
     "optional budget - using the exact solver allocations, and reported "
     "as PASS or FAIL."),
    ("STEP 10 - Decision Support",
     "A decision-support report (TXT / CSV / PDF) is produced with inputs, "
     "allocations, blended metrics and the PASS/FAIL verdict, together "
     "with an in-memory session audit trail."),
]


def page_about() -> None:
    brand_header()

    st.markdown(
        '<div class="panel"><h3>MnBrain - Geospatial AI & Grade Blending '
        "Intelligence</h3>"
        "<p>SIH 2026 / Ministry of Steel &middot; Problem Statement ID "
        "26009<br>"
        "<b>Subtitle:</b> 3D Subsurface Modeling + Prescriptive Grade "
        "Blending</p>"
        "<p>MnBrain delivers two integrated modules: (1) 3D Subsurface Grade "
        "Modeling with Ordinary Kriging, and (2) Prescriptive Grade Blending "
        "Optimization using linear programming. Data can come from the "
        "bundled Demonstration Dataset or from validated user uploads. The "
        "system runs fully offline.</p></div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Processing Pipeline</div>',
                unsafe_allow_html=True)
    flow_cols = st.columns([2, 8, 2])
    with flow_cols[1]:
        for i, stage in enumerate(DATA_FLOW):
            st.markdown(f'<div class="flow-box">{stage}</div>',
                        unsafe_allow_html=True)
            if i < len(DATA_FLOW) - 1:
                st.markdown('<div class="flow-arrow">&#8595;</div>',
                            unsafe_allow_html=True)

    st.markdown('<div class="section-title">How the Model Works</div>',
                unsafe_allow_html=True)
    with st.expander("How the Model Works - 10 step methodology",
                     expanded=True):
        for title, body in METHOD_STEPS:
            st.markdown(f"<b>{title}</b><br>&nbsp;&nbsp;{body}",
                        unsafe_allow_html=True)
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Mathematical Formulation</div>',
                unsafe_allow_html=True)
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown(
            '<div class="panel"><h3>Module 1 - Ordinary Kriging</h3>'
            "<p>Z_hat(x0) = &Sigma; &lambda;_i &middot; Z(x_i) &nbsp;, "
            "&nbsp; subject to &Sigma; &lambda;_i = 1</p>"
            "<p>Spherical variogram:</p>"
            "<p>&gamma;(h) = C0 + C&middot;[1.5(h/a) - 0.5(h/a)&sup3;] "
            "for 0 &lt; h &le; a</p>"
            "<p>&gamma;(h) = C0 + C&nbsp;&nbsp; for h &gt; a</p>"
            "<p>Kriging variance &sigma;&sup2; = &Sigma; &lambda;_i "
            "&gamma;(h_i0) + &mu;</p></div>",
            unsafe_allow_html=True,
        )
    with col_m2:
        st.markdown(
            '<div class="panel"><h3>Module 2 - Grade Blending LP</h3>'
            "<p>Minimize: &nbsp; &Sigma;(c_j &middot; w_j)</p>"
            "<p>Subject to:</p>"
            "<p>&Sigma; w_j = target_tonnage<br>"
            "&Sigma;(Mn_j &middot; w_j) &ge; target_Mn &middot; "
            "target_tonnage<br>"
            "&Sigma;(Fe_j &middot; w_j) &le; Fe_max &middot; "
            "target_tonnage<br>"
            "&Sigma;(c_j &middot; w_j) &le; cost_max &middot; "
            "target_tonnage &nbsp; (optional)<br>"
            "0 &le; w_j &le; Available_Tons_j</p>"
            "<p><b>Verification:</b> the solution is re-checked against every "
            "constraint and reported PASS/FAIL.</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Notes</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="note" style="margin-bottom:0.5rem"><b>Data sources:</b> '
        "the bundled Demonstration Dataset is synthetic. Uploaded files are "
        "validated but the user remains responsible for data quality and "
        "units (metres and tonnes).</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="note" style="margin-bottom:0.5rem"><b>UNFC 111:</b> in '
        "this application, UNFC code 111 is a screening status derived from "
        "the kriging-variance threshold, not a regulatory resource "
        "classification.</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="panel" style="margin-top:1rem"><h3>Technology Stack</h3>'
        "<p>Python &middot; NumPy &middot; Pandas &middot; SciPy (linprog, "
        "KDTree, ndimage) &middot; Plotly &middot; Streamlit &middot; "
        "openpyxl (Excel ingestion)</p>"
        "<p>Runs fully offline. No external APIs, databases, GIS servers or "
        "commercial solvers required. Reports (TXT / CSV / PDF) are generated "
        "with the standard library only.</p></div>",
        unsafe_allow_html=True,
    )

    render_audit_trail()


# ===========================================================================
# SIDEBAR NAVIGATION
# ===========================================================================

def main() -> None:
    inject_css()
    init_state()

    with st.sidebar:
        st.markdown(
            """
            <div class="mb-brand">
              <div class="title" style="font-size:1.4rem">Mn<span>Brain</span>
              </div>
              <div class="tagline">Geospatial AI & Grade Blending</div>
              <span class="badge synth">Demonstration Dataset</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        choice = st.radio(
            "Navigation", ["Geospatial AI", "Grade Blending",
                           "About / Methodology"],
            label_visibility="collapsed",
        )
        st.markdown(
            '<div class="note" style="margin-top:1rem">SIH 2026 &middot; '
            "Ministry of Steel &middot; PS ID 26009<br>Geospatial AI &amp; "
            "Optimization Engine</div>",
            unsafe_allow_html=True,
        )

    # Main content header (on functional pages)
    if choice == "Geospatial AI":
        src = st.session_state.geo_run.get("source", "Demonstration Dataset")
        brand_header("Uploaded Dataset" if str(src).startswith("Upload")
                     else "Demonstration Dataset")
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        page_geospatial()
    elif choice == "Grade Blending":
        src = st.session_state.get("opt_source", "Demonstration Dataset")
        brand_header("Uploaded Dataset" if str(src).startswith("Upload")
                     else "Demonstration Dataset")
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        page_blending()
    else:
        page_about()


if __name__ == "__main__":
    main()