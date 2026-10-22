"""Sensora interface: pressure range in, robust composite designs out."""
from __future__ import annotations

from dataclasses import asdict
import json
import sqlite3

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .composite_physics import Targets, planar_capacitance_response
from .composite_optimization import (
    CompositeOptimizationResult,
    GeneticOptimizationSettings,
    optimize_composites,
)
from .assistant_ui import assistant_page
from .config import ROOT
from .experiment_plan import build_experiment_plan
from .database import MaterialDatabase, read_database
from .material_store import MaterialStoreError, add_material, delete_materials
from .supabase_store import configured_store


DILUTE = "Below 5 vol%"
CONCENTRATED = "5 to 20 vol%"
CONCENTRATED_SETTINGS = dict(
    loading_regime="concentrated", minimum_total_filler_vol_percent=5.0,
    maximum_total_filler_vol_percent=20.0, loading_step_vol_percent=0.5,
)
ICON = ROOT / "assets/icon.png"
LOGO = ROOT / "assets/logo.png"


# Reference categorical palette (light surface), used in fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
FONT = "Roboto, sans-serif"
INK, INK_SOFT, GRID = "#142b40", "#52606d", "#e6ebf0"
BLUE_RAMP = [[0, "#e8f0fb"], [0.5, "#6ea3e6"], [1, "#1d4f93"]]

CSS = """
<style>
.block-container{max-width:1280px;padding-top:1.6rem;padding-bottom:3rem}
#MainMenu,footer{visibility:hidden}
section[data-testid='stSidebar'],[data-testid='stSidebarCollapsedControl'],
[data-testid='stExpandSidebarButton'],[data-testid='stSidebarCollapseButton'],
[data-testid='stSidebarNav']{display:none!important}
@media (min-width:769px){
  [data-testid='stTopNav']{position:absolute;left:50%;transform:translateX(-50%);display:flex;
    justify-content:center;align-items:center}
}
[data-testid='stHorizontalBlock']:has([data-testid='stExpander']){align-items:stretch}
[data-testid='stHorizontalBlock']:has([data-testid='stExpander']) [data-testid='stColumn'],
[data-testid='stHorizontalBlock']:has([data-testid='stExpander']) [data-testid='stColumn']>[data-testid='stVerticalBlock'],
[data-testid='stHorizontalBlock']:has([data-testid='stExpander']) [data-testid='stElementContainer'],
[data-testid='stHorizontalBlock']:has([data-testid='stExpander']) [data-testid='stExpander'],
[data-testid='stHorizontalBlock']:has([data-testid='stExpander']) [data-testid='stExpander'] details{height:100%}
.st-key-mobile_nav{display:none}
@media (max-width:768px){.st-key-mobile_nav{display:block}}
h1{letter-spacing:-.035em;margin-bottom:0}
.sf-tag{color:#52606d;margin:.1rem 0 1.2rem;font-size:1.02rem}
.sf-hero{background:linear-gradient(120deg,#0f2a43 0%,#167d8d 100%);color:#fff;border-radius:16px;
  padding:1.1rem 1.4rem;margin:.4rem 0 1rem}
.sf-hero small{opacity:.8;letter-spacing:.08em;text-transform:uppercase;font-size:.72rem}
.sf-hero div{font-size:1.45rem;font-weight:650;letter-spacing:-.02em;margin-top:.15rem}
.sf-hero span{opacity:.85;font-size:.92rem}
div[data-testid='stMetric']{background:#fff;padding:.8rem 1rem;border-radius:12px;border:1px solid #e2e8ef}
div[data-testid='stSliderThumbValue']{display:none}
.sf-hint{color:#167d8d;font-size:.78rem;line-height:1.25;margin:0 0 .35rem}
div[data-testid='stMetricValue']{font-size:1.5rem}
</style>
"""


def _style(fig: go.Figure, title: str, height: int = 380, centered: bool = False) -> go.Figure:
    has_legend = len(fig.data) > 1
    fig.update_layout(
        title=dict(text=title, font=dict(family=FONT, size=15, color=INK), x=0.5 if centered else 0, xanchor="center" if centered else "left"),
        height=height,
        margin=dict(l=280 if has_legend else 8, r=8, t=48, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK_SOFT, size=12),
        legend=dict(orientation="v", xref="container", x=0, xanchor="left", yref="container", y=0.82, yanchor="top", font=dict(color=INK_SOFT)),
        hoverlabel=dict(bgcolor="white", font=dict(family=FONT, color=INK)),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, zeroline=False, ticks="outside", tickcolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor="rgba(0,0,0,0)")
    return fig


_ACRONYMS = {"id": "ID", "ga": "GA", "pa": "(Pa)", "mpa": "(MPa)", "kpa": "(kPa)", "json": "JSON",
             "csv": "CSV", "sha256": "SHA-256", "url": "URL", "doi": "DOI", "er": "εᵣ"}


_KEEP_AS_WRITTEN = {"name", "formulation", "paper_url", "source_url", "url"}


def _capitalize(value):
    """Capitalize the first letter of descriptive text; leave links and numbers alone."""
    if isinstance(value, str) and value[:1].islower() and not value.startswith(("http", "doi", "www")):
        return value[0].upper() + value[1:]
    return value


def _pretty_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn snake_case names into capitalized words and capitalize descriptive cell text."""
    frame = frame.copy()
    for column in frame.columns:
        if column not in _KEEP_AS_WRITTEN and frame[column].dtype == object:
            frame[column] = frame[column].map(_capitalize)
    def label(name: str) -> str:
        words = str(name).replace("-", "_").split("_")
        return " ".join(_ACRONYMS.get(w.lower(), w[:1].upper() + w[1:]) for w in words if w)
    return frame.rename(columns={column: label(column) for column in frame.columns})


def _count_range(label: str, default: tuple[int, int]) -> tuple[int, int]:
    """A 1-4 range slider with the chosen minimum on its left and maximum on its right."""
    low_box, slider_box, high_box = st.columns([1.3, 6, 1.3])
    low, high = slider_box.slider(label, 1, 4, default, label_visibility="collapsed")
    pad = "<div style='padding-top:.45rem;font-size:.8rem;font-weight:600;white-space:nowrap'>"
    low_box.markdown(f"{pad}Min = {low}</div>", unsafe_allow_html=True)
    high_box.markdown(f"{pad}Max = {high}</div>", unsafe_allow_html=True)
    return low, high


def _material_option(row: dict) -> str:
    return row["name"]


# --------------------------------------------------------------------------- charts

def response_chart(evaluated: pd.DataFrame, targets: Targets) -> go.Figure | None:
    """Predicted ΔC/C₀ for the best GA formulation of each of the top matrices."""
    if evaluated.empty:
        return None
    best = (
        evaluated.sort_values("robust_fitness", ascending=False)
        .groupby("matrix_id", sort=False).head(1).head(5)
    )
    pressure = np.linspace(targets.pressure_min_kpa, targets.pressure_max_kpa, 101)
    fig = go.Figure()
    for color, (_, row) in zip(SERIES, best.iterrows()):
        response = planar_capacitance_response(
            pressure * 1000, row.effective_youngs_modulus_mpa * 1e6,
            row.effective_poisson_ratio,
        )
        fig.add_trace(go.Scatter(
            x=pressure, y=response, name=row.formulation, mode="lines",
            line=dict(color=color, width=2.5),
            hovertemplate=f"<b>{row.formulation}</b><br>%{{x:.2f}} kPa<br>ΔC/C₀ = %{{y:.4f}}<extra></extra>",
        ))
    fig.update_xaxes(title="Pressure (kPa)")
    fig.update_yaxes(title="Predicted ΔC/C₀", rangemode="tozero")
    fig.update_layout(hovermode="x unified")
    return _style(fig, "Predicted Response: Best GA Formulation for Each Top Matrix", 400, centered=True)


def ranking_chart(table: pd.DataFrame) -> go.Figure:
    top = table.head(10).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top.robust_fitness, y=top.formulation, orientation="h",
        marker=dict(color=SERIES[0], cornerradius=4),
        hovertemplate="<b>%{y}</b><br>Robust score %{x:.5f}<extra></extra>",
    ))
    fig.update_xaxes(title="Robust Score", rangemode="tozero")
    fig.update_yaxes(showgrid=False, automargin=True, tickfont=dict(size=11))
    return _style(fig, "Top Ten Optimized Formulations", 400, centered=True)


def tradeoff_chart(evaluated: pd.DataFrame) -> go.Figure | None:
    """Permittivity versus stiffness for every feasible formulation the GA evaluated."""
    if evaluated.empty:
        return None
    best = evaluated.loc[evaluated.robust_fitness.idxmax()]
    fig = go.Figure(go.Scatter(
        x=evaluated.effective_youngs_modulus_mpa, y=evaluated.effective_relative_permittivity,
        mode="markers", text=evaluated.formulation,
        marker=dict(
            size=11, color=evaluated.robust_fitness, colorscale=BLUE_RAMP,
            line=dict(color="white", width=1.5),
            colorbar=dict(title="Robust Score", thickness=10, len=0.8),
        ),
        hovertemplate="<b>%{text}</b><br>E = %{x:.2f} MPa<br>εᵣ = %{y:.2f}<br>"
                      "Robust score %{marker.color:.5f}<extra></extra>",
    ))
    fig.add_annotation(
        x=np.log10(best.effective_youngs_modulus_mpa), y=best.effective_relative_permittivity,
        text="Best", showarrow=True, arrowhead=0, ax=-40, ay=-30,
        font=dict(color=INK, size=11), arrowcolor=INK_SOFT,
    )
    fig.update_xaxes(title="Effective Modulus (MPa)", type="log")
    fig.update_yaxes(title="Effective Permittivity (εᵣ)")
    return _style(fig, "Trade-Off Map: Permittivity vs Stiffness (Formulations Evaluated by the GA)", 400, centered=True)


def convergence_chart(history: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for column, name, color in [
        ("best_robust_fitness", "Best in Generation", SERIES[0]),
        ("mean_feasible_robust_fitness", "Mean of Feasible", SERIES[1]),
    ]:
        fig.add_trace(go.Scatter(
            x=history.generation, y=history[column], name=name, mode="lines",
            line=dict(color=color, width=2.5),
        ))
    fig.update_xaxes(title="Generation")
    fig.update_yaxes(title="Robust Score")
    fig.update_layout(hovermode="x unified")
    return _style(fig, "GA convergence", 320)


# --------------------------------------------------------------------------- results

def show_composite_results(
    result: CompositeOptimizationResult,
    database: MaterialDatabase,
    targets: Targets,
) -> None:
    if result.table.empty:
        st.warning(
            "No formulation stays below 10% strain under the robustness stress test for this range. "
            "Lower the maximum pressure or allow stiffer matrices."
        )
        return
    best = result.table.iloc[0]
    margin = 10 - best.worst_case_maximum_strain_percent
    st.markdown(
        f"<div class='sf-hero'><small>Best Formulation · {targets.pressure_min_kpa:g}–"
        f"{targets.pressure_max_kpa:g} kPa</small><div>{best.formulation}</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    a, b, c, d = st.columns(4)
    a.metric("Robust Score", f"{best.robust_fitness:.5f}",
             help="Conservative εᵣ × interval-average ΔC/C₀ sensitivity (per kPa), proportional to absolute capacitance sensitivity for fixed geometry.")
    b.metric("Effective εᵣ", f"{best.effective_relative_permittivity:.2f}")
    c.metric("Effective Modulus", f"{best.effective_youngs_modulus_mpa:.2f} MPa")
    d.metric("Strain Margin", f"{margin:.1f} pts",
             help="10% strain limit minus the worst-case strain at the maximum pressure.")

    chart = response_chart(result.evaluated, targets)
    if chart is not None:
        st.plotly_chart(chart, width="stretch")
    left, right = st.columns(2)
    left.plotly_chart(ranking_chart(result.table), width="stretch")
    trade_off = tradeoff_chart(result.evaluated)
    if trade_off is not None:
        right.plotly_chart(trade_off, width="stretch")

    tab_table, tab_inputs, tab_ga, tab_plan = st.tabs(
        ["Ranked Formulations", "Material Inputs", "Optimizer", "Experiment Plan"]
    )
    with tab_table:
        visible = result.table[[
            "rank", "formulation", "robust_fitness",
            "effective_relative_permittivity", "effective_youngs_modulus_mpa",
            "total_filler_vol_percent", "worst_case_maximum_strain_percent",
        ]]
        st.dataframe(
            visible, hide_index=True, width="stretch",
            column_config={
                "rank": "Rank", "formulation": "Formulation",
                "robust_fitness": st.column_config.NumberColumn("Robust Score", format="%.5f"),
                "effective_relative_permittivity": st.column_config.NumberColumn("εᵣ", format="%.2f"),
                "effective_youngs_modulus_mpa": st.column_config.NumberColumn("E (MPa)", format="%.2f"),
                "total_filler_vol_percent": st.column_config.NumberColumn("Filler (vol%)", format="%.1f"),
                "worst_case_maximum_strain_percent": st.column_config.NumberColumn("Worst strain (%)", format="%.2f"),
            },
        )
    with tab_inputs:
        ids = result.table.candidate_id.tolist()
        by_id = result.table.set_index("candidate_id")
        selected_id = st.selectbox(
            "Formulation", ids,
            format_func=lambda value: f"#{ids.index(value) + 1} · {by_id.loc[value, 'formulation']}",
        )
        candidate = by_id.loc[selected_id]
        material_ids = [*json.loads(candidate["matrix_components_json"]), *json.loads(candidate["components_json"])]
        records = [r for r in database.material_candidates if r["material_id"] in material_ids]
        st.dataframe(
            _material_frame(records, [
                "name", "role", "formulation_or_phase", "youngs_modulus_kpa", "poisson_ratio",
                "relative_permittivity", "density_kg_m3", "limitations",
            ]),
            hide_index=True, width="stretch",
        )
    with tab_ga:
        report = result.report
        x, y, z = st.columns(3)
        x.metric("Formulations Scored", report["unique_formulations_evaluated"])
        y.metric("Strain-Feasible", report["feasible_formulations_evaluated"])
        z.metric("Population × Iterations",
                 f"{report['settings']['population_size']} × {report['settings']['generations']}")
        st.plotly_chart(convergence_chart(result.history), width="stretch")
    with tab_plan:
        plan = build_experiment_plan(result, database, targets)
        st.dataframe(
            plan.table, hide_index=True, width="stretch",
            column_config={
                "experiment_id": "ID", "purpose": "Purpose", "formulation": "Formulation",
                "filler_vol_percent": st.column_config.NumberColumn("Filler (vol%)", format="%.2f"),
                "predicted_relative_permittivity": st.column_config.NumberColumn("Predicted εᵣ", format="%.2f"),
                "predicted_youngs_modulus_kpa": st.column_config.NumberColumn("Predicted E (kPa)", format="%.4g"),
                "predicted_sensitivity_per_kpa": st.column_config.NumberColumn("Predicted Sensitivity (per kPa)", format="%.5f"),
                "predicted_response_at_pmax": st.column_config.NumberColumn("Predicted ΔC/C₀ at Max Pressure", format="%.4f"),
                "predicted_strain_at_pmax_percent": st.column_config.NumberColumn("Strain at Max Pressure (%)", format="%.2f"),
                "replicates": "Batches",
            },
        )
        st.download_button("Experiment Table (CSV)", plan.table.to_csv(index=False),
                           "experiment_plan.csv", "text/csv", width="stretch")

    d1, d2, d3 = st.columns(3)
    d1.download_button("Ranked Formulations (CSV)", result.table.to_csv(index=False),
                       "ranked_composite_formulations.csv", "text/csv", width="stretch")
    d2.download_button("Predicted Curves (CSV)", result.curves.to_csv(index=False),
                       "predicted_response_curves.csv", "text/csv", width="stretch")
    d3.download_button("Report (JSON)", json.dumps(result.report, indent=2),
                       "composite_optimization_report.json", "application/json", width="stretch")


def composite_selector(database: MaterialDatabase) -> None:
    matrix_labels = {
        row["material_id"]: _material_option(row)
        for row in database.material_candidates if row["role"] == "matrix"
    }
    filler_labels = {
        row["material_id"]: _material_option(row)
        for row in database.material_candidates if row["role"] == "filler"
    }
    with st.container(border=True):
        st.markdown("**Pressure Range**")
        p1, p2 = st.columns(2)
        pmin = p1.number_input("Minimum Pressure (kPa)", min_value=0.0, value=0.0)
        pmax = p2.number_input("Maximum Pressure (kPa)", min_value=0.001, value=10.0)
    with st.expander("Genetic Algorithm (GA) Properties", expanded=False):
        g1, g2 = st.columns(2)
        population_size = g1.number_input("GA Population Size", min_value=2, max_value=200, value=20, step=1)
        g1.markdown("<div class='sf-hint'>Candidate formulations tested in each round.</div>", unsafe_allow_html=True)
        generation_count = g2.number_input("GA Iteration Count", min_value=10, max_value=300, value=50, step=1)
        g2.markdown("<div class='sf-hint'>Rounds of selection and mutation; more rounds refine the result.</div>", unsafe_allow_html=True)
    left, right = st.columns(2)
    with left, st.expander("Matrix Details", expanded=True):
        matrix_range = _count_range("Matrix Count", (1, 1))
        selected_matrices = st.multiselect(
            "Allowed Matrices", list(matrix_labels), default=list(matrix_labels),
            format_func=matrix_labels.get,
        )
    with right, st.expander("Filler Details", expanded=True):
        label_column, option_column = st.columns([1, 2.6], vertical_alignment="center")
        label_column.markdown("**Filler Loading**")
        loading = option_column.radio(
            "Filler Loading", [DILUTE, CONCENTRATED], horizontal=True, label_visibility="collapsed"
        )
        filler_range = _count_range("Filler Count", (1, 4))
        selected_fillers = st.multiselect(
            "Allowed Fillers", list(filler_labels), default=list(filler_labels),
            format_func=filler_labels.get,
        )
    submitted = st.button("Optimize Formulations", type="primary")
    selected = [*selected_matrices, *selected_fillers]
    targets = Targets(pmin, pmax)
    settings = GeneticOptimizationSettings(
        population_size=int(population_size), generations=int(generation_count),
        minimum_active_matrices=matrix_range[0], maximum_active_matrices=matrix_range[1],
        minimum_active_fillers=filler_range[0], maximum_active_fillers=filler_range[1],
        **(CONCENTRATED_SETTINGS if loading == CONCENTRATED else {}),
    )
    signature = json.dumps({
        "database": database.fingerprint(), "selected": selected, "targets": asdict(targets),
        "settings": asdict(settings),
    }, sort_keys=True)
    if submitted:
        st.session_state.pop("composite_result", None)
        try:
            st.session_state.composite_result = optimize_composites(
                database, selected, targets, settings=settings
            )
            st.session_state.composite_signature = signature
        except ValueError as exc:
            st.error(str(exc))
    if "composite_result" in st.session_state:
        if st.session_state.get("composite_signature") != signature:
            st.info("The pressure range or allowed materials changed. Run the screening again.")
        else:
            show_composite_results(
                st.session_state.composite_result, database, targets,
            )
    else:
        st.caption("Set a pressure range and press **Optimize Formulations**.")


# --------------------------------------------------------------------------- database

_TABLE_COLUMNS = [
    "material_id", "name", "formulation_or_phase", "youngs_modulus_kpa", "poisson_ratio",
    "relative_permittivity", "density_kg_m3", "limitations",
]
_COLUMN_CONFIG = {
    "Youngs Modulus (kPa)": st.column_config.NumberColumn(format="%.4g"),
    "Poisson Ratio": st.column_config.NumberColumn(format="%.3f"),
    "Relative Permittivity": st.column_config.NumberColumn(format="%.2f"),
    "Density Kg M3": st.column_config.NumberColumn("Density (kg/m³)", format="%.0f"),
}


def _material_frame(rows: list[dict], columns: list[str]) -> pd.DataFrame:
    """Material records for display, with Young's modulus converted from Pa to kPa."""
    if not rows:
        return _pretty_columns(pd.DataFrame(columns=columns))
    frame = pd.DataFrame(rows)
    frame["youngs_modulus_kpa"] = frame["youngs_modulus_pa"].astype(float) / 1000
    return _pretty_columns(frame[columns])


def _material_section(database: MaterialDatabase, role: str, heading: str, singular: str) -> None:
    rows = [row for row in database.material_candidates if row["role"] == role]
    st.subheader(heading)
    event = st.dataframe(
        _material_frame(rows, _TABLE_COLUMNS),
        hide_index=True, width="stretch", column_config=_COLUMN_CONFIG,
        on_select="rerun", selection_mode="multi-row", key=f"select_{role}",
    )
    chosen = [rows[index]["material_id"] for index in event.selection.rows]
    if st.button(f"Delete Selected ({len(chosen)})", key=f"delete_{role}", disabled=not chosen):
        try:
            counts = delete_materials(chosen)
        except MaterialStoreError as exc:
            st.error(str(exc))
        else:
            st.session_state.database_message = (
                f"Deleted {counts['materials']} {role}(s)."
            )
            st.rerun()
    with st.expander(f"Add {singular}"):
        with st.form(f"add_{role}", clear_on_submit=True):
            name = st.text_input("Name")
            phase = st.text_input("Formulation Or Phase")
            c1, c2 = st.columns(2)
            modulus = c1.number_input("Young's Modulus (kPa)", min_value=0.0, value=None, format="%g")
            poisson = c2.number_input("Poisson Ratio", min_value=-0.99, max_value=0.49, value=None, format="%.3f")
            c3, c4 = st.columns(2)
            permittivity = c3.number_input("Relative Permittivity", min_value=0.0, value=None, format="%g")
            density = c4.number_input("Density (kg/m³)", min_value=0.0, value=None, format="%g")
            limitations = st.text_area("Limitations")
            submitted = st.form_submit_button(f"Add {singular}", type="primary")
        if submitted:
            try:
                new_id = add_material(role, dict(
                    name=name, formulation_or_phase=phase,
                    youngs_modulus_pa=None if modulus is None else modulus * 1000,
                    poisson_ratio=poisson, relative_permittivity=permittivity,
                    density_kg_m3=density, limitations=limitations,
                ))
            except MaterialStoreError as exc:
                st.error(str(exc))
            else:
                st.session_state.database_message = f"Added {singular.lower()} {new_id}: {name.strip()}."
                st.rerun()


def database_page(database: MaterialDatabase) -> None:
    message = st.session_state.pop("database_message", None)
    if message:
        st.success(message)
    _material_section(database, "matrix", "Available Matrices", "Matrix")
    _material_section(database, "filler", "Available Fillers", "Filler")
    if configured_store() is not None:
        st.caption("Materials are stored in Supabase. Additions and deletions are saved in the change history.")
    else:
        st.caption("New records get the next free ID. A backup of the database is saved before every change.")


def optimizer_page(database: MaterialDatabase) -> None:
    st.title("Composite Optimizer")
    st.markdown(
        "<div class='sf-tag'>Find the matrix, fillers and loadings that maximize robust capacitance "
        "sensitivity over your pressure range.</div>",
        unsafe_allow_html=True,
    )
    roles = {row["role"] for row in database.material_candidates}
    if not {"matrix", "filler"} <= roles:
        st.info("Add at least one matrix and one filler in the Database tab to start optimizing.")
        return
    composite_selector(database)


def materials_page(database: MaterialDatabase) -> None:
    st.title("Database")
    st.markdown(
        "<div class='sf-tag'>The matrices and fillers the optimizer can choose from, with the "
        "properties it uses.</div>",
        unsafe_allow_html=True,
    )
    database_page(database)


def run() -> None:
    st.set_page_config(
        page_title="Sensora", page_icon=str(ICON), layout="wide", initial_sidebar_state="collapsed"
    )
    st.markdown(CSS, unsafe_allow_html=True)
    st.logo(str(LOGO), size="large")
    try:
        database = read_database()
    except (ValueError, sqlite3.Error) as exc:
        st.error(str(exc))
        return
    pages = [
        st.Page(lambda: optimizer_page(database), title="Composite Optimizer",
                url_path="optimizer", default=True),
        st.Page(lambda: materials_page(database), title="Database", url_path="database"),
        st.Page(lambda: assistant_page(database), title="AI Assistant", url_path="assistant"),
    ]
    current = st.navigation(pages, position="top")
    # The top nav is not shown on narrow screens, so offer the same links there.
    with st.container(key="mobile_nav"):
        for column, page in zip(st.columns(len(pages)), pages):
            column.page_link(page)
    current.run()
