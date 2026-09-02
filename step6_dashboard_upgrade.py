"""Dashboard upgrade: adjusted models, robustness, and research exports.
It reads both analysis outputs and returns a callback-free Dash layout that can be added as a new tab to the existing app.
Expected files:
  results/statistical_analysis/regulation_model_summary.csv
  results/statistical_analysis/model_sample_sizes.csv
  results/robustness_analysis/robustness_summary.csv
  results/robustness_analysis/leave_one_region_out.csv
  results/robustness_analysis/influence_sensitivity.csv
  results/robustness_analysis/country_bootstrap_summary.csv
  results/robustness_analysis/ihme_uncertainty_summary.csv
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dash_table, dcc, html

ROOT = Path(__file__).resolve().parent
STEP4_DIR = ROOT / "results" / "statistical_analysis"
STEP5_DIR = ROOT / "results" / "robustness_analysis"
COLORS = {
    "bg": "#F6F8FB",
    "card": "#FFFFFF",
    "text": "#1F2937",
    "muted": "#6B7280",
    "grid": "#E5E7EB",
    "blue": "#2563EB",
    "red": "#EF4444",
    "teal": "#0F766E",
    "orange": "#C2410C",
    "light_blue": "#EFF6FF",
    "light_orange": "#FFF7ED",
    "light_gray": "#F9FAFB",
}
MODEL_ORDER = ["M0_unadjusted", "M1_development", "M2_primary", "M3_health_system", "M4_environment", "M5_industrial"]
MODEL_LABELS = {
    "M0_unadjusted": "M0 Unadjusted",
    "M1_development": "M1 GDP adjusted",
    "M2_primary": "M2 Primary adjusted",
    "M3_health_system": "M3 Health-system sensitivity",
    "M4_environment": "M4 Environmental sensitivity",
    "M5_industrial": "M5 Industrial sensitivity",
}
ROBUSTNESS_ORDER = [
    "Primary model, HC3 SE",
    "Country bootstrap, primary model",
    "Primary model, region-clustered SE",
    "Huber robust regression",
    "Quantile regression q=0.50",
    "Gamma GLM with log link",
    "Alternative six-continent region adjustment",
    "Only 2022 primary covariates",
]
CARD_STYLE = {
    "backgroundColor": COLORS["card"],
    "border": f"1px solid {COLORS['grid']}",
    "borderRadius": "16px",
    "boxShadow": "0 8px 24px rgba(15, 23, 42, 0.06)",
    "padding": "16px",
    "boxSizing": "border-box",
    "minWidth": 0,
}
GRID_STYLE = {
    "display": "grid",
    "gridTemplateColumns": "repeat(auto-fit, minmax(420px, 1fr))",
    "gap": "16px",
    "width": "100%",
}

def _read_csv(path: Path, required: Iterable[str] = ()) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    data = pd.read_csv(path)
    missing = set(required).difference(data.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")
    return data

def load_research_results() -> dict[str, pd.DataFrame]:
    return {
        "models": _read_csv(STEP4_DIR / "regulation_model_summary.csv", ["outcome_label", "model", "n", "percent_difference", "percent_ci_low", "percent_ci_high", "p_value"]),
        "samples": _read_csv(STEP4_DIR / "model_sample_sizes.csv"),
        "robustness": _read_csv(STEP5_DIR / "robustness_summary.csv", ["outcome", "analysis", "percent_difference", "percent_ci_low", "percent_ci_high"]),
        "regions": _read_csv(STEP5_DIR / "leave_one_region_out.csv"),
        "influence": _read_csv(STEP5_DIR / "influence_sensitivity.csv"),
        "bootstrap": _read_csv(STEP5_DIR / "country_bootstrap_summary.csv"),
        "ihme": _read_csv(STEP5_DIR / "ihme_uncertainty_summary.csv"),
    }

def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font={"size": 15, "color": COLORS["muted"]})
    fig.update_layout(
        template="plotly_white", height=430, xaxis={"visible": False}, yaxis={"visible": False},
        paper_bgcolor=COLORS["card"], plot_bgcolor=COLORS["card"], margin={"l": 25, "r": 25, "t": 45, "b": 25},
    )
    return fig

def _theme(fig: go.Figure, height: int = 540) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        font={"family": "Inter, Segoe UI, Arial, sans-serif", "color": COLORS["text"]},
        title={"x": 0.02, "xanchor": "left"},
        hoverlabel={"bgcolor": "white"},
        margin={"l": 80, "r": 35, "t": 80, "b": 65},
    )
    fig.update_xaxes(showgrid=True, gridcolor=COLORS["grid"], zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False, automargin=True)
    return fig

def _effect_text(value: float) -> str:
    return f"{value:+.1f}%"

def _p_text(value: float) -> str:
    if pd.isna(value):
        return "Not calculated"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def adjusted_model_forest(models: pd.DataFrame, outcome_label: str) -> go.Figure:
    if models.empty:
        return empty_figure("Step 4 model summary was not found.")
    subset = models[models["outcome_label"] == outcome_label].copy()
    if subset.empty:
        return empty_figure(f"No Step 4 results found for {outcome_label}.")
    subset["model"] = pd.Categorical(subset["model"], MODEL_ORDER, ordered=True)
    subset = subset.sort_values("model")
    subset["display_model"] = subset["model"].astype(str).map(MODEL_LABELS)
    subset["hover"] = subset.apply(
        lambda row: (
            f"{row['display_model']}<br>n = {int(row['n'])}"
            f"<br>Estimate: {_effect_text(row['percent_difference'])}"
            f"<br>95% CI: {row['percent_ci_low']:.1f}% to {row['percent_ci_high']:.1f}%"
            f"<br>p = {_p_text(row['p_value'])}"
        ), axis=1,
    )
    x = subset["percent_difference"].to_numpy()
    low = subset["percent_ci_low"].to_numpy()
    high = subset["percent_ci_high"].to_numpy()
    colors = [COLORS["red"] if model == "M0_unadjusted" else COLORS["blue"] for model in subset["model"].astype(str)]
    fig = go.Figure(go.Scatter(
        x=x,
        y=subset["display_model"],
        mode="markers",
        marker={"size": 12, "color": colors},
        error_x={"type": "data", "symmetric": False, "array": high - x, "arrayminus": x - low, "thickness": 2, "width": 6},
        text=subset["hover"],
        hovertemplate="%{text}<extra></extra>",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=f"{outcome_label}: Unadjusted and Adjusted Regulation Estimates",
        xaxis_title="Estimated difference: regulated vs not regulated (%)",
        yaxis_title="",
    )
    fig.update_yaxes(autorange="reversed")
    return _theme(fig, 550)


def robustness_forest(robustness: pd.DataFrame, outcome: str, outcome_label: str) -> go.Figure:
    if robustness.empty:
        return empty_figure("Step 5 robustness summary was not found.")
    subset = robustness[(robustness["outcome"] == outcome) & robustness["analysis"].isin(ROBUSTNESS_ORDER)].copy()
    if subset.empty:
        return empty_figure(f"No Step 5 robustness results found for {outcome_label}.")
    subset["analysis"] = pd.Categorical(subset["analysis"], ROBUSTNESS_ORDER, ordered=True)
    subset = subset.sort_values("analysis")
    x = subset["percent_difference"].to_numpy()
    low = subset["percent_ci_low"].to_numpy()
    high = subset["percent_ci_high"].to_numpy()
    subset["hover"] = subset.apply(
        lambda row: (
            f"{row['analysis']}<br>Estimate: {_effect_text(row['percent_difference'])}"
            f"<br>95% CI: {row['percent_ci_low']:.1f}% to {row['percent_ci_high']:.1f}%"
            f"<br>p = {_p_text(row.get('p_value', np.nan))}"
        ), axis=1,
    )
    fig = go.Figure(go.Scatter(
        x=x, y=subset["analysis"].astype(str), mode="markers",
        marker={"size": 11, "color": COLORS["teal"]},
        error_x={"type": "data", "symmetric": False, "array": high - x, "arrayminus": x - low, "thickness": 2, "width": 6},
        text=subset["hover"], hovertemplate="%{text}<extra></extra>",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=f"{outcome_label}: Robustness Across Alternative Analyses",
        xaxis_title="Estimated difference: regulated vs not regulated (%)",
        yaxis_title="",
    )
    fig.update_yaxes(autorange="reversed")
    return _theme(fig, 650)


def leave_one_region_plot(regions: pd.DataFrame) -> go.Figure:
    if regions.empty:
        return empty_figure("Leave-one-region-out results were not found.")
    fig = go.Figure()
    palette = {"daly_rate": COLORS["blue"], "death_rate": COLORS["orange"]}
    labels = {"daly_rate": "DALY Rate", "death_rate": "Death Rate"}
    for outcome in ["daly_rate", "death_rate"]:
        subset = regions[regions["outcome"] == outcome].copy()
        x = subset["percent_difference"].to_numpy()
        low = subset["percent_ci_low"].to_numpy()
        high = subset["percent_ci_high"].to_numpy()
        fig.add_trace(go.Scatter(
            x=x,
            y=subset["excluded_region"],
            mode="markers",
            name=labels[outcome],
            marker={"size": 10, "color": palette[outcome]},
            error_x={"type": "data", "symmetric": False, "array": high - x, "arrayminus": x - low, "thickness": 1.5, "width": 5},
            customdata=np.column_stack([subset["n"], subset["p_value"]]),
            hovertemplate=(
                "Excluded: %{y}<br>Estimate: %{x:.1f}%"
                "<br>n = %{customdata[0]:.0f}<br>p = %{customdata[1]:.3f}<extra></extra>"
            ),
        ))
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Geographic Robustness: Leave One World Bank Region Out",
        xaxis_title="Estimated difference: regulated vs not regulated (%)",
        yaxis_title="Excluded region",
        legend={"orientation": "h", "y": 1.08, "x": 0},
    )
    return _theme(fig, 620)


def influence_range_card(influence: pd.DataFrame) -> html.Div:
    if influence.empty:
        return insight_card("Influence sensitivity", "Influence-sensitivity results were not found.", COLORS["light_gray"])
    bullets = []
    for outcome, label in [("daly_rate", "DALY"), ("death_rate", "Death rate")]:
        subset = influence[(influence["outcome"] == outcome) & (influence["analysis"] == "Leave one influential country out")]
        all_removed = influence[(influence["outcome"] == outcome) & (influence["analysis"] == "All Cook's D flagged countries removed")]
        if not subset.empty:
            minimum = subset["percent_difference"].min()
            maximum = subset["percent_difference"].max()
            text = f"{label}: leave-one-country-out estimates ranged from {minimum:.1f}% to {maximum:.1f}%."
            bullets.append(html.Li(text))
        if not all_removed.empty:
            row = all_removed.iloc[0]
            bullets.append(html.Li(
                f"{label}: removing all flagged countries gave {row['percent_difference']:.1f}% "
                f"(95% CI {row['percent_ci_low']:.1f}% to {row['percent_ci_high']:.1f}%)."
            ))
    return html.Div([
        html.H3("Influential-Country Sensitivity", style={"marginTop": 0}),
        html.Ul(bullets, style={"lineHeight": "1.6"}),
        html.P(
            "Influence flags are diagnostics, not automatic exclusion rules. Countries remain in the primary model unless a data error is identified.",
            style={"color": COLORS["muted"], "fontSize": "13px"},
        ),
    ], style=CARD_STYLE)


def insight_card(title: str, body, background: str = COLORS["light_blue"]) -> html.Div:
    children = [html.H3(title, style={"marginTop": 0, "marginBottom": "8px"})]
    if isinstance(body, str):
        children.append(html.P(body, style={"margin": 0, "lineHeight": "1.55"}))
    else:
        children.append(body)
    return html.Div(children, style={**CARD_STYLE, "backgroundColor": background})


def headline_cards(models: pd.DataFrame, robustness: pd.DataFrame) -> html.Div:
    if models.empty:
        return insight_card("Research results unavailable", "Run statistical_analysis.py before opening this tab.", COLORS["light_orange"])

    def row(outcome_label: str, model: str):
        match = models[(models["outcome_label"] == outcome_label) & (models["model"] == model)]
        return None if match.empty else match.iloc[0]

    daly_m0 = row("DALY Rate", "M0_unadjusted")
    daly_m2 = row("DALY Rate", "M2_primary")
    death_m2 = row("Death Rate", "M2_primary")

    cards = []
    if daly_m0 is not None and daly_m2 is not None:
        cards.append(insight_card(
            "Unadjusted vs Adjusted DALY",
            html.Div([
                html.P(f"Unadjusted: {_effect_text(daly_m0['percent_difference'])} "
                       f"(95% CI {daly_m0['percent_ci_low']:.1f}% to {daly_m0['percent_ci_high']:.1f}%)."),
                html.P(f"Primary adjusted: {_effect_text(daly_m2['percent_difference'])} "
                       f"(95% CI {daly_m2['percent_ci_low']:.1f}% to {daly_m2['percent_ci_high']:.1f}%)."),
                html.Strong("The large unadjusted difference is strongly attenuated after adjustment."),
            ]),
            COLORS["light_orange"],
        ))
    if death_m2 is not None:
        cards.append(insight_card(
            "Adjusted Death-Rate Result",
            f"Primary adjusted estimate: {_effect_text(death_m2['percent_difference'])} "
            f"(95% CI {death_m2['percent_ci_low']:.1f}% to {death_m2['percent_ci_high']:.1f}%). "
            "The interval includes zero.",
        ))
    cards.append(insight_card(
        "Robustness Conclusion",
        "Adjusted estimates remain modestly negative across bootstrap, Huber, quantile, Gamma, regional, and temporal analyses, but all reported intervals include zero.",
    ))
    return html.Div(cards, style=GRID_STYLE)


def compact_table(data: pd.DataFrame, table_id: str, columns: list[str]) -> dash_table.DataTable:
    view = data[columns].copy() if not data.empty else pd.DataFrame(columns=columns)
    for column in view.columns:
        if pd.api.types.is_numeric_dtype(view[column]):
            view[column] = view[column].round(3)
    return dash_table.DataTable(
        id=table_id,
        data=view.to_dict("records"),
        columns=[{"name": column.replace("_", " ").title(), "id": column} for column in view.columns],
        sort_action="native",
        filter_action="native",
        export_format="csv",
        export_headers="display",
        page_size=12,
        style_table={"overflowX": "auto"},
        style_cell={"fontFamily": "Inter, Arial", "fontSize": "12px", "padding": "8px", "textAlign": "center", "minWidth": "95px"},
        style_header={"backgroundColor": COLORS["text"], "color": "white", "fontWeight": "bold"},
        style_data={"backgroundColor": "white", "border": f"1px solid {COLORS['grid']}"},
        style_data_conditional=[
            {"if": {"column_id": "percent_difference"}, "fontWeight": "bold", "backgroundColor": COLORS["light_blue"]},
            {"if": {"filter_query": "{p_value} < 0.05", "column_id": "p_value"}, "backgroundColor": "#FEF2F2"},
        ],
    )


def uncertainty_notice(ihme: pd.DataFrame) -> html.Div:
    unavailable = True
    if not ihme.empty and "available" in ihme.columns:
        available_text = ihme["available"].astype(str).str.lower()
        unavailable = not available_text.isin(["true", "1"]).any()
    if unavailable:
        body = (
            "IHME lower and upper uncertainty bounds were not present in the analytical extract. "
            "Step 5 therefore evaluates sampling and model-specification uncertainty but does not propagate outcome-estimation uncertainty."
        )
        return insight_card("IHME Uncertainty Status", body, COLORS["light_orange"])
    return insight_card(
        "IHME Uncertainty Status",
        "IHME outcome uncertainty was propagated through Monte Carlo simulation. Review the uncertainty summary before publication.",
    )


def downloads_card() -> html.Div:
    files = [
        "results/statistical_analysis/regulation_model_summary.csv",
        "results/statistical_analysis/all_model_coefficients.csv",
        "results/statistical_analysis/model_sample_sizes.csv",
        "results/robustness_analysis/robustness_summary.csv",
        "results/robustness_analysis/leave_one_region_out.csv",
        "results/robustness_analysis/influence_sensitivity.csv",
        "results/robustness_analysis/country_bootstrap_summary.csv",
    ]
    return html.Div([
        html.H3("Research Output Files", style={"marginTop": 0}),
        html.P("Dash tables below can be exported as CSV. The complete generated files remain available in the project folders."),
        html.Ul([html.Li(html.Code(path)) for path in files], style={"lineHeight": "1.6"}),
    ], style=CARD_STYLE)


def research_results_layout() -> html.Div:
    results = load_research_results()
    models = results["models"]
    robustness = results["robustness"]
    regions = results["regions"]
    influence = results["influence"]
    ihme = results["ihme"]

    model_columns = [
        "outcome_label", "model", "n", "percent_difference",
        "percent_ci_low", "percent_ci_high", "p_value", "adjusted_r_squared",
    ]
    model_columns = [column for column in model_columns if column in models.columns]

    robustness_columns = [
        "outcome", "analysis", "n", "percent_difference",
        "percent_ci_low", "percent_ci_high", "p_value",
    ]
    robustness_columns = [column for column in robustness_columns if column in robustness.columns]

    return html.Div([
        html.Div([
            html.H2("Adjusted Analysis and Robustness", style={"marginTop": 0}),
            html.P(
                "This section separates the original unadjusted comparison from the prespecified multivariable model and Step 5 sensitivity analyses.",
                style={"color": COLORS["muted"], "fontSize": "15px"},
            ),
            html.Div(
                "Interpretation: estimates are country-level associations, not causal effects. Binary regulation status does not measure implementation, enforcement, legal thresholds, or adoption timing.",
                style={
                    "padding": "12px 14px", "backgroundColor": COLORS["light_orange"],
                    "border": "1px solid #FED7AA", "borderRadius": "10px",
                    "color": "#7C2D12", "fontStyle": "italic", "lineHeight": "1.5",
                },
            ),
        ], style={**CARD_STYLE, "marginBottom": "16px"}),

        headline_cards(models, robustness),

        html.Div([
            html.Div(dcc.Graph(figure=adjusted_model_forest(models, "DALY Rate"), config={"displayModeBar": False, "responsive": True}), style=CARD_STYLE),
            html.Div(dcc.Graph(figure=adjusted_model_forest(models, "Death Rate"), config={"displayModeBar": False, "responsive": True}), style=CARD_STYLE),
        ], style={**GRID_STYLE, "marginTop": "16px"}),

        html.Div([
            html.Div(dcc.Graph(figure=robustness_forest(robustness, "daly_rate", "DALY Rate"), config={"displayModeBar": False, "responsive": True}), style=CARD_STYLE),
            html.Div(dcc.Graph(figure=robustness_forest(robustness, "death_rate", "Death Rate"), config={"displayModeBar": False, "responsive": True}), style=CARD_STYLE),
        ], style={**GRID_STYLE, "marginTop": "16px"}),

        html.Div(dcc.Graph(figure=leave_one_region_plot(regions), config={"displayModeBar": False, "responsive": True}), style={**CARD_STYLE, "marginTop": "16px"}),

        html.Div([
            influence_range_card(influence),
            uncertainty_notice(ihme),
        ], style={**GRID_STYLE, "marginTop": "16px"}),

        html.Div([
            html.H3("Step 4 Regulation Estimates", style={"marginTop": 0}),
            compact_table(models, "step6-model-table", model_columns),
        ], style={**CARD_STYLE, "marginTop": "16px"}),

        html.Div([
            html.H3("Step 5 Robustness Estimates", style={"marginTop": 0}),
            compact_table(robustness, "step6-robustness-table", robustness_columns),
        ], style={**CARD_STYLE, "marginTop": "16px"}),

        html.Div(downloads_card(), style={"marginTop": "16px"}),
    ])


def research_insight_cards() -> tuple[html.Div, html.Div]:
    """Optional replacements for the two blank cards in overview_layout()."""
    results = load_research_results()
    models = results["models"]
    if models.empty:
        return (
            insight_card("Adjusted results pending", "Run statistical_analysis.py to populate this research insight."),
            insight_card("Robustness results pending", "Run robustness_analysis.py to populate this research insight."),
        )

    daly_unadjusted = models[(models["outcome_label"] == "DALY Rate") & (models["model"] == "M0_unadjusted")]
    daly_adjusted = models[(models["outcome_label"] == "DALY Rate") & (models["model"] == "M2_primary")]
    if not daly_unadjusted.empty and not daly_adjusted.empty:
        u = daly_unadjusted.iloc[0]
        a = daly_adjusted.iloc[0]
        first = insight_card(
            "Adjustment Changes the DALY Interpretation",
            html.Div([
                html.P(f"Unadjusted estimate: {_effect_text(u['percent_difference'])}."),
                html.P(f"Primary adjusted estimate: {_effect_text(a['percent_difference'])}."),
                html.P("The adjusted confidence interval crosses zero; the large unadjusted difference is not an independent adjusted association.\nFor more info, see the \"Adjusted Analysis & Robustness\" tab."),
            ]),
            COLORS["light_orange"],
        )
    else:
        first = insight_card("DALY adjustment", "Required Step 4 rows were not found.")

    second = insight_card(
        "Robustness Summary",
        "Bootstrap, robust, quantile, Gamma, geographic, influence, and temporal analyses support a modest negative but statistically uncertain adjusted association for both outcomes. Check the \"Adjusted Analysis & Robustness\" tab for more info.",
        COLORS["light_blue"],
    )
    return first, second
