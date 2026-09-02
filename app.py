from __future__ import annotations
from pathlib import Path
import sys
from typing import Dict
import numpy as np
import pandas as pd
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from dash import Dash, Input, Output, dash_table, dcc, html
from data_pipeline import PROCESSED_DIR, compute_effect_size, fit_policy_regression, run_group_tests, summarize_by_regulation
from dash.dash_table.Format import Format, Scheme
from step6_dashboard_upgrade import research_results_layout, research_insight_cards

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUTPUT_PATH = PROCESSED_DIR / "lead_policy_expanded.csv"

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def load_or_build_data(path: Path = OUTPUT_PATH) -> pd.DataFrame:
    """Load the processed dataset, building it first when it is absent."""
    if not path.exists():
        raise FileNotFoundError(f"Expanded dataset not found: {path}\nRun \"data_pipeline.py\" and then \"covariate_pipeline.py\" before starting the dashboard.")
    data = pd.read_csv(path)
    required = {
        "country", "iso_code", "regulation", "regulation_label", "region", "daly_rate",  "death_rate", 
        "reg_binary", "wb_region", "log_gdp_per_capita_ppp", "urban_population_pct", "population_age_65plus_pct",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            f"Processed dataset is missing columns {sorted(missing)}. "
            "Run data_pipeline.py and then covariate_pipeline.py to rebuild it."
        )
    return data

df = load_or_build_data()

# -----------------------------------------------------------------------------
# Theme and shared helpers
# -----------------------------------------------------------------------------
COLORS = {
    "bg": "#F6F8FB",
    "card": "#FFFFFF",
    "text": "#1F2937",
    "muted": "#6B7280",
    "grid": "#E5E7EB",
    "yes": "#2563EB",
    "no": "#EF4444",
    "accent": "#14B8A6",
    "warning_bg": "#FFF7ED",
    "warning_border": "#FED7AA",
    "warning_text": "#7C2D12",
}
REGULATION_COLORS = {"Yes": COLORS["yes"], "No": COLORS["no"]}
REGULATION_COLORS_DISPLAY = {
    "Regulated": COLORS["yes"],
    "Not regulated": COLORS["no"],
}
PAGE_STYLE = {
    "backgroundColor": COLORS["bg"],
    "minHeight": "100vh",
    "padding": "20px clamp(12px, 2vw, 28px) 36px",
    "fontFamily": "Inter, Segoe UI, Arial, sans-serif",
    "color": COLORS["text"],
    "boxSizing": "border-box",
    "overflowX": "hidden",
}
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
FULL_WIDTH_STYLE = {**CARD_STYLE, "gridColumn": "1 / -1"}
OUTCOME_OPTIONS = [
    {"label": "DALY Rate", "value": "daly_rate"},
    {"label": "Death Rate", "value": "death_rate"},
]

def apply_figure_theme(fig: go.Figure, title: str | None = None) -> go.Figure:
    if title is not None:
        title_text = title
    elif fig.layout.title and fig.layout.title.text:
        title_text = fig.layout.title.text
    else:
        title_text = ""
    fig.update_layout(
        template="plotly_white",
        title={"text": title_text, "x": 0.02, "xanchor": "left"},
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        font={"family": "Inter, Segoe UI, Arial, sans-serif", "color": COLORS["text"]},
        title_font={"size": 19},
        hoverlabel={"bgcolor": "white", "font_size": 13},
    )
    fig.update_xaxes(showgrid=True, gridcolor=COLORS["grid"], zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=COLORS["grid"], zeroline=False)
    return fig

def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False, font={"size": 16, "color": COLORS["muted"]},
    )
    fig.update_layout(
        template="plotly_white", height=420,
        xaxis={"visible": False}, yaxis={"visible": False},
        paper_bgcolor=COLORS["card"], plot_bgcolor=COLORS["card"],
    )
    return fig

def graph_card(fig: go.Figure, graph_id: str | None = None, min_height: str = "520px") -> html.Div:
    graph_properties = {
        "figure": fig,
        "config": {
            "displayModeBar": False,
            "responsive": True,
        },
        "style": {
            "width": "100%",
            "minHeight": min_height,
        },
    }
    # The ID is added only when it exists.
    if graph_id is not None:
        graph_properties["id"] = graph_id
    return html.Div(
        dcc.Graph(**graph_properties),
        style=CARD_STYLE,
    )

def limitation_note() -> html.Div:
    return html.Div(
        "Important limitation: this dashboard presents cross-sectional associations. "
        "It does not prove that lead paint regulation causes lower health burden, and "
        "the policy indicator does not measure enforcement quality or legal strength.",
        style={
            "padding": "12px 14px", "backgroundColor": COLORS["warning_bg"],
            "border": f"1px solid {COLORS['warning_border']}",
            "borderRadius": "10px", "color": COLORS["warning_text"],
            "fontSize": "14px", "fontStyle": "italic", "lineHeight": "1.5",
            "marginTop": "12px",
        },
    )

# -----------------------------------------------------------------------------
# Overview figures
# -----------------------------------------------------------------------------
def compute_kpis(data: pd.DataFrame) -> Dict[str, str]:
    return {
        "Countries": f"{data['country'].nunique():,}",
        "Regulated": f"{(data['regulation'] == 'Yes').sum():,}",
        "Not regulated": f"{(data['regulation'] == 'No').sum():,}",
        "Avg DALY": f"{data['daly_rate'].mean():,.0f}",
        "Avg Death Rate": f"{data['death_rate'].mean():,.0f}",
    }

def kpi_cards(data: pd.DataFrame) -> html.Div:
    cards = []
    for label, value in compute_kpis(data).items():
        cards.append(html.Div([
            html.Div(label, style={"fontSize": "12px", "color": COLORS["muted"], "marginBottom": "8px"}),
            html.Div(value, style={"fontSize": "28px", "fontWeight": 750}),
        ], style={**CARD_STYLE, "flex": "1 1 190px"}))
    return html.Div(cards, style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "margin": "14px 0 18px"})

def fig_map(data: pd.DataFrame) -> go.Figure:
    clean = data.dropna(subset=["iso_code", "regulation_label", "country"])
    if clean.empty:
        return empty_figure("No data available for map")
    fig = px.choropleth(
        clean, locations="iso_code", locationmode="ISO-3", color="regulation_label",
        hover_name="country", hover_data={"iso_code": False},
        color_discrete_map=REGULATION_COLORS_DISPLAY,
        category_orders={"regulation_label": ["Regulated", "Not regulated"]},
        title="Global Legal Controls on Lead Paint (2023)",
    )
    fig.update_geos(
        projection_type="natural earth", showcountries=True, countrycolor="white",
        showcoastlines=False, showframe=False, bgcolor=COLORS["card"],
    )
    fig.update_traces(marker_line_color="white", marker_line_width=0.4)
    fig.update_layout(height=570, margin={"l": 0, "r": 0, "t": 65, "b": 0})
    return apply_figure_theme(fig)

def fig_country_counts(data: pd.DataFrame) -> go.Figure:
    counts = data["regulation"].value_counts().rename_axis("regulation").reset_index(name="count")
    counts["regulation_label"] = counts["regulation"].map({"Yes": "Regulated", "No": "Not regulated"})
    counts = counts.sort_values("count", ascending=False)
    fig = px.bar(
        counts, x="regulation_label", y="count", color="regulation_label", text="count",
        color_discrete_map=REGULATION_COLORS_DISPLAY,
        title="Countries by Lead Paint Regulation Status",
        labels={"regulation_label": "", "count": "Number of Countries"},
    )
    fig.update_traces(textposition="outside", cliponaxis=False, hovertemplate="%{x}: %{y} countries<extra></extra>")
    fig.update_layout(height=520, showlegend=False, margin={"l": 65, "r": 25, "t": 75, "b": 55})
    return apply_figure_theme(fig)

def regional_regulation_stacked_bar(data: pd.DataFrame) -> go.Figure:
    grouped = data.groupby(["region", "regulation"], observed=True).size().reset_index(name="country_count")
    totals = grouped.groupby("region", observed=True)["country_count"].sum().rename("region_total")
    grouped = grouped.merge(totals, on="region")
    grouped["percent"] = grouped["country_count"] / grouped["region_total"] * 100
    grouped["regulation_label"] = grouped["regulation"].map({"Yes": "Regulated", "No": "Not regulated"})
    order = (
        grouped.loc[grouped["regulation"] == "Yes", ["region", "percent"]]
        .set_index("region")["percent"]
        .reindex(grouped["region"].unique(), fill_value=0)
        .sort_values(ascending=False).index.tolist()
    )
    grouped["text_label"] = grouped["percent"].map(lambda value: f"{value:.1f}%" if value >= 7 else "")
    fig = px.bar(
        grouped, y="region", x="percent", color="regulation_label", orientation="h",
        color_discrete_map=REGULATION_COLORS_DISPLAY, text="text_label",
        category_orders={"regulation_label": ["Regulated", "Not regulated"]},
        title="Lead Paint Regulation by Region",
        labels={"region": "", "percent": "Percentage of Countries", "regulation_label": "Regulation"},
    )
    fig.update_layout(barmode="stack", height=520, margin={"l": 125, "r": 25, "t": 80, "b": 55})
    fig.update_traces(textposition="inside", insidetextanchor="middle", hovertemplate="<b>%{y}</b><br>%{fullData.name}: %{x:.1f}%<extra></extra>")
    fig.update_xaxes(ticksuffix="%", range=[0, 100])
    fig.update_yaxes(categoryorder="array", categoryarray=order, autorange="reversed")
    return apply_figure_theme(fig)

def daly_death_scatter(data: pd.DataFrame, region: str | None = None) -> go.Figure:
    clean = data.dropna(subset=["daly_rate", "death_rate", "regulation_label", "country"]).copy()
    if region and region != "All":
        clean = clean.loc[clean["region"] == region]
    if clean.empty:
        return empty_figure("No data available for scatterplot")
    correlation = clean[["daly_rate", "death_rate"]].corr().loc["daly_rate", "death_rate"]
    fig = px.scatter(
        clean, x="daly_rate", y="death_rate", color="regulation_label", hover_name="country",
        color_discrete_map=REGULATION_COLORS_DISPLAY, trendline="ols",
        title=f"DALY Rate vs Death Rate<br><sup>Pearson correlation = {correlation:.3f}</sup>",
        labels={"daly_rate": "DALY Rate", "death_rate": "Death Rate", "regulation_label": "Regulation"},
    )
    fig.update_traces(marker={"size": 9, "opacity": 0.75})
    fig.update_layout(height=560, margin={"l": 70, "r": 25, "t": 95, "b": 60})
    return apply_figure_theme(fig)

def ranked_lollipop(data: pd.DataFrame, metric: str, top_n: int = 15) -> go.Figure:
    labels = {"daly_rate": "DALY Rate", "death_rate": "Death Rate"}
    clean = data.dropna(subset=[metric, "country", "regulation", "regulation_label"]).copy()
    if clean.empty:
        return empty_figure("No data available for ranking")
    clean = clean.nlargest(top_n, metric).sort_values(metric)
    clean["value_label"] = clean[metric].map(
        lambda value: f"{value / 1000:.1f}k" if metric == "daly_rate" and value >= 1000 else f"{value:.1f}"
    )
    fig = go.Figure()
    for _, row in clean.iterrows():
        fig.add_trace(go.Scatter(
            x=[0, row[metric]], y=[row["country"], row["country"]], mode="lines",
            line={"color": "#D1D5DB", "width": 2}, showlegend=False, hoverinfo="skip",
        ))
    for status in ["No", "Yes"]:
        subset = clean.loc[clean["regulation"] == status]
        if subset.empty:
            continue
        display = "Regulated" if status == "Yes" else "Not regulated"
        custom = np.column_stack([subset["country"], subset["regulation_label"], subset[metric].round(2)])
        fig.add_trace(go.Scatter(
            x=subset[metric], y=subset["country"], mode="markers+text",
            marker={"size": 11, "color": REGULATION_COLORS_DISPLAY[display], "line": {"width": 1, "color": "white"}},
            text=subset["value_label"], textposition="middle right", name=display,
            customdata=custom,
            hovertemplate=f"<b>%{{customdata[0]}}</b><br>Regulation: %{{customdata[1]}}<br>{labels[metric]}: %{{customdata[2]}}<extra></extra>",
            cliponaxis=False,
        ))
    max_value = clean[metric].max()
    fig.update_layout(
        title={"text": f"Top {top_n} Countries by {labels[metric]} (Global)", "x": 0.02},
        xaxis_title=labels[metric], yaxis_title="Country", height=650,
        legend={"orientation": "h", "y": 1.03, "x": 0},
        margin={"l": 165, "r": 90, "t": 105, "b": 60},
    )
    fig.update_xaxes(range=[0, max_value * 1.20], nticks=6, tickangle=0)
    fig.update_yaxes(automargin=True)
    return apply_figure_theme(fig)

def make_binned_scale(start_color: str, end_color: str, bins: int = 5):
    sampled = sample_colorscale([start_color, end_color], [i / (bins - 1) for i in range(bins)])
    scale = []
    for index, color in enumerate(sampled):
        scale.extend([[index / bins, color], [(index + 1) / bins, color]])
    return scale

def regional_indicator_heatmap(data: pd.DataFrame) -> go.Figure:
    clean = data.dropna(subset=["region", "regulation", "daly_rate", "death_rate"])
    summary = clean.groupby("region", observed=True).agg(
        percent_regulated=("regulation", lambda x: (x == "Yes").mean() * 100),
        avg_daly_rate=("daly_rate", "mean"),
        avg_death_rate=("death_rate", "mean"),
    ).reset_index().sort_values("percent_regulated", ascending=False)
    labels = {
        "percent_regulated": "% Countries With Regulation",
        "avg_daly_rate": "Average DALY Rate",
        "avg_death_rate": "Average Death Rate",
    }
    long = summary.melt(id_vars="region", var_name="indicator", value_name="actual_value")
    long["indicator_label"] = long["indicator"].map(labels)
    long["normalized_value"] = long.groupby("indicator")["actual_value"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
    )
    long["text_value"] = long.apply(
        lambda row: f"{row['actual_value']:.1f}%" if row["indicator"] == "percent_regulated" else f"{row['actual_value']:.1f}", axis=1
    )
    z = long.pivot(index="region", columns="indicator_label", values="normalized_value")
    text = long.pivot(index="region", columns="indicator_label", values="text_value").reindex(index=z.index, columns=z.columns)
    fig = px.imshow(
        z, text_auto=False, color_continuous_scale=make_binned_scale(COLORS["yes"], COLORS["no"]),
        aspect="auto", title="Regional Summary: Regulation and Lead Burden",
        labels={"color": "Normalized Value"},
    )
    fig.update_traces(text=text.values, texttemplate="%{text}", hovertemplate="Region: %{y}<br>Indicator: %{x}<br>Normalized value: %{z:.2f}<extra></extra>")
    fig.update_layout(height=520, margin={"l": 125, "r": 35, "t": 80, "b": 85})
    return apply_figure_theme(fig)

# -----------------------------------------------------------------------------
# Statistical figures and tables
# -----------------------------------------------------------------------------
def make_data_quality_card(data: pd.DataFrame) -> html.Div:
    return html.Div([
        html.H3("Data Quality & Preprocessing Summary", style={"marginTop": 0}),
        html.Ul([
            html.Li(f"Final merged dataset: {len(data):,} country-level records."),
            html.Li(f"Unique countries: {data['country'].nunique():,}."),
            html.Li(f"Countries with regulation: {(data['regulation'] == 'Yes').sum():,}."),
            html.Li(f"Countries without regulation: {(data['regulation'] == 'No').sum():,}."),
            html.Li(f"Missing DALY-rate values: {data['daly_rate'].isna().sum():,}."),
            html.Li(f"Missing death-rate values: {data['death_rate'].isna().sum():,}."),
            html.Li("Positive outcomes are log-transformed for regression modeling."),
        ])
    ], style={**CARD_STYLE, "marginTop": "16px", "lineHeight": "1.55"})

def make_boxplot(data: pd.DataFrame, outcome: str) -> go.Figure:
    title = "DALY Rate" if outcome == "daly_rate" else "Death Rate"
    clean = data.dropna(subset=[outcome, "regulation_label"])
    fig = px.box(
        clean, x="regulation_label", y=outcome, color="regulation_label", points="outliers",
        color_discrete_map=REGULATION_COLORS_DISPLAY,
        category_orders={"regulation_label": ["Not regulated", "Regulated"]},
        title=f"{title} Distribution by Regulation Status",
        labels={"regulation_label": "", outcome: title},
    )
    fig.update_traces(marker={"size": 5, "opacity": 0.55}, boxmean=True)
    fig.update_layout(height=520, showlegend=False, margin={"l": 70, "r": 30, "t": 75, "b": 60})
    return apply_figure_theme(fig)

def make_correlation_heatmap(data: pd.DataFrame) -> go.Figure:
    corr = data[["daly_rate", "death_rate"]].corr()
    corr.index = ["DALY Rate", "Death Rate"]
    corr.columns = ["DALY Rate", "Death Rate"]
    fig = px.imshow(
        corr, text_auto=".3f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Correlation Heatmap: Lead Burden Indicators", labels={"color": "Correlation"},
    )
    fig.update_layout(height=450, margin={"l": 70, "r": 30, "t": 75, "b": 55})
    return apply_figure_theme(fig)

def compact_datatable(table_df: pd.DataFrame, table_id: str) -> dash_table.DataTable:
    columns = []
    for column in table_df.columns:
        if column == "Regulation":
            columns.append({"name": column, "id": column, "type": "text"})
        elif column == "n":
            columns.append({
                "name": column,
                "id": column,
                "type": "numeric",
                "format": Format(precision=0, scheme=Scheme.fixed, group=True),
            })
        elif column == "Mean % Difference":
            columns.append({
                "name": column,
                "id": column,
                "type": "numeric",
                "format": Format(precision=1, scheme=Scheme.fixed, sign="+"),
            })
        else:
            columns.append({"name": column, "id": column, "type": "numeric", "format": Format(precision=1, scheme=Scheme.fixed, group=True)})
    return dash_table.DataTable(
        id=table_id, data=table_df.to_dict("records"), columns=columns, sort_action="native", 
        export_format="csv", export_headers="display", page_action="none",
        style_table={"overflowX": "auto", "width": "100%", "borderRadius": "10px"},
        style_cell={
            "fontFamily": "Inter, Segoe UI, Arial, sans-serif", "fontSize": "12px",
            "padding": "9px", "textAlign": "center", "minWidth": "84px", "whiteSpace": "normal",
        },
        style_cell_conditional=[
            {"if": {"column_id": "Regulation",}, "textAlign": "left", "fontWeight": "600", "minWidth": "130px"},
            {"if": {"column_id": "Mean % Difference",}, "minWidth": "125px"},
        ],
        style_header={
            "backgroundColor": COLORS["text"], "color": "white", "fontWeight": "bold", "textAlign": "center", "border": "none"
        },
        style_data={"backgroundColor": "white", "color": COLORS["text"], "border": f"1px solid {COLORS['grid']}"},
        style_data_conditional=[
            {
                "if": {"filter_query": '{Regulation} = "Not regulated"',},
                "backgroundColor": "#FEF2F2",
            },
            {
                "if": {"filter_query": '{Regulation} = "Regulated"',},
                "backgroundColor": "#EFF6FF",
            },
            {
                "if": {"column_id": "Mean",},
                "fontWeight": "bold",
            },
            {
                "if": {"column_id": "Mean % Difference",},
                "backgroundColor": "#F3F4F6",
                "fontWeight": "bold",
            },
            {
                "if": {"filter_query": "{Mean % Difference} < 0", "column_id": "Mean % Difference"},
                "color": "#0F766E",
            },
            {
                "if": {"filter_query": "{Mean % Difference} > 0", "column_id": "Mean % Difference"},
                "color": "#B45309",
            },
        ],
        tooltip_header={
            "n": "Number of countries with a nonmissing value for this outcome.",
            "Mean": "Arithmetic mean of the observed country-level rate.",
            "Median": "Middle value of the observed country-level distribution.",
            "Std": "Standard deviation of the observed country-level rate.",
            "Q1": "25th percentile.",
            "Q3": "75th percentile.",
            "IQR": "Interquartile range, calculated as Q3 minus Q1.",
            "Mean % Difference": "Unadjusted percentage difference in the group mean relative to the Not-regulated group.",
        },
        tooltip_delay=0,
        tooltip_duration=None,
    )

def make_summary_tables(data: pd.DataFrame) -> html.Div:
    summary = summarize_by_regulation(data)
    summary["Regulation"] = summary["regulation"].map({"Yes": "Regulated", "No": "Not regulated"})
    daly = summary[["Regulation", "daly_n", "daly_mean", "daly_median", "daly_std", "daly_q1", "daly_q3", "daly_iqr"]].rename(columns={
        "daly_n": "n", "daly_mean": "Mean", "daly_median": "Median", "daly_std": "Std",
        "daly_q1": "Q1", "daly_q3": "Q3", "daly_iqr": "IQR"
    })
    death = summary[["Regulation", "death_n", "death_mean", "death_median", "death_std", "death_q1", "death_q3", "death_iqr"]].rename(columns={
        "death_n": "n", "death_mean": "Mean", "death_median": "Median", "death_std": "Std",
        "death_q1": "Q1", "death_q3": "Q3", "death_iqr": "IQR"
    })
    return html.Div([
        html.H3("Summary Statistics by Regulation Group"),
        html.P("These descriptive summaries compare observed country-level rates by regulation status without adjustment for economic, demographic, urbanization, or regional differences."),
        html.Div([
            html.Div([html.H4("Unadjusted DALY Rate Summary"), compact_datatable(daly, "daly-summary-table")], style=CARD_STYLE),
            html.Div([html.H4("Unadjusted Death Rate Summary"), compact_datatable(death, "death-summary-table")], style=CARD_STYLE),
        ], style=GRID_STYLE),
    ], style={"marginTop": "20px"})

def _series_value(series, name: str, model) -> float:
    if hasattr(series, "index") and name in series.index:
        return float(series[name])
    names = list(model.model.exog_names)
    return float(np.asarray(series)[names.index(name)])

def _confidence_interval(model, name: str) -> tuple[float, float]:
    intervals = model.conf_int()
    if hasattr(intervals, "index") and name in intervals.index:
        values = intervals.loc[name]
    else:
        values = np.asarray(intervals)[list(model.model.exog_names).index(name)]
    return float(values[0]), float(values[1])

def make_regression_coefficient_plot(model) -> go.Figure:
    coef = _series_value(model.params, "reg_binary", model)
    low, high = _confidence_interval(model, "reg_binary")
    fig = go.Figure(go.Scatter(
        x=[coef], y=["Regulated vs Not regulated"], mode="markers",
        marker={"size": 14, "color": COLORS["accent"]},
        error_x={"type": "data", "symmetric": False, "array": [high - coef], "arrayminus": [coef - low], "thickness": 2, "width": 8},
        name="Coefficient",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(title="Regulation Coefficient with 95% Confidence Interval", xaxis_title="Coefficient on Log Outcome", height=420, margin={"l": 125, "r": 35, "t": 75, "b": 60})
    return apply_figure_theme(fig)

def make_residuals_plot(model) -> go.Figure:
    fig = px.scatter(x=model.fittedvalues, y=model.resid, title="Residuals vs Fitted Values", labels={"x": "Fitted Values (Log Scale)", "y": "Residuals"})
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_traces(marker={"size": 8, "opacity": 0.72, "color": COLORS["yes"]})
    fig.update_layout(height=450, margin={"l": 70, "r": 30, "t": 75, "b": 60})
    return apply_figure_theme(fig)

def make_qq_plot(model) -> go.Figure:
    theoretical, sample = stats.probplot(model.resid, dist="norm", fit=False)
    slope, intercept, _ = stats.probplot(model.resid, dist="norm", fit=True)[1]
    line_x = np.array([min(theoretical), max(theoretical)])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=theoretical, y=sample, mode="markers", marker={"size": 8, "color": COLORS["no"], "opacity": 0.75}, name="Residual quantiles"))
    fig.add_trace(go.Scatter(x=line_x, y=intercept + slope * line_x, mode="lines", line={"color": "gray", "dash": "dash"}, name="Normal reference"))
    fig.update_layout(title="Q-Q Plot of Regression Residuals", xaxis_title="Theoretical Normal Quantiles", yaxis_title="Sample Residual Quantiles", height=450, margin={"l": 70, "r": 30, "t": 75, "b": 60})
    return apply_figure_theme(fig)

# -----------------------------------------------------------------------------
# Layouts
# -----------------------------------------------------------------------------
def overview_layout(data: pd.DataFrame) -> html.Div:
    adjusted_insight, robustness_insight = research_insight_cards()
    return html.Div([
        kpi_cards(data),
        html.Div([
            html.Div(dcc.Graph(figure=fig_map(data), config={"displayModeBar": False, "responsive": True}), style=FULL_WIDTH_STYLE),
            graph_card(fig_country_counts(data)),
            adjusted_insight,
            graph_card(regional_regulation_stacked_bar(data)),
            html.Div(dcc.Graph(figure=daly_death_scatter(data), config={"displayModeBar": False, "responsive": True}), style=FULL_WIDTH_STYLE),
            graph_card(ranked_lollipop(data, "daly_rate"), min_height="730px"),
            robustness_insight,
            graph_card(ranked_lollipop(data, "death_rate"), min_height="730px"),
            html.Div(dcc.Graph(figure=regional_indicator_heatmap(data), config={"displayModeBar": False, "responsive": True}), style=FULL_WIDTH_STYLE),
        ], style=GRID_STYLE),
    ])

def hypothesis_layout(data: pd.DataFrame) -> html.Div:
    return html.Div([
        html.H2("Hypothesis (Unadjusted) Testing Achieved?"),
        html.P("Compare 2023 lead-attributable burden between regulated and not-regulated countries."),
        make_data_quality_card(data),
        html.Div([
            html.Label("Select outcome variable:", style={"fontWeight": 650}),
            dcc.Dropdown(id="hypothesis-outcome", options=OUTCOME_OPTIONS, value="daly_rate", clearable=False),
        ], style={**CARD_STYLE, "marginTop": "16px", "maxWidth": "520px"}),
        html.Div(id="test-output", style={"marginTop": "16px"}),
        html.Div([
            html.H4("Why two tests?", style={"marginTop": 0}),
            html.P("Welch's t-test compares means without assuming equal variances. The Mann-Whitney U test provides a non-parametric distributional comparison that is less sensitive to skewness and outliers."),
        ], style={**CARD_STYLE, "marginTop": "16px", "backgroundColor": "#EEF2FF"}),
        html.Div([
            graph_card(go.Figure(), graph_id="hypothesis-boxplot"),
            graph_card(daly_death_scatter(data)),
            graph_card(make_correlation_heatmap(data)),
        ], style={**GRID_STYLE, "marginTop": "16px"}),
        html.Div(id="summary-table-container"),
    ])

def regression_layout() -> html.Div:
    return html.Div([
        html.H2("Unadjusted Cross-Sectional Regression"),
        html.P(
            "Baseline model: log(outcome) = intercept + regulation status. HC3 heteroskedasticity-robust standard errors are used. "
            "Multivariable results are presented in the 'Adjusted Analysis & Robustness' tab."
        ),
        html.Div([
            html.Label("Select outcome variable:", style={"fontWeight": 650}),
            dcc.Dropdown(id="regression-outcome", options=OUTCOME_OPTIONS, value="daly_rate", clearable=False),
        ], style={**CARD_STYLE, "marginTop": "16px", "maxWidth": "520px"}),
        html.Div(id="regression-summary", style={**CARD_STYLE, "marginTop": "16px", "whiteSpace": "pre-wrap", "fontFamily": "ui-monospace, Consolas, monospace"}),
        html.Div([
            graph_card(go.Figure(), graph_id="regression-coefficient", min_height="420px"),
            graph_card(go.Figure(), graph_id="regression-residuals", min_height="450px"),
            graph_card(go.Figure(), graph_id="regression-qq", min_height="450px"),
        ], style={**GRID_STYLE, "marginTop": "16px"}),
        html.Div([
            html.H3("How to Read the Diagnostics", style={"marginTop": 0}),
            html.Ul([
                html.Li("Residuals versus fitted values assess centering, changing spread, and systematic structure. Two vertical fitted-value bands are expected because regulation is binary."),
                html.Li("The fitted values are predictions on the log-outcome scale, not probabilities, so values greater than one are expected."),
                html.Li("The Q-Q plot assesses approximate residual normality; tail deviations indicate outliers or heavier tails."),
                html.Li("The model is exploratory and associative, not causal or fully predictive."),
            ])
        ], style={**CARD_STYLE, "marginTop": "16px"}),
    ])

app = Dash(__name__, title="Global Lead Paint Policy and Health Burden")
server = app.server
app.layout = html.Div([
    html.Div([
        html.H1("Global Lead Paint Regulation and Health Burden", style={"margin": 0, "textAlign": "center", "fontSize": "30px"}),
        html.P("Integrated visual-statistical environmental health dashboard", style={"textAlign": "center", "color": COLORS["muted"], "marginBottom": 0}),
        limitation_note(),
    ], style={**CARD_STYLE, "marginBottom": "14px"}),
    dcc.Tabs([
        dcc.Tab(label="Visual Overview", children=html.Div(overview_layout(df), style={"paddingTop": "12px"})),
        dcc.Tab(label="Unadjusted Group Comparisons", children=html.Div(hypothesis_layout(df), style={"padding": "18px 0"})),
        dcc.Tab(label="Unadjusted Regression Modeling", children=html.Div(regression_layout(), style={"padding": "18px 0"})),
        dcc.Tab(label="Adjusted Analysis & Robustness", children=html.Div(research_results_layout(), style={"padding": "18px 0"})),
    ]),
], style=PAGE_STYLE)

# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------
@app.callback(
    Output("test-output", "children"),
    Output("hypothesis-boxplot", "figure"),
    Output("summary-table-container", "children"),
    Input("hypothesis-outcome", "value"),
)
def update_hypothesis(outcome: str):
    results = run_group_tests(df, outcome)
    effect = compute_effect_size(df, outcome)
    outcome_label = "DALY Rate" if outcome == "daly_rate" else "Death Rate"
    if not results["valid"]:
        return html.Div(results["message"], style=CARD_STYLE), empty_figure(results["message"]), make_summary_tables(df)
    effect_text = "Effect size unavailable"
    if effect["valid"]:
        effect_text = f"Cohen's d = {effect['cohen_d']:.3f} ({effect['interpretation']} effect)"
    cards = html.Div([
        html.H3(f"Group Comparison Results: {outcome_label}"),
        html.Div([
            html.Div([html.H4("Sample Sizes"), html.P(f"Regulated: {results['n_yes']}"), html.P(f"Not regulated: {results['n_no']}")], style=CARD_STYLE),
            html.Div([html.H4("Central Tendency"), html.P(f"Mean, regulated: {results['mean_yes']:,.3f}"), html.P(f"Mean, not regulated: {results['mean_no']:,.3f}"), html.P(f"Median, regulated: {results['median_yes']:,.3f}"), html.P(f"Median, not regulated: {results['median_no']:,.3f}")], style=CARD_STYLE),
            html.Div([html.H4("Statistical Evidence"), html.P(f"Welch t: {results['welch_t_stat']:.3f}; p = {results['welch_p_value']:.4g}"), html.P(f"Mann-Whitney U: {results['mannwhitney_u_stat']:.3f}; p = {results['mannwhitney_p_value']:.4g}"), html.P(effect_text)], style=CARD_STYLE),
        ], style=GRID_STYLE),
    ])
    return cards, make_boxplot(df, outcome), make_summary_tables(df)

@app.callback(
    Output("regression-summary", "children"),
    Output("regression-coefficient", "figure"),
    Output("regression-residuals", "figure"),
    Output("regression-qq", "figure"),
    Input("regression-outcome", "value"),
)
def update_regression(outcome: str):
    model, error = fit_policy_regression(df, outcome, robust=True)
    if error:
        figure = empty_figure(error)
        return error, figure, figure, figure
    coef = _series_value(model.params, "reg_binary", model)
    p_value = _series_value(model.pvalues, "reg_binary", model)
    low, high = _confidence_interval(model, "reg_binary")
    outcome_label = "DALY Rate" if outcome == "daly_rate" else "Death Rate"
    percent_difference = (np.exp(coef) - 1) * 100
    summary = (
        f"Model: log({outcome_label}) = intercept + regulation status\n"
        f"Covariance estimator: HC3 heteroskedasticity-robust\n\n"
        f"Regulation coefficient: {coef:.4f}\n"
        f"95% confidence interval: [{low:.4f}, {high:.4f}]\n"
        f"P-value: {p_value:.4g}\n"
        f"R-squared: {model.rsquared:.4f}\n"
        f"Estimated percentage difference: {percent_difference:.1f}%\n\n"
        "Interpretation: the coefficient estimates the average difference in the log outcome "
        "between regulated and not-regulated countries. This is an association, not a causal effect."
    )
    return summary, make_regression_coefficient_plot(model), make_residuals_plot(model), make_qq_plot(model)

if __name__ == "__main__":
    app.run(debug=True, port=8050)