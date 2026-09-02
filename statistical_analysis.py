from __future__ import annotations
from pathlib import Path
from typing import Any
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "data" / "processed" / "lead_policy_expanded.csv"
OUTPUT_DIR = ROOT / "results" / "statistical_analysis"
FIGURE_DIR = ROOT / "figures" / "manuscript"
OUTCOMES = {"daly_rate": "DALY Rate", "death_rate": "Death Rate"}
CONTINUOUS = [
    "log_gdp_per_capita_ppp",
    "urban_population_pct",
    "population_age_65plus_pct",
    "log_health_expenditure_per_capita",
    "pm25_mean_annual_ug_m3",
    "manufacturing_value_added_pct_gdp",
]
MODEL_SPECS = {
    "M0_unadjusted": {
        "covariates": [],
        "description": "Unadjusted regulation association",
    },
    "M1_development": {
        "covariates": ["z_log_gdp_per_capita_ppp"],
        "description": "Adjusted for log GDP per capita",
    },
    "M2_primary": {
        "covariates": [
            "z_log_gdp_per_capita_ppp",
            "z_population_age_65plus_pct",
            "z_urban_population_pct",
            "C(wb_region)",
        ],
        "description": "Primary model: GDP, age structure, urbanization, and World Bank region",
    },
    "M3_health_system": {
        "covariates": [
            "z_log_health_expenditure_per_capita",
            "z_population_age_65plus_pct",
            "z_urban_population_pct",
            "C(wb_region)",
        ],
        "description": "Health-system sensitivity model replacing GDP with health spending",
    },
    "M4_environment": {
        "covariates": [
            "z_log_gdp_per_capita_ppp",
            "z_population_age_65plus_pct",
            "z_urban_population_pct",
            "z_pm25_mean_annual_ug_m3",
            "C(wb_region)",
        ],
        "description": "Environmental-context sensitivity model adding PM2.5",
    },
    "M5_industrial": {
        "covariates": [
            "z_log_gdp_per_capita_ppp",
            "z_population_age_65plus_pct",
            "z_urban_population_pct",
            "z_manufacturing_value_added_pct_gdp",
            "C(wb_region)",
        ],
        "description": "Industrial-context sensitivity model adding manufacturing share",
    },
}

def load_analysis_data(path: Path = INPUT_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["reg_binary"] = pd.to_numeric(df["reg_binary"], errors="coerce")
    for outcome in OUTCOMES:
        df[f"log_{outcome}"] = np.where(df[outcome] > 0, np.log(df[outcome]), np.nan)
    df["wb_region"] = df["wb_region"].astype("string").str.strip()
    return df

def add_standardized_covariates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = df.copy()
    metadata = []
    for column in CONTINUOUS:
        values = pd.to_numeric(result[column], errors="coerce")
        mean = values.mean(skipna=True)
        sd = values.std(skipna=True, ddof=1)
        z_column = f"z_{column}"
        result[z_column] = (values - mean) / sd if np.isfinite(sd) and sd > 0 else np.nan
        metadata.append({
            "variable": column,
            "standardized_variable": z_column,
            "mean": mean,
            "standard_deviation": sd,
            "available_n": int(values.notna().sum()),
        })
    return result, pd.DataFrame(metadata)

def required_columns(covariates: list[str]) -> list[str]:
    cols = ["reg_binary"]
    for term in covariates:
        if term.startswith("C("):
            cols.append(term[2:-1])
        else:
            cols.append(term)
    return cols

def fit_model(data: pd.DataFrame, outcome: str, model_name: str, sample_index=None):
    spec = MODEL_SPECS[model_name]
    log_outcome = f"log_{outcome}"
    needed = [log_outcome] + required_columns(spec["covariates"])
    working = data.loc[sample_index].copy() if sample_index is not None else data.copy()
    working = working.dropna(subset=needed)
    formula_rhs = "reg_binary"
    if spec["covariates"]:
        formula_rhs += " + " + " + ".join(spec["covariates"])
    formula = f"{log_outcome} ~ {formula_rhs}"
    model = smf.ols(formula, data=working).fit(cov_type="HC3")
    return model, working, formula

def extract_model_results(model, outcome: str, model_name: str, sample_name: str, formula: str) -> list[dict[str, Any]]:
    ci = model.conf_int()
    rows = []
    for term in model.params.index:
        coef = float(model.params[term])
        low, high = map(float, ci.loc[term])
        row = {
            "outcome": outcome,
            "outcome_label": OUTCOMES[outcome],
            "model": model_name,
            "model_description": MODEL_SPECS[model_name]["description"],
            "sample": sample_name,
            "formula": formula,
            "n": int(model.nobs),
            "term": term,
            "coefficient_log_scale": coef,
            "standard_error_hc3": float(model.bse[term]),
            "ci_low_log_scale": low,
            "ci_high_log_scale": high,
            "p_value": float(model.pvalues[term]),
            "r_squared": float(model.rsquared),
            "adjusted_r_squared": float(model.rsquared_adj),
            "aic_nonrobust_likelihood": float(model.aic),
            "bic_nonrobust_likelihood": float(model.bic),
        }
        if term == "reg_binary":
            row.update({
                "percent_difference": 100 * (np.exp(coef) - 1),
                "percent_ci_low": 100 * (np.exp(low) - 1),
                "percent_ci_high": 100 * (np.exp(high) - 1),
            })
        else:
            row.update({"percent_difference": np.nan, "percent_ci_low": np.nan, "percent_ci_high": np.nan})
        rows.append(row)
    return rows

def calculate_vif(model, working: pd.DataFrame, model_name: str, outcome: str) -> pd.DataFrame:
    matrix = pd.DataFrame(model.model.exog, columns=model.model.exog_names)
    rows = []
    for idx, name in enumerate(matrix.columns):
        if name == "Intercept":
            continue
        try:
            vif = variance_inflation_factor(matrix.values, idx)
        except Exception:
            vif = np.nan
        rows.append({"outcome": outcome, "model": model_name, "n": len(working), "term": name, "vif": vif})
    return pd.DataFrame(rows)

def influence_table(model, working: pd.DataFrame, outcome: str, model_name: str) -> pd.DataFrame:
    # Influence values are calculated from the underlying OLS fit. HC3 changes inference, not fitted values.
    base_model = smf.ols(model.model.formula, data=working).fit()
    infl = base_model.get_influence()
    frame = infl.summary_frame()
    threshold = 4 / len(working)
    result = pd.DataFrame({
        "country": working["country"].to_numpy(),
        "iso_code": working["iso_code"].to_numpy(),
        "outcome": outcome,
        "model": model_name,
        "cooks_distance": frame["cooks_d"].to_numpy(),
        "studentized_residual": frame["student_resid"].to_numpy(),
        "hat_value": frame["hat_diag"].to_numpy(),
        "cooks_threshold_4_over_n": threshold,
    })
    result["flag_influential"] = result["cooks_distance"] > threshold
    result["flag_large_studentized_residual"] = result["studentized_residual"].abs() > 2
    return result.sort_values("cooks_distance", ascending=False)

def same_sample_analysis(data: pd.DataFrame, outcome: str) -> list[dict[str, Any]]:
    # Determine the primary-model sample, then refit M0 and M2 on exactly that sample.
    primary_model, primary_data, primary_formula = fit_model(data, outcome, "M2_primary")
    rows = extract_model_results(primary_model, outcome, "M2_primary", "primary_complete_case", primary_formula)
    unadjusted_same, same_data, same_formula = fit_model(data, outcome, "M0_unadjusted", sample_index=primary_data.index)
    rows += extract_model_results(unadjusted_same, outcome, "M0_unadjusted", "primary_complete_case", same_formula)
    return rows

def fit_all_models(data: pd.DataFrame):
    all_results = []
    all_vif = []
    all_influence = []
    model_objects = {}
    for outcome in OUTCOMES:
        for model_name in MODEL_SPECS:
            model, working, formula = fit_model(data, outcome, model_name)
            model_objects[(outcome, model_name)] = (model, working)
            all_results.extend(extract_model_results(model, outcome, model_name, "model_specific_complete_case", formula))
            all_vif.append(calculate_vif(model, working, model_name, outcome))
            if model_name == "M2_primary":
                all_influence.append(influence_table(model, working, outcome, model_name))
        all_results.extend(same_sample_analysis(data, outcome))
    return (
        pd.DataFrame(all_results),
        pd.concat(all_vif, ignore_index=True),
        pd.concat(all_influence, ignore_index=True),
        model_objects,
    )

def leave_one_out_influential(data: pd.DataFrame, influence: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome in OUTCOMES:
        flagged = influence[(influence["outcome"] == outcome) & influence["flag_influential"]]
        baseline, baseline_data, _ = fit_model(data, outcome, "M2_primary")
        baseline_coef = float(baseline.params["reg_binary"])
        for country in flagged["country"]:
            reduced_index = baseline_data.index[baseline_data["country"] != country]
            model, working, _ = fit_model(data, outcome, "M2_primary", sample_index=reduced_index)
            coef = float(model.params["reg_binary"])
            low, high = map(float, model.conf_int().loc["reg_binary"])
            rows.append({
                "outcome": outcome,
                "excluded_country": country,
                "n": int(model.nobs),
                "baseline_coefficient": baseline_coef,
                "leave_one_out_coefficient": coef,
                "coefficient_change": coef - baseline_coef,
                "percent_difference": 100 * (np.exp(coef) - 1),
                "percent_ci_low": 100 * (np.exp(low) - 1),
                "percent_ci_high": 100 * (np.exp(high) - 1),
                "p_value": float(model.pvalues["reg_binary"]),
            })
    return pd.DataFrame(rows)

def residual_normality(model_objects) -> pd.DataFrame:
    from scipy.stats import jarque_bera, shapiro
    rows = []
    for (outcome, model_name), (model, working) in model_objects.items():
        if model_name not in {"M0_unadjusted", "M2_primary"}:
            continue
        residuals = np.asarray(model.resid)
        jb = jarque_bera(residuals)
        sw = shapiro(residuals)
        rows.append({
            "outcome": outcome,
            "model": model_name,
            "n": len(residuals),
            "jarque_bera_statistic": float(jb.statistic),
            "jarque_bera_p_value": float(jb.pvalue),
            "shapiro_statistic": float(sw.statistic),
            "shapiro_p_value": float(sw.pvalue),
        })
    return pd.DataFrame(rows)

def regulation_forest_plot(results: pd.DataFrame, output_path: Path) -> None:
    plot_data = results[(results["term"] == "reg_binary") & (results["sample"] == "model_specific_complete_case")].copy()
    order = list(MODEL_SPECS.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for ax, outcome in zip(axes, OUTCOMES):
        subset = plot_data[plot_data["outcome"] == outcome].set_index("model").reindex(order).reset_index()
        y = np.arange(len(subset))
        x = subset["percent_difference"].to_numpy()
        low = subset["percent_ci_low"].to_numpy()
        high = subset["percent_ci_high"].to_numpy()
        ax.errorbar(x, y, xerr=[x - low, high - x], fmt="o", capsize=4)
        ax.axvline(0, linestyle="--", color="gray", linewidth=1)
        ax.set_title(OUTCOMES[outcome])
        ax.set_xlabel("Estimated difference for regulated countries (%)")
        ax.set_yticks(y)
        ax.set_yticklabels(subset["model"])
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Regulation Association Across Staged Models")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def model_summary_table(results: pd.DataFrame) -> pd.DataFrame:
    table = results[(results["term"] == "reg_binary") & (results["sample"] == "model_specific_complete_case")][[
        "outcome_label", "model", "model_description", "n", "percent_difference", "percent_ci_low", "percent_ci_high", "p_value", "r_squared", "adjusted_r_squared",
    ]].copy()
    return table.sort_values(["outcome_label", "model"]).round(4)

def run_statistical_upgrade() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data = load_analysis_data()
    data, standardization = add_standardized_covariates(data)
    standardization.to_csv(OUTPUT_DIR / "standardization_parameters.csv", index=False)
    results, vif, influence, model_objects = fit_all_models(data)
    results.to_csv(OUTPUT_DIR / "all_model_coefficients.csv", index=False)
    model_summary_table(results).to_csv(OUTPUT_DIR / "regulation_model_summary.csv", index=False)
    vif.to_csv(OUTPUT_DIR / "vif_diagnostics.csv", index=False)
    influence.to_csv(OUTPUT_DIR / "primary_model_influence.csv", index=False)
    leave_one_out = leave_one_out_influential(data, influence)
    leave_one_out.to_csv(OUTPUT_DIR / "influential_country_sensitivity.csv", index=False)
    normality = residual_normality(model_objects)
    normality.to_csv(OUTPUT_DIR / "residual_normality_tests.csv", index=False)
    sample_report = []
    for outcome in OUTCOMES:
        for model_name, (model, working) in {key[1]: value for key, value in model_objects.items() if key[0] == outcome}.items():
            sample_report.append({
                "outcome": outcome,
                "model": model_name,
                "n": int(model.nobs),
                "regulated_n": int((working["reg_binary"] == 1).sum()),
                "not_regulated_n": int((working["reg_binary"] == 0).sum()),
            })
    pd.DataFrame(sample_report).to_csv(OUTPUT_DIR / "model_sample_sizes.csv", index=False)
    regulation_forest_plot(results, FIGURE_DIR / "regulation_model_forest_plot.png")
    print("Step 4 statistical upgrade completed.")
    print(f"Results directory: {OUTPUT_DIR}")
    print(f"Forest plot: {FIGURE_DIR / 'regulation_model_forest_plot.png'}")
    print("\nPrimary regulation results:")
    print(model_summary_table(results).to_string(index=False))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        run_statistical_upgrade()
