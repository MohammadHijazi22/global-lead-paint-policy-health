"""Uncertainty and robustness analysis.
This script keeps the Step 4 primary specification fixed and evaluates whether its regulation estimate is stable under alternative defensible assumptions.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm
import matplotlib.pyplot as plt
from statistical_analysis import load_analysis_data

ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "data" / "processed" / "lead_policy_expanded.csv"
OUTPUT_DIR = ROOT / "results" / "robustness_analysis"
FIGURE_DIR = ROOT / "figures" / "manuscript"
RANDOM_SEED = 614604
BOOTSTRAP_REPLICATES = 2000
IHME_MONTE_CARLO_DRAWS = 1000
OUTCOMES = {"daly_rate": "DALY Rate", "death_rate": "Death Rate"}
PRIMARY_CONTINUOUS = ["log_gdp_per_capita_ppp", "population_age_65plus_pct", "urban_population_pct"]
PRIMARY_FORMULA = ("log_{outcome} ~ reg_binary + z_log_gdp_per_capita_ppp + z_population_age_65plus_pct + z_urban_population_pct + C(wb_region)")

def standardize(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = data.copy()
    for column in columns:
        values = pd.to_numeric(result[column], errors="coerce")
        sd = values.std(ddof=1)
        result[f"z_{column}"] = (values - values.mean()) / sd if pd.notna(sd) and sd > 0 else np.nan
    return result

def primary_sample(data: pd.DataFrame, outcome: str) -> pd.DataFrame:
    needed = [outcome, f"log_{outcome}", "reg_binary", "z_log_gdp_per_capita_ppp", "z_population_age_65plus_pct", "z_urban_population_pct", "wb_region", "country", "iso_code"]
    return data.dropna(subset=needed).copy()

def fit_primary(data: pd.DataFrame, outcome: str, covariance: str = "HC3"):
    formula = PRIMARY_FORMULA.format(outcome=outcome)
    return smf.ols(formula, data=data).fit(cov_type=covariance)

def regulation_result(model, outcome: str, analysis: str, n: int, notes: str = "") -> dict[str, Any]:
    coefficient = float(model.params["reg_binary"])
    low, high = map(float, model.conf_int().loc["reg_binary"])
    return {
        "outcome": outcome,
        "outcome_label": OUTCOMES[outcome],
        "analysis": analysis,
        "n": n,
        "coefficient_log_scale": coefficient,
        "percent_difference": 100 * (np.exp(coefficient) - 1),
        "percent_ci_low": 100 * (np.exp(low) - 1),
        "percent_ci_high": 100 * (np.exp(high) - 1),
        "p_value": float(model.pvalues["reg_binary"]),
        "notes": notes,
    }

def baseline_and_covariance_checks(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome in OUTCOMES:
        sample = primary_sample(data, outcome)
        for covariance in ["nonrobust", "HC0", "HC1", "HC2", "HC3"]:
            model = fit_primary(sample, outcome, covariance=covariance)
            rows.append(regulation_result(
                model, outcome, f"Primary model, {covariance} SE", len(sample),
                "Coefficient is unchanged; standard errors and confidence intervals vary.",
            ))
    return pd.DataFrame(rows)

def country_bootstrap(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nonparametric country bootstrap of the fixed primary specification."""
    rng = np.random.default_rng(RANDOM_SEED)
    summary_rows = []
    draw_rows = []
    for outcome in OUTCOMES:
        sample = primary_sample(data, outcome).reset_index(drop=True)
        estimates = []
        failures = 0
        for draw in range(BOOTSTRAP_REPLICATES):
            sampled = sample.iloc[rng.integers(0, len(sample), len(sample))].copy()
            try:
                # Skip samples missing either regulation group or a region required for a stable fit.
                if sampled["reg_binary"].nunique() < 2:
                    failures += 1
                    continue
                model = smf.ols(PRIMARY_FORMULA.format(outcome=outcome), data=sampled).fit()
                estimate = float(model.params["reg_binary"])
                if np.isfinite(estimate):
                    estimates.append(estimate)
                    draw_rows.append({"outcome": outcome, "draw": draw, "coefficient": estimate})
            except Exception:
                failures += 1
        estimates = np.asarray(estimates)
        low, high = np.quantile(estimates, [0.025, 0.975])
        estimate = float(np.mean(estimates))
        summary_rows.append({
            "outcome": outcome,
            "analysis": "Country bootstrap, primary model",
            "n": len(sample),
            "successful_replicates": len(estimates),
            "failed_replicates": failures,
            "coefficient_log_scale": estimate,
            "percent_difference": 100 * (np.exp(estimate) - 1),
            "percent_ci_low": 100 * (np.exp(low) - 1),
            "percent_ci_high": 100 * (np.exp(high) - 1),
            "p_value": np.nan,
            "notes": "Percentile interval from country-level bootstrap replicates.",
        })
    return pd.DataFrame(summary_rows), pd.DataFrame(draw_rows)

def region_clustered_inference(data: pd.DataFrame) -> pd.DataFrame:
    """Cluster standard errors by World Bank region; interpret cautiously with few clusters."""
    rows = []
    for outcome in OUTCOMES:
        sample = primary_sample(data, outcome)
        model = smf.ols(PRIMARY_FORMULA.format(outcome=outcome), data=sample).fit(cov_type="cluster", cov_kwds={"groups": sample["wb_region"]})
        rows.append(regulation_result(
            model, outcome, "Primary model, region-clustered SE", len(sample),
            f"Clustered across {sample['wb_region'].nunique()} World Bank regions; few-cluster inference is exploratory.",
        ))
    return pd.DataFrame(rows)

def leave_one_region_out(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome in OUTCOMES:
        sample = primary_sample(data, outcome)
        regions = sorted(sample["wb_region"].dropna().unique())
        for excluded_region in regions:
            reduced = sample[sample["wb_region"] != excluded_region].copy()
            model = fit_primary(reduced, outcome, covariance="HC3")
            result = regulation_result(
                model, outcome, "Leave one World Bank region out", len(reduced),
                f"Excluded region: {excluded_region}",
            )
            result["excluded_region"] = excluded_region
            rows.append(result)
    return pd.DataFrame(rows)

def influence_sensitivity(data: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-country-out for Cook's D flagged observations and all flagged removed."""
    rows = []
    for outcome in OUTCOMES:
        sample = primary_sample(data, outcome)
        formula = PRIMARY_FORMULA.format(outcome=outcome)
        ordinary = smf.ols(formula, data=sample).fit()
        influence = ordinary.get_influence().summary_frame()
        cook = np.asarray(influence["cooks_d"])
        threshold = 4 / len(sample)
        flagged_positions = np.where(cook > threshold)[0]
        flagged_countries = sample.iloc[flagged_positions]["country"].tolist()
        for country in flagged_countries:
            reduced = sample[sample["country"] != country]
            model = fit_primary(reduced, outcome, covariance="HC3")
            result = regulation_result(
                model, outcome, "Leave one influential country out", len(reduced),
                f"Excluded country: {country}",
            )
            result["excluded_country"] = country
            rows.append(result)
        if flagged_countries:
            reduced = sample[~sample["country"].isin(flagged_countries)]
            model = fit_primary(reduced, outcome, covariance="HC3")
            result = regulation_result(
                model, outcome, "All Cook's D flagged countries removed", len(reduced),
                f"Removed {len(flagged_countries)} countries using Cook's D > 4/n.",
            )
            rows.append(result)
    return pd.DataFrame(rows)

def robust_regression(data: pd.DataFrame) -> pd.DataFrame:
    """Huber M-estimation as an outlier-resistant sensitivity model."""
    rows = []
    for outcome in OUTCOMES:
        sample = primary_sample(data, outcome)
        formula = PRIMARY_FORMULA.format(outcome=outcome)
        model = smf.rlm(formula, data=sample, M=sm.robust.norms.HuberT()).fit()
        coefficient = float(model.params["reg_binary"])
        se = float(model.bse["reg_binary"])
        low, high = coefficient - 1.96 * se, coefficient + 1.96 * se
        p_value = 2 * norm.sf(abs(coefficient / se))
        rows.append({
            "outcome": outcome,
            "outcome_label": OUTCOMES[outcome],
            "analysis": "Huber robust regression",
            "n": len(sample),
            "coefficient_log_scale": coefficient,
            "percent_difference": 100 * (np.exp(coefficient) - 1),
            "percent_ci_low": 100 * (np.exp(low) - 1),
            "percent_ci_high": 100 * (np.exp(high) - 1),
            "p_value": p_value,
            "notes": "Outlier-resistant M-estimation; normal-approximation interval.",
        })
    return pd.DataFrame(rows)

def quantile_regression(data: pd.DataFrame) -> pd.DataFrame:
    """Conditional median and quartile models for log outcomes."""
    rows = []
    for outcome in OUTCOMES:
        sample = primary_sample(data, outcome)
        formula = PRIMARY_FORMULA.format(outcome=outcome)
        for quantile in [0.25, 0.50, 0.75]:
            model = smf.quantreg(formula, data=sample).fit(q=quantile, max_iter=5000)
            coefficient = float(model.params["reg_binary"])
            low, high = map(float, model.conf_int().loc["reg_binary"])
            rows.append({
                "outcome": outcome,
                "outcome_label": OUTCOMES[outcome],
                "analysis": f"Quantile regression q={quantile:.2f}",
                "n": len(sample),
                "coefficient_log_scale": coefficient,
                "percent_difference": 100 * (np.exp(coefficient) - 1),
                "percent_ci_low": 100 * (np.exp(low) - 1),
                "percent_ci_high": 100 * (np.exp(high) - 1),
                "p_value": float(model.pvalues["reg_binary"]),
                "notes": "Tests whether the association differs across the conditional outcome distribution.",
            })
    return pd.DataFrame(rows)

def alternative_region_definition(data: pd.DataFrame) -> pd.DataFrame:
    """Replace World Bank region with the six-continent dashboard region."""
    rows = []
    for outcome in OUTCOMES:
        needed = [f"log_{outcome}", "reg_binary", "z_log_gdp_per_capita_ppp", "z_population_age_65plus_pct", "z_urban_population_pct", "region",]
        sample = data.dropna(subset=needed).copy()
        formula = (f"log_{outcome} ~ reg_binary + z_log_gdp_per_capita_ppp + z_population_age_65plus_pct + z_urban_population_pct + C(region)")
        model = smf.ols(formula, data=sample).fit(cov_type="HC3")
        rows.append(regulation_result(
            model, outcome, "Alternative six-continent region adjustment", len(sample),
            "Replaces World Bank analytical regions with dashboard continent categories.",
        ))
    return pd.DataFrame(rows)

def outcome_scale_checks(data: pd.DataFrame) -> pd.DataFrame:
    """Compare main log-OLS with Gamma GLM using a log link on raw positive rates."""
    rows = []
    for outcome in OUTCOMES:
        sample = primary_sample(data, outcome)
        rhs = ("reg_binary + z_log_gdp_per_capita_ppp + z_population_age_65plus_pct + z_urban_population_pct + C(wb_region)")
        model = smf.glm(f"{outcome} ~ {rhs}", data=sample, family=sm.families.Gamma(link=sm.families.links.Log())).fit(cov_type="HC3")
        rows.append(regulation_result(
            model, outcome, "Gamma GLM with log link", len(sample),
            "Models positive raw rates with multiplicative mean structure.",
        ))
    return pd.DataFrame(rows)

def temporal_carry_forward_check(data: pd.DataFrame) -> pd.DataFrame:
    """Exclude observations whose selected primary covariates are older than 2022."""
    rows = []
    lag_columns = ["gdp_per_capita_ppp_2021_lag", "urban_population_pct_lag", "population_age_65plus_pct_lag"]
    for outcome in OUTCOMES:
        sample = primary_sample(data, outcome)
        available_lags = [column for column in lag_columns if column in sample.columns]
        if available_lags:
            current = sample[(sample[available_lags].fillna(99) == 0).all(axis=1)].copy()
        else:
            current = sample
        model = fit_primary(current, outcome, covariance="HC3")
        rows.append(regulation_result(
            model, outcome, "Only 2022 primary covariates", len(current),
            "Excludes countries with carried-forward primary covariate values.",
        ))
    return pd.DataFrame(rows)

def ihme_uncertainty_monte_carlo(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Propagate IHME outcome uncertainty if lower and upper bounds are available.
    Assumes the reported lower and upper bounds approximate a 95% interval on
    the log-rate scale. This is an approximation because country estimates are
    sampled independently unless a full IHME draw file is supplied.
    """
    rng = np.random.default_rng(RANDOM_SEED + 1)
    summaries = []
    draws_output = []
    for outcome in OUTCOMES:
        lower_col = f"{outcome}_lower"
        upper_col = f"{outcome}_upper"
        if lower_col not in data.columns or upper_col not in data.columns:
            summaries.append({
                "outcome": outcome,
                "analysis": "IHME uncertainty Monte Carlo",
                "available": False,
                "successful_draws": 0,
                "notes": f"Skipped: columns {lower_col} and {upper_col} were not found.",
            })
            continue
        sample = primary_sample(data, outcome)
        sample = sample.dropna(subset=[lower_col, upper_col]).copy()
        valid = ((sample[outcome] > 0) & (sample[lower_col] > 0) & (sample[upper_col] > sample[lower_col]))
        sample = sample[valid].copy()
        if len(sample) < 50:
            summaries.append({"outcome": outcome, "analysis": "IHME uncertainty Monte Carlo", "available": False, "successful_draws": 0, "notes": "Skipped: insufficient records with valid uncertainty bounds."})
            continue
        log_point = np.log(sample[outcome].to_numpy())
        log_lower = np.log(sample[lower_col].to_numpy())
        log_upper = np.log(sample[upper_col].to_numpy())
        sigma = (log_upper - log_lower) / (2 * 1.96)
        estimates = []
        for draw in range(IHME_MONTE_CARLO_DRAWS):
            working = sample.copy()
            working[f"log_{outcome}"] = rng.normal(log_point, sigma)
            try:
                model = smf.ols(PRIMARY_FORMULA.format(outcome=outcome), data=working).fit()
                estimate = float(model.params["reg_binary"])
                estimates.append(estimate)
                draws_output.append({"outcome": outcome, "draw": draw, "coefficient": estimate})
            except Exception:
                continue
        estimates = np.asarray(estimates)
        low, high = np.quantile(estimates, [0.025, 0.975])
        estimate = float(np.mean(estimates))
        summaries.append({
            "outcome": outcome,
            "analysis": "IHME uncertainty Monte Carlo",
            "available": True,
            "successful_draws": len(estimates),
            "n": len(sample),
            "coefficient_log_scale": estimate,
            "percent_difference": 100 * (np.exp(estimate) - 1),
            "percent_ci_low": 100 * (np.exp(low) - 1),
            "percent_ci_high": 100 * (np.exp(high) - 1),
            "notes": "Approximate propagation of reported 95% uncertainty bounds on log-rate scale.",
        })
    return pd.DataFrame(summaries), pd.DataFrame(draws_output)

def make_robustness_forest(summary: pd.DataFrame, path: Path) -> None:
    display = summary.dropna(subset=["percent_difference", "percent_ci_low", "percent_ci_high"]).copy()
    preferred = [
        "Primary model, HC3 SE",
        "Country bootstrap, primary model",
        "Primary model, region-clustered SE",
        "Huber robust regression",
        "Quantile regression q=0.50",
        "Gamma GLM with log link",
        "Alternative six-continent region adjustment",
        "Only 2022 primary covariates",
    ]
    display = display[display["analysis"].isin(preferred)].copy()
    display["analysis"] = pd.Categorical(display["analysis"], preferred, ordered=True)
    display = display.sort_values(["outcome", "analysis"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
    for ax, outcome in zip(axes, OUTCOMES):
        subset = display[display["outcome"] == outcome].copy()
        y = np.arange(len(subset))
        x = subset["percent_difference"].to_numpy()
        low = subset["percent_ci_low"].to_numpy()
        high = subset["percent_ci_high"].to_numpy()
        ax.errorbar(x, y, xerr=[x - low, high - x], fmt="o", capsize=4)
        ax.axvline(0, linestyle="--", color="gray", linewidth=1)
        ax.set_title(OUTCOMES[outcome])
        ax.set_xlabel("Regulated vs not regulated (%)")
        ax.set_yticks(y)
        ax.set_yticklabels(subset["analysis"])
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Robustness of the Adjusted Regulation Association")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def run_robustness_analysis() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data = standardize(load_analysis_data(), PRIMARY_CONTINUOUS)
    covariance = baseline_and_covariance_checks(data)
    bootstrap_summary, bootstrap_draws = country_bootstrap(data)
    clustered = region_clustered_inference(data)
    region_loo = leave_one_region_out(data)
    influential = influence_sensitivity(data)
    robust = robust_regression(data)
    quantile = quantile_regression(data)
    alternative_region = alternative_region_definition(data)
    scale = outcome_scale_checks(data)
    temporal = temporal_carry_forward_check(data)
    ihme_summary, ihme_draws = ihme_uncertainty_monte_carlo(data)
    covariance.to_csv(OUTPUT_DIR / "covariance_estimator_sensitivity.csv", index=False)
    bootstrap_summary.to_csv(OUTPUT_DIR / "country_bootstrap_summary.csv", index=False)
    bootstrap_draws.to_csv(OUTPUT_DIR / "country_bootstrap_draws.csv", index=False)
    clustered.to_csv(OUTPUT_DIR / "region_clustered_inference.csv", index=False)
    region_loo.to_csv(OUTPUT_DIR / "leave_one_region_out.csv", index=False)
    influential.to_csv(OUTPUT_DIR / "influence_sensitivity.csv", index=False)
    robust.to_csv(OUTPUT_DIR / "huber_robust_regression.csv", index=False)
    quantile.to_csv(OUTPUT_DIR / "quantile_regression.csv", index=False)
    alternative_region.to_csv(OUTPUT_DIR / "alternative_region_definition.csv", index=False)
    scale.to_csv(OUTPUT_DIR / "outcome_scale_sensitivity.csv", index=False)
    temporal.to_csv(OUTPUT_DIR / "temporal_alignment_sensitivity.csv", index=False)
    ihme_summary.to_csv(OUTPUT_DIR / "ihme_uncertainty_summary.csv", index=False)
    ihme_draws.to_csv(OUTPUT_DIR / "ihme_uncertainty_draws.csv", index=False)
    combined = pd.concat([covariance, bootstrap_summary, clustered, robust, quantile, alternative_region, scale, temporal], ignore_index=True, sort=False)
    combined.to_csv(OUTPUT_DIR / "robustness_summary.csv", index=False)
    make_robustness_forest(combined, FIGURE_DIR / "robustness_forest_plot.png")
    print("Step 5 robustness analysis completed.")
    print(f"Results: {OUTPUT_DIR}")
    print(f"Figure: {FIGURE_DIR / 'robustness_forest_plot.png'}")
    if not ihme_summary.empty and not ihme_summary.get("available", pd.Series(dtype=bool)).fillna(False).any():
        print("IHME uncertainty propagation was skipped because lower/upper outcome bounds were not present.")

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_robustness_analysis()