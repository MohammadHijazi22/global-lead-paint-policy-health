"""Build a frozen, manuscript-ready analysis package.
This script does not refit models. 
It just validate and freeze the existing data, results, tables, figures, and provenance used for manuscript writing.
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import hashlib, json, platform, shutil, subprocess, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed" / "lead_policy_expanded.csv"
COV_PROVENANCE = ROOT / "data" / "external_covariates" / "world_bank_covariate_provenance.csv"
COV_COVERAGE = ROOT / "data" / "external_covariates" / "world_bank_covariate_coverage.csv"
S4 = ROOT / "results" / "statistical_analysis"
S5 = ROOT / "results" / "robustness_analysis"
OUT = ROOT / "manuscript_ready"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
SUPPLEMENT = OUT / "supplement"
AUDIT = OUT / "audit"

EXPECTED = {
    "data": DATA,
    "model_summary": S4 / "regulation_model_summary.csv",
    "all_coefficients": S4 / "all_model_coefficients.csv",
    "sample_sizes": S4 / "model_sample_sizes.csv",
    "vif": S4 / "vif_diagnostics.csv",
    "influence": S4 / "primary_model_influence.csv",
    "normality": S4 / "residual_normality_tests.csv",
    "standardization": S4 / "standardization_parameters.csv",
    "robustness": S5 / "robustness_summary.csv",
    "bootstrap_summary": S5 / "country_bootstrap_summary.csv",
    "bootstrap_draws": S5 / "country_bootstrap_draws.csv",
    "region_loo": S5 / "leave_one_region_out.csv",
    "influence_sensitivity": S5 / "influence_sensitivity.csv",
    "ihme_uncertainty": S5 / "ihme_uncertainty_summary.csv",
}
MODEL_ORDER = ["M0_unadjusted", "M1_development", "M2_primary", "M3_health_system", "M4_environment", "M5_industrial"]
MODEL_LABEL = {
    "M0_unadjusted": "M0: Unadjusted",
    "M1_development": "M1: GDP-adjusted",
    "M2_primary": "M2: Primary adjusted",
    "M3_health_system": "M3: Health-system sensitivity",
    "M4_environment": "M4: Environmental sensitivity",
    "M5_industrial": "M5: Industrial sensitivity",
}
OUTCOME_LABEL = {"daly_rate": "DALY rate", "death_rate": "Death rate"}

def require_files() -> None:
    missing = [f"{name}: {path}" for name, path in EXPECTED.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def fmt_p(value: float) -> str:
    if pd.isna(value):
        return "NA"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"

def format_effect(row: pd.Series) -> str:
    return (f"{row['percent_difference']:.1f}% ({row['percent_ci_low']:.1f}% to {row['percent_ci_high']:.1f}%)")

def validate_data(df: pd.DataFrame) -> list[dict]:
    checks = []
    def add(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})
    add("one row per ISO3", df["iso_code"].is_unique, f"rows={len(df)}, unique_iso={df['iso_code'].nunique()}")
    add("regulation binary valid", set(df["reg_binary"].dropna().astype(int).unique()).issubset({0, 1}), sorted(df["reg_binary"].dropna().unique()))
    add("positive DALY values", (df["daly_rate"].dropna() > 0).all(), f"nonpositive={(df['daly_rate'].dropna() <= 0).sum()}")
    add("positive death values", (df["death_rate"].dropna() > 0).all(), f"nonpositive={(df['death_rate'].dropna() <= 0).sum()}")
    add("complete health outcomes", df[["daly_rate", "death_rate"]].notna().all(axis=1).sum() == 163,
        f"complete={df[['daly_rate','death_rate']].notna().all(axis=1).sum()}")
    add("primary covariates available", all(c in df.columns for c in [
        "log_gdp_per_capita_ppp", "population_age_65plus_pct", "urban_population_pct", "wb_region"]), "required columns")
    add("no impossible urban percentages", df["urban_population_pct"].dropna().between(0, 100).all(), "expected range 0-100")
    add("no impossible age percentages", df["population_age_65plus_pct"].dropna().between(0, 100).all(), "expected range 0-100")
    return checks

def validate_results(models: pd.DataFrame, samples: pd.DataFrame, robust: pd.DataFrame) -> list[dict]:
    checks = []
    def add(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})
    add("12 staged model rows", len(models) == 12, f"rows={len(models)}")
    add("all six models per outcome", all(models.groupby("outcome_label")["model"].nunique() == 6),
        models.groupby("outcome_label")["model"].nunique().to_dict())
    primary = models[models["model"] == "M2_primary"]
    add("primary n=160", (primary["n"] == 160).all(), primary[["outcome_label", "n"]].to_dict("records"))
    add("confidence intervals ordered", (models["percent_ci_low"] <= models["percent_difference"]).all() and
        (models["percent_difference"] <= models["percent_ci_high"]).all(), "low <= estimate <= high")
    add("primary intervals include zero", ((primary["percent_ci_low"] <= 0) & (primary["percent_ci_high"] >= 0)).all(),
        primary[["outcome_label", "percent_ci_low", "percent_ci_high"]].to_dict("records"))
    add("sample-size file consistent", set(samples["n"]) <= {155, 160, 163}, sorted(samples["n"].unique()))
    boot = robust[robust["analysis"] == "Country bootstrap, primary model"]
    add("bootstrap rows present", len(boot) == 2, f"rows={len(boot)}")
    if "successful_replicates" in boot:
        add("bootstrap 2000 successful", (boot["successful_replicates"] == 2000).all(), boot["successful_replicates"].tolist())
    return checks

def table1_descriptive(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("Overall", df), ("Not regulated", df[df.reg_binary == 0]), ("Regulated", df[df.reg_binary == 1])]
    variables = [
        ("DALY rate", "daly_rate"),
        ("Death rate", "death_rate"),
        ("GDP per capita, PPP", "gdp_per_capita_ppp_2021"),
        ("Urban population (%)", "urban_population_pct"),
        ("Population aged 65+ (%)", "population_age_65plus_pct"),
        ("Health expenditure per capita (US$)", "health_expenditure_per_capita_usd"),
        ("PM2.5 exposure (µg/m³)", "pm25_mean_annual_ug_m3"),
        ("Manufacturing value added (% GDP)", "manufacturing_value_added_pct_gdp"),
    ]
    for label, column in variables:
        row = {"Characteristic": label}
        for group_name, subset in groups:
            values = pd.to_numeric(subset[column], errors="coerce").dropna()
            row[group_name] = f"{values.median():,.1f} ({values.quantile(.25):,.1f} to {values.quantile(.75):,.1f})"
            row[f"{group_name} n"] = len(values)
        rows.append(row)
    return pd.DataFrame(rows)

def table2_models(models: pd.DataFrame) -> pd.DataFrame:
    result = models.copy()
    result["Model"] = result["model"].map(MODEL_LABEL)
    result["Regulation estimate, % (95% CI)"] = result.apply(format_effect, axis=1)
    result["P value"] = result["p_value"].map(fmt_p)
    result["Adjusted R²"] = result["adjusted_r_squared"].map(lambda x: f"{x:.3f}")
    return result[["outcome_label", "Model", "model_description", "n", "Regulation estimate, % (95% CI)", "P value", "Adjusted R²"]].rename(
        columns={"outcome_label": "Outcome", "model_description": "Adjustment set", "n": "N"})

def table3_primary_coefficients(coeff: pd.DataFrame) -> pd.DataFrame:
    subset = coeff[(coeff["model"] == "M2_primary") &
                   (coeff["sample"] == "model_specific_complete_case") &
                   (coeff["term"] != "Intercept")].copy()
    subset["Estimate (95% CI), log scale"] = subset.apply(
        lambda r: f"{r.coefficient_log_scale:.3f} ({r.ci_low_log_scale:.3f} to {r.ci_high_log_scale:.3f})", axis=1)
    subset["P value"] = subset["p_value"].map(fmt_p)
    return subset[["outcome_label", "term", "Estimate (95% CI), log scale", "P value"]].rename(
        columns={"outcome_label": "Outcome", "term": "Term"})

def figure_bootstrap(draws: pd.DataFrame, models: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, outcome in zip(axes, ["daly_rate", "death_rate"]):
        vals = draws.loc[draws.outcome == outcome, "coefficient"].dropna()
        effect = 100 * (np.exp(vals) - 1)
        primary_label = "DALY Rate" if outcome == "daly_rate" else "Death Rate"
        primary = models[(models.outcome_label == primary_label) & (models.model == "M2_primary")].iloc[0]
        ax.hist(effect, bins=35, color="#2563EB", alpha=0.78, edgecolor="white")
        ax.axvline(0, color="black", linestyle="--", linewidth=1, label="Null")
        ax.axvline(primary.percent_difference, color="#C2410C", linewidth=2, label="Primary estimate")
        ax.set_title(primary_label)
        ax.set_xlabel("Regulated vs not regulated (%)")
        ax.set_ylabel("Bootstrap replicates")
        ax.legend(frameon=False)
    fig.suptitle("Country-bootstrap Distribution of Adjusted Regulation Estimates")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def figure_region_loo(regions: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for ax, outcome in zip(axes, ["daly_rate", "death_rate"]):
        s = regions[regions.outcome == outcome].copy()
        s = s.sort_values("percent_difference")
        y = np.arange(len(s))
        x = s.percent_difference.to_numpy()
        lo = s.percent_ci_low.to_numpy()
        hi = s.percent_ci_high.to_numpy()
        ax.errorbar(x, y, xerr=[x-lo, hi-x], fmt="o", capsize=3, color="#0F766E")
        ax.axvline(0, color="gray", linestyle="--")
        ax.set_yticks(y)
        ax.set_yticklabels(s.excluded_region)
        ax.set_title(OUTCOME_LABEL[outcome])
        ax.set_xlabel("Regulated vs not regulated (%)")
    fig.suptitle("Leave-One-Region-Out Sensitivity Analysis")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def write_readme(models: pd.DataFrame, checks: pd.DataFrame) -> None:
    primary = models[models.model == "M2_primary"].set_index("outcome_label")
    text = f"""# Frozen manuscript analysis package

Created: {datetime.now(timezone.utc).isoformat()}

## Primary findings

- DALY rate: {format_effect(primary.loc['DALY Rate'])}; p={fmt_p(primary.loc['DALY Rate','p_value'])}; n={int(primary.loc['DALY Rate','n'])}.
- Death rate: {format_effect(primary.loc['Death Rate'])}; p={fmt_p(primary.loc['Death Rate','p_value'])}; n={int(primary.loc['Death Rate','n'])}.

## Interpretation lock

The unadjusted DALY association attenuated substantially after adjustment. The primary and robustness confidence intervals 
include zero. Results are cross-sectional country-level associations and must not be described as causal.

## Package structure

- `tables/`: main manuscript tables in CSV form.
- `figures/`: publication-ready 300-dpi figures.
- `supplement/`: diagnostics and complete sensitivity outputs.
- `audit/`: validation checks, file hashes, and software versions.

## Validation

Passed {int(checks.passed.sum())} of {len(checks)} automated checks. Review
`audit/validation_report.csv` before manuscript drafting.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")

def main() -> None:
    require_files()
    for directory in [TABLES, FIGURES, SUPPLEMENT, AUDIT]:
        directory.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    models = pd.read_csv(EXPECTED["model_summary"])
    coeff = pd.read_csv(EXPECTED["all_coefficients"])
    samples = pd.read_csv(EXPECTED["sample_sizes"])
    robust = pd.read_csv(EXPECTED["robustness"])
    draws = pd.read_csv(EXPECTED["bootstrap_draws"])
    regions = pd.read_csv(EXPECTED["region_loo"])
    checks = validate_data(df) + validate_results(models, samples, robust)
    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(AUDIT / "validation_report.csv", index=False)
    if not checks_df["passed"].all():
        failed = checks_df.loc[~checks_df.passed, ["check", "detail"]]
        raise RuntimeError("Validation failed:\n" + failed.to_string(index=False))
    table1_descriptive(df).to_csv(TABLES / "table1_descriptive_characteristics.csv", index=False)
    table2_models(models).to_csv(TABLES / "table2_staged_regression_models.csv", index=False)
    table3_primary_coefficients(coeff).to_csv(TABLES / "table3_primary_model_coefficients.csv", index=False)
    figure_bootstrap(draws, models, FIGURES / "figure2_bootstrap_distributions.png")
    figure_region_loo(regions, FIGURES / "figure3_leave_one_region_out.png")
    existing_figures = ROOT / "figures" / "manuscript"
    for name in ["regulation_model_forest_plot.png", "robustness_forest_plot.png"]:
        source = existing_figures / name
        if source.exists():
            shutil.copy2(source, FIGURES / name)
    for key in ["vif", "influence", "normality", "standardization", "bootstrap_summary",
                "region_loo", "influence_sensitivity", "ihme_uncertainty", "robustness"]:
        shutil.copy2(EXPECTED[key], SUPPLEMENT / EXPECTED[key].name)
    if COV_PROVENANCE.exists():
        shutil.copy2(COV_PROVENANCE, SUPPLEMENT / COV_PROVENANCE.name)
    if COV_COVERAGE.exists():
        shutil.copy2(COV_COVERAGE, SUPPLEMENT / COV_COVERAGE.name)
    manifest_rows = []
    manifest_inputs = list(EXPECTED.values()) + [p for p in [COV_PROVENANCE, COV_COVERAGE] if p.exists()]
    for path in manifest_inputs:
        manifest_rows.append({
            "path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
            "sha256": sha256(path), "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        })
    pd.DataFrame(manifest_rows).to_csv(AUDIT / "input_file_manifest.csv", index=False)
    try:
        pip_freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    except Exception as exc:
        pip_freeze = f"Unavailable: {exc}"
    (AUDIT / "software_environment.txt").write_text(
        f"Python: {sys.version}\nPlatform: {platform.platform()}\n\n{pip_freeze}", encoding="utf-8")
    analysis_lock = {
        "design": "Cross-sectional country-level ecological analysis",
        "exposure": "Legally binding lead paint regulation status in 2023",
        "outcomes": ["Lead-attributable DALY rate", "Lead-attributable death rate"],
        "primary_model": "M2_primary",
        "primary_covariance": "HC3",
        "primary_adjustments": ["log GDP per capita PPP", "population aged 65+", "urban population", "World Bank region"],
        "primary_sample_n": 160,
        "missing_data": "Model-specific complete-case analysis",
        "causal_claim": False,
        "ihme_uncertainty_propagated": False,
        "random_seed": 614604,
    }
    (AUDIT / "analysis_lock.json").write_text(json.dumps(analysis_lock, indent=2), encoding="utf-8")
    write_readme(models, checks_df)
    print(f"Manuscript package created: {OUT}")
    print(f"Validation checks passed: {len(checks_df)}/{len(checks_df)}")

if __name__ == "__main__":
    main()