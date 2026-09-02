from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import pycountry
import pycountry_convert as pc
from scipy.stats import mannwhitneyu, ttest_ind
import statsmodels.api as sm

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
GBD_PATH = RAW_DIR / "IHME-GBD_2023_DATA-32e3c624-1.csv"
REGULATION_PATH = RAW_DIR / "legal-controls-lead-paint(in).csv"
OUTPUT_PATH = PROCESSED_DIR / "lead_policy_merged.csv"

IHME_TO_REG_COUNTRY = {
    "Bolivia (Plurinational State of)": "Bolivia",
    "Brunei Darussalam": "Brunei",
    "Cabo Verde": "Cape Verde",
    "Côte d'Ivoire": "Cote d'Ivoire",
    "Democratic People's Republic of Korea": "North Korea",
    "Democratic Republic of the Congo": "Democratic Republic of Congo",
    "Iran (Islamic Republic of)": "Iran",
    "Lao People's Democratic Republic": "Laos",
    "Micronesia (Federated States of)": "Micronesia",
    "Republic of Korea": "South Korea",
    "Republic of Moldova": "Moldova",
    "Russian Federation": "Russia",
    "Syrian Arab Republic": "Syria",
    "Timor-Leste": "Timor-Leste",
    "Türkiye": "Turkey",
    "United Republic of Tanzania": "Tanzania",
    "United States of America": "United States",
    "Venezuela (Bolivarian Republic of)": "Venezuela",
    "Viet Nam": "Vietnam",
}

REG_COUNTRY_FIXES = {"Timor": "Timor-Leste"}
ISO_FIXES = {"Timor-Leste": "TLS"}
MANUAL_ISO3_REGION_MAP = {"TLS": "Asia", "XKX": "Europe"}
CONTINENT_MAP = {
    "AF": "Africa",
    "EU": "Europe",
    "AS": "Asia",
    "NA": "North America",
    "SA": "South America",
    "OC": "Oceania",
    "AN": "Antarctica",
}


def load_gbd_data(path: Path | str = GBD_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def iso3_to_region(iso3: Any) -> str:
    if not isinstance(iso3, str):
        return "Other"
    code = iso3.strip().upper()
    if len(code) != 3:
        return "Other"
    if code in MANUAL_ISO3_REGION_MAP:
        return MANUAL_ISO3_REGION_MAP[code]
    try:
        country = pycountry.countries.get(alpha_3=code)
        if country is None:
            return "Other"
        continent_code = pc.country_alpha2_to_continent_code(country.alpha_2)
        return CONTINENT_MAP.get(continent_code, "Other")
    except (KeyError, AttributeError, TypeError, ValueError):
        return "Other"


def extract_death_and_daly_rates(
    gbd_df: pd.DataFrame, year: int = 2023
) -> pd.DataFrame:
    required = {
        "year", "age_name", "sex_name", "metric_name",
        "measure_name", "location_name", "val",
    }
    missing = required.difference(gbd_df.columns)
    if missing:
        raise ValueError(f"GBD data is missing required columns: {sorted(missing)}")

    filtered = gbd_df.loc[
        (gbd_df["year"] == year)
        & (gbd_df["age_name"] == "All ages")
        & (gbd_df["sex_name"] == "Both")
        & (gbd_df["metric_name"] == "Rate")
    ].copy()

    deaths = filtered.loc[
        filtered["measure_name"] == "Deaths", ["location_name", "year", "val"]
    ].rename(columns={"location_name": "ihme_country", "year": "Year", "val": "death_rate"})

    dalys = filtered.loc[
        filtered["measure_name"].str.contains("DALYs", case=False, na=False),
        ["location_name", "year", "val"],
    ].rename(columns={"location_name": "ihme_country", "year": "Year", "val": "daly_rate"})

    health = deaths.merge(dalys, on=["ihme_country", "Year"], how="inner", validate="one_to_one")
    health["country_for_merge"] = (
        health["ihme_country"].replace(IHME_TO_REG_COUNTRY).astype(str).str.strip()
    )
    return health


def prepare_regulation_data(
    path: Path | str = REGULATION_PATH, year: int = 2023
) -> pd.DataFrame:
    reg = pd.read_csv(path).rename(
        columns={
            "Entity": "country",
            "Code": "iso_code",
            "lead_paint_regulation": "regulation",
        }
    )
    required = {"country", "iso_code", "Year", "regulation"}
    missing = required.difference(reg.columns)
    if missing:
        raise ValueError(f"Regulation data is missing required columns: {sorted(missing)}")

    reg = reg.loc[reg["Year"] == year].copy()
    reg["country"] = reg["country"].replace(REG_COUNTRY_FIXES).astype(str).str.strip()
    reg["iso_code"] = reg["iso_code"].astype("string").str.strip().str.upper()
    reg.loc[reg["iso_code"].isin(["", "NAN", "NONE", "<NA>"]), "iso_code"] = pd.NA

    for country, iso_code in ISO_FIXES.items():
        reg.loc[reg["country"] == country, "iso_code"] = iso_code

    reg["region"] = reg["iso_code"].apply(iso3_to_region)
    reg["regulation"] = reg["regulation"].astype(str).str.strip().str.title()
    reg = reg.loc[reg["regulation"].isin(["Yes", "No"])].copy()
    reg["regulation_label"] = reg["regulation"].map(
        {"Yes": "Regulated", "No": "Not regulated"}
    )
    return reg


def build_dataset(
    year: int = 2023,
    gbd_path: Path | str = GBD_PATH,
    regulation_path: Path | str = REGULATION_PATH,
    output_path: Path | str = OUTPUT_PATH,
) -> pd.DataFrame:
    health = extract_death_and_daly_rates(load_gbd_data(gbd_path), year=year)
    regulation = prepare_regulation_data(regulation_path, year=year)

    final = regulation.merge(
        health,
        left_on=["country", "Year"],
        right_on=["country_for_merge", "Year"],
        how="left",
        validate="one_to_one",
    )

    final["reg_binary"] = final["regulation"].map({"No": 0, "Yes": 1}).astype("Int64")
    final["log_death_rate"] = np.where(final["death_rate"] > 0, np.log(final["death_rate"]), np.nan)
    final["log_daly_rate"] = np.where(final["daly_rate"] > 0, np.log(final["daly_rate"]), np.nan)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output_path, index=False)

    matched = int(final[["daly_rate", "death_rate"]].notna().all(axis=1).sum())
    print(f"Dataset created: {output_path}")
    print(f"Rows: {len(final)} | Complete health records: {matched}")
    return final


def summarize_by_regulation(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("regulation", observed=True)
        .agg(
            n=("country", "count"),
            daly_n=("daly_rate", "count"),
            daly_mean=("daly_rate", "mean"),
            daly_median=("daly_rate", "median"),
            daly_std=("daly_rate", "std"),
            daly_min=("daly_rate", "min"),
            daly_q1=("daly_rate", lambda x: x.quantile(0.25)),
            daly_q3=("daly_rate", lambda x: x.quantile(0.75)),
            daly_max=("daly_rate", "max"),
            death_n=("death_rate", "count"),
            death_mean=("death_rate", "mean"),
            death_median=("death_rate", "median"),
            death_std=("death_rate", "std"),
            death_min=("death_rate", "min"),
            death_q1=("death_rate", lambda x: x.quantile(0.25)),
            death_q3=("death_rate", lambda x: x.quantile(0.75)),
            death_max=("death_rate", "max"),
        )
        .reset_index()
    )
    summary["daly_iqr"] = summary["daly_q3"] - summary["daly_q1"]
    summary["death_iqr"] = summary["death_q3"] - summary["death_q1"]
    summary["daly_cv"] = summary["daly_std"] / summary["daly_mean"]
    summary["death_cv"] = summary["death_std"] / summary["death_mean"]

    if "No" in summary["regulation"].values:
        base_daly = summary.loc[summary["regulation"] == "No", "daly_mean"].iloc[0]
        base_death = summary.loc[summary["regulation"] == "No", "death_mean"].iloc[0]
        summary["daly_mean_pct_diff_vs_no"] = (summary["daly_mean"] / base_daly - 1) * 100
        summary["death_mean_pct_diff_vs_no"] = (summary["death_mean"] / base_death - 1) * 100
    else:
        summary["daly_mean_pct_diff_vs_no"] = np.nan
        summary["death_mean_pct_diff_vs_no"] = np.nan
    return summary.round(3)


def _two_groups(df: pd.DataFrame, outcome: str) -> tuple[pd.Series, pd.Series]:
    if outcome not in {"daly_rate", "death_rate"}:
        raise ValueError("outcome must be 'daly_rate' or 'death_rate'.")
    yes = df.loc[df["regulation"] == "Yes", outcome].dropna()
    no = df.loc[df["regulation"] == "No", outcome].dropna()
    return yes, no


def compute_effect_size(df: pd.DataFrame, outcome: str) -> dict[str, Any]:
    yes, no = _two_groups(df, outcome)
    if len(yes) < 2 or len(no) < 2:
        return {"valid": False, "message": "Insufficient observations for effect size."}
    pooled_variance = (
        (len(yes) - 1) * yes.var(ddof=1) + (len(no) - 1) * no.var(ddof=1)
    ) / (len(yes) + len(no) - 2)
    if not np.isfinite(pooled_variance) or pooled_variance <= 0:
        return {"valid": False, "message": "Pooled variance is unavailable or zero."}
    d = (yes.mean() - no.mean()) / np.sqrt(pooled_variance)
    magnitude = abs(d)
    interpretation = (
        "negligible" if magnitude < 0.2 else
        "small" if magnitude < 0.5 else
        "medium" if magnitude < 0.8 else
        "large"
    )
    return {"valid": True, "cohen_d": float(d), "interpretation": interpretation}


def run_group_tests(df: pd.DataFrame, outcome: str) -> dict[str, Any]:
    yes, no = _two_groups(df, outcome)
    if len(yes) < 2 or len(no) < 2:
        return {"valid": False, "message": "Insufficient observations in one or both groups."}
    t_stat, t_p = ttest_ind(yes, no, equal_var=False, nan_policy="omit")
    u_stat, u_p = mannwhitneyu(yes, no, alternative="two-sided")
    return {
        "valid": True,
        "n_yes": len(yes), "n_no": len(no),
        "mean_yes": float(yes.mean()), "mean_no": float(no.mean()),
        "median_yes": float(yes.median()), "median_no": float(no.median()),
        "welch_t_stat": float(t_stat), "welch_p_value": float(t_p),
        "mannwhitney_u_stat": float(u_stat), "mannwhitney_p_value": float(u_p),
    }


def fit_policy_regression(df: pd.DataFrame, outcome: str, robust: bool = True):
    if outcome not in {"daly_rate", "death_rate"}:
        return None, "Outcome must be 'daly_rate' or 'death_rate'."
    model_df = df.loc[(df[outcome] > 0) & df["reg_binary"].notna(), [outcome, "reg_binary"]].copy()
    if len(model_df) < 10 or model_df["reg_binary"].nunique() < 2:
        return None, "Not enough observations or variation for regression."
    y = np.log(model_df[outcome])
    X = sm.add_constant(model_df["reg_binary"].astype(float))
    fitted = sm.OLS(y, X).fit(cov_type="HC3" if robust else "nonrobust")
    return fitted, None


if __name__ == "__main__":
    build_dataset()
