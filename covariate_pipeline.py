"""Acquire and merge prespecified World Bank covariates.
This module uses only observations from 2019-2022 for the 2023 cross-sectional outcome analysis
and selects the latest value available on or before 2022 for every country-indicator pair.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import json, time, requests
import numpy as np
import pandas as pd
from data_pipeline import PROJECT_ROOT

PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "lead_policy_merged.csv"
EXPANDED_PATH = PROJECT_ROOT / "data" / "processed" / "lead_policy_expanded.csv"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external_covariates"
COVARIATE_PATH = EXTERNAL_DIR / "world_bank_covariates.csv"
PROVENANCE_PATH = EXTERNAL_DIR / "world_bank_covariate_provenance.csv"
COVERAGE_PATH = EXTERNAL_DIR / "world_bank_covariate_coverage.csv"
WB_API = "https://api.worldbank.org/v2"
START_YEAR = 2019
REFERENCE_YEAR = 2022
REQUEST_TIMEOUT = 60

@dataclass(frozen=True)
class IndicatorSpec:
    code: str
    column: str
    label: str
    role: str
    transform: str = "none"

INDICATORS = [
    IndicatorSpec(
        "NY.GDP.PCAP.PP.KD", "gdp_per_capita_ppp_2021", 
        "GDP per capita, PPP (constant 2021 international $)",
        "Economic development confounder", "log1p",
    ),
    IndicatorSpec(
        "SP.URB.TOTL.IN.ZS", "urban_population_pct",
        "Urban population (% of total population)",
        "Urbanization and exposure-context confounder",
    ),
    IndicatorSpec(
        "SP.POP.65UP.TO.ZS", "population_age_65plus_pct",
        "Population ages 65 and above (% of total population)",
        "Demographic confounder for mortality and DALYs",
    ),
    IndicatorSpec(
        "SH.XPD.CHEX.PC.CD", "health_expenditure_per_capita_usd",
        "Current health expenditure per capita (current US$)",
        "Health-system capacity confounder", "log1p",
    ),
    IndicatorSpec(
        "SH.UHC.SRVS.CV.XD", "uhc_service_coverage_index",
        "UHC service coverage index",
        "Health-system access and coverage confounder",
    ),
    IndicatorSpec(
        "NV.IND.MANF.ZS", "manufacturing_value_added_pct_gdp",
        "Manufacturing, value added (% of GDP)",
        "Industrial activity proxy",
    ),
    IndicatorSpec(
        "EN.ATM.PM25.MC.M3", "pm25_mean_annual_ug_m3",
        "PM2.5 air pollution, mean annual exposure (micrograms per cubic meter)",
        "Environmental pollution-context proxy",
    ),
]

def _request_json(url: str, params: dict[str, Any], retries: int = 3) -> Any:
    last_error = None
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"World Bank API request failed after {retries} attempts: {last_error}")

def fetch_indicator(spec: IndicatorSpec) -> pd.DataFrame:
    """Fetch all economies for one indicator and retain country-level values."""
    url = f"{WB_API}/country/all/indicator/{spec.code}"
    payload = _request_json(
        url,
        {
            "format": "json",
            "date": f"{START_YEAR}:{REFERENCE_YEAR}",
            "per_page": 20000,
        },
    )
    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        return pd.DataFrame(columns=["iso_code", "year", "value"])
    records = []
    for item in payload[1]:
        iso3 = item.get("countryiso3code")
        value = item.get("value")
        year = item.get("date")
        if iso3 and len(iso3) == 3 and value is not None:
            records.append({"iso_code": iso3, "year": int(year), "value": float(value)})
    return pd.DataFrame(records)

def select_latest_pre_outcome(raw: pd.DataFrame, spec: IndicatorSpec) -> pd.DataFrame:
    """Select latest nonmissing value on or before REFERENCE_YEAR by country."""
    if raw.empty:
        return pd.DataFrame(columns=["iso_code", spec.column, f"{spec.column}_year", f"{spec.column}_lag"])
    selected = (
        raw.sort_values(["iso_code", "year"], ascending=[True, False])
        .drop_duplicates("iso_code", keep="first")
        .rename(columns={"value": spec.column, "year": f"{spec.column}_year"})
    )
    selected[f"{spec.column}_lag"] = REFERENCE_YEAR - selected[f"{spec.column}_year"]
    return selected[["iso_code", spec.column, f"{spec.column}_year", f"{spec.column}_lag"]]

def fetch_country_metadata() -> pd.DataFrame:
    """Fetch World Bank economy metadata, including income classification."""
    payload = _request_json(
        f"{WB_API}/country",
        {"format": "json", "per_page": 400}, # instructs the API to return up to 400 country records in a single page   
    )
    if not isinstance(payload, list) or len(payload) < 2:
        raise RuntimeError("World Bank country metadata response was invalid.")
    rows = []
    for item in payload[1]:
        iso3 = item.get("id")
        region = (item.get("region") or {}).get("value")
        income = (item.get("incomeLevel") or {}).get("value")
        if iso3 and len(iso3) == 3 and region != "Aggregates":
            rows.append({
                "iso_code": iso3,
                "wb_country_name": item.get("name"),
                "wb_region": region,
                "income_group": income,
                "lending_type": (item.get("lendingType") or {}).get("value"),
            })
    return pd.DataFrame(rows)

def build_covariate_dataset() -> pd.DataFrame:
    """Download, harmonize, report, and merge the prespecified covariates."""
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(f"Base analytical dataset not found: {PROCESSED_PATH}")
    base = pd.read_csv(PROCESSED_PATH)
    base["iso_code"] = base["iso_code"].astype("string").str.upper().str.strip()
    country_codes = set(base["iso_code"].dropna())
    covariates = fetch_country_metadata()
    covariates = covariates[covariates["iso_code"].isin(country_codes)].copy()
    provenance_rows = []
    for spec in INDICATORS:
        print(f"Fetching {spec.code}: {spec.label}")
        raw = fetch_indicator(spec)
        selected = select_latest_pre_outcome(raw, spec)
        covariates = covariates.merge(selected, on="iso_code", how="outer", validate="one_to_one")
        provenance_rows.append({
            "indicator_code": spec.code,
            "column_name": spec.column,
            "indicator_label": spec.label,
            "analytical_role": spec.role,
            "planned_transform": spec.transform,
            "source": "World Bank World Development Indicators API",
            "reference_year": REFERENCE_YEAR,
            "allowed_year_window": f"{START_YEAR}-{REFERENCE_YEAR}",
            "selection_rule": "Latest nonmissing value on or before reference year",
        })
    covariates = covariates[covariates["iso_code"].isin(country_codes)].copy()
    # Derived analysis variables. Continuous originals are preserved.
    if "gdp_per_capita_ppp_2021" in covariates:
        covariates["log_gdp_per_capita_ppp"] = np.log1p(covariates["gdp_per_capita_ppp_2021"]) # np.log1p -> np's ln (1 + x)
    if "health_expenditure_per_capita_usd" in covariates:
        covariates["log_health_expenditure_per_capita"] = np.log1p(covariates["health_expenditure_per_capita_usd"])
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    covariates.to_csv(COVARIATE_PATH, index=False)
    pd.DataFrame(provenance_rows).to_csv(PROVENANCE_PATH, index=False)
    analytical_columns = [spec.column for spec in INDICATORS]
    coverage_rows = []
    for column in analytical_columns:
        available = int(covariates[column].notna().sum()) if column in covariates else 0
        coverage_rows.append({
            "variable": column,
            "available_n": available,
            "base_n": len(base),
            "coverage_pct": round(100 * available / len(base), 1),
            "missing_n": len(base) - available,
        })
    coverage = pd.DataFrame(coverage_rows).sort_values("coverage_pct", ascending=False)
    coverage.to_csv(COVERAGE_PATH, index=False)
    expanded = base.merge(covariates, on="iso_code", how="left", validate="one_to_one")
    expanded.to_csv(EXPANDED_PATH, index=False)
    print(f"Covariates written to: {COVARIATE_PATH}")
    print(f"Coverage report written to: {COVERAGE_PATH}")
    print(f"Expanded analytical dataset written to: {EXPANDED_PATH}")
    print("\nCoverage summary:")
    print(coverage.to_string(index=False))
    return expanded

if __name__ == "__main__":
    build_covariate_dataset()