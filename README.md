# Global Lead Paint Regulation and Health Burden

An interactive environmental-health dashboard and reproducible country-level analysis examining associations between legally binding lead paint regulation and lead-attributable health burden in 2023.

## Project overview

This project integrates:

- country-level lead paint regulation status;
- 2023 lead-attributable disability-adjusted life-year (DALY) rates;
- 2023 lead-attributable death rates;
- economic, demographic, health-system, industrial, and environmental covariates from the World Bank;
- staged multivariable regression models;
- uncertainty and robustness analyses;
- an interactive Dash dashboard for research communication.

The study is a cross-sectional ecological analysis. Results describe country-level associations and must not be interpreted as causal effects of regulation.

## Main findings

In the unadjusted analysis, regulated countries had an estimated 19.9% lower DALY rate than countries without legally binding regulation. After adjustment for log GDP per capita, population aged 65 years and older, urbanization, and World Bank region, the estimated difference was 3.3% lower, with a 95% confidence interval from 10.0% lower to 3.9% higher.

For death rates, the primary adjusted estimate was 3.7% lower in regulated countries, with a 95% confidence interval from 12.1% lower to 5.5% higher.

Bootstrap, robust regression, quantile regression, Gamma regression, geographic exclusions, influential-country analyses, and temporal restrictions produced the same overall interpretation: adjusted estimates were modestly negative but statistically uncertain.

## Important interpretation limits

The binary regulation indicator identifies the presence or absence of legally binding lead paint controls. It does not measure:

- implementation quality;
- enforcement intensity;
- compliance;
- legal thresholds;
- years since adoption;
- policy strength;
- paint-specific lead exposure reduction.

The health outcomes represent lead-attributable burden and are not necessarily attributable specifically to lead paint. Country-level IHME uncertainty bounds were not present in the analytical extract and therefore were not propagated.

## Project structure

```text
.
├── app.py
├── data_pipeline.py
├── covariate_pipeline.py
├── statistical_analysis.py
├── robustness_analysis.py
├── step6_dashboard_upgrade.py
├── manuscript_package.py
├── requirements.txt
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── lead_policy_merged.csv
│   │   └── lead_policy_expanded.csv
│   └── external_covariates/
├── results/
│   ├── statistical_analysis/
│   └── robustness_analysis/
├── figures/
│   └── manuscript/
├── manuscript_ready/
└── assets/
```

## IHME data

The IHME source file is not included in this repository.

Authorized users should obtain the GBD 2023 extract from the IHME GBD Results Tool using the query configuration documented in:

``supplementary/ihme_verification/ihme_query_configuration.csv``

Place the downloaded file at:

``data/raw/IHME-GBD_2023_DATA.csv``

## Analytical workflow

### 1. Build the base dataset

```bash
python data_pipeline.py
```

This script:

- loads the regulation and IHME data;
- selects 2023, both sexes, all ages, and rate metrics;
- extracts DALY and death rates;
- harmonizes country names and ISO-3 codes;
- merges regulation status with health outcomes;
- creates binary and log-transformed analysis variables;
- writes `data/processed/lead_policy_merged.csv`.

### 2. Add external covariates

```bash
python covariate_pipeline.py
```

This script retrieves World Bank covariates using values from 2019 through 2022 and selects the latest nonmissing value on or before 2022. It creates:

- `data/external_covariates/world_bank_covariates.csv`;
- `data/external_covariates/world_bank_covariate_provenance.csv`;
- `data/external_covariates/world_bank_covariate_coverage.csv`;
- `data/processed/lead_policy_expanded.csv`.

The covariates include GDP per capita, urbanization, population aged 65 years and older, health expenditure, manufacturing share, and PM2.5 exposure. The UHC service coverage indicator was unavailable under the selected time window and was excluded from regression modeling.

### 3. Run the staged statistical analysis

```bash
python statistical_analysis.py
```

The script fits models for DALY and death rates using HC3 heteroskedasticity-robust standard errors:

- **M0:** regulation only;
- **M1:** regulation and log GDP per capita;
- **M2 primary:** regulation, log GDP per capita, population aged 65 years and older, urbanization, and World Bank region;
- **M3:** health expenditure replaces GDP;
- **M4:** primary model plus PM2.5;
- **M5:** primary model plus manufacturing share.

Continuous covariates are standardized. Regulation effects are transformed to percentage differences using:

```text
100 × [exp(beta) - 1]
```

Outputs are written to `results/statistical_analysis/`.

### 4. Run uncertainty and robustness analyses

```bash
python robustness_analysis.py
```

The robustness framework includes:

- conventional, HC0, HC1, HC2, and HC3 covariance estimators;
- 2,000 country-level bootstrap replicates;
- exploratory World Bank region-clustered inference;
- leave-one-region-out analysis;
- influential-country analysis using Cook's distance;
- Huber robust regression;
- quantile regression at the 25th, 50th, and 75th percentiles;
- an alternative six-continent regional definition;
- Gamma generalized linear models with a log link;
- restriction to primary covariates measured in 2022;
- optional IHME uncertainty propagation when lower and upper bounds are available.

Outputs are written to `results/robustness_analysis/`.

### 5. Create the frozen manuscript package

```bash
python manuscript_package.py
```

This step validates and freezes the finalized analysis without refitting the models. It creates manuscript tables, publication figures, supplementary outputs, file hashes, an analysis lock, and a software-environment record under `manuscript_ready/`.

### 6. Start the dashboard

```bash
python app.py
```

Open the local address shown in the terminal, normally:

```text
http://127.0.0.1:8050/
```

The dashboard contains four tabs:

1. Visual Overview
2. Unadjusted Group Comparisons
3. Unadjusted Regression Modeling
4. Adjusted Analysis & Robustness

## Installation

Python 3.10 or later is recommended.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Recommended `requirements.txt` contents:

```text
pandas>=1.5
numpy>=1.23
scipy>=1.10
requests>=2.31
plotly>=5.18
dash>=2.14
matplotlib>=3.7
pycountry>=23.12.11
pycountry-convert>=0.7.2
statsmodels>=0.14
```

`seaborn`, `jupyterlab`, and `ipykernel` are not required by the current Python scripts. Keep them only in an optional development requirements file if notebooks are used.

For exact environment reproducibility after a successful run:

```bash
python -m pip freeze > requirements-lock.txt
```

## Reproducing the complete project

From the project root, run:

```bash
python data_pipeline.py
python covariate_pipeline.py
python statistical_analysis.py
python robustness_analysis.py
python manuscript_package.py
python app.py
```

## Principal outputs

### Statistical analysis

```text
results/statistical_analysis/
├── all_model_coefficients.csv
├── regulation_model_summary.csv
├── model_sample_sizes.csv
├── standardization_parameters.csv
├── vif_diagnostics.csv
├── primary_model_influence.csv
├── influential_country_sensitivity.csv
└── residual_normality_tests.csv
```

### Robustness analysis

```text
results/robustness_analysis/
├── robustness_summary.csv
├── covariance_estimator_sensitivity.csv
├── country_bootstrap_summary.csv
├── country_bootstrap_draws.csv
├── region_clustered_inference.csv
├── leave_one_region_out.csv
├── influence_sensitivity.csv
├── huber_robust_regression.csv
├── quantile_regression.csv
├── alternative_region_definition.csv
├── outcome_scale_sensitivity.csv
├── temporal_alignment_sensitivity.csv
├── ihme_uncertainty_summary.csv
└── ihme_uncertainty_draws.csv
```

## Data provenance

The analytical dataset combines user-supplied regulation and IHME extracts with World Bank World Development Indicators obtained through the World Bank API. Consult the following generated files for indicator definitions, years, transformations, selection rules, and coverage:

```text
data/external_covariates/world_bank_covariate_provenance.csv
data/external_covariates/world_bank_covariate_coverage.csv
```

Raw source data should be retained unchanged. Derived datasets and result files should be regenerated through the documented scripts.

## Reproducibility and version control

- Python source files;
- `README.md`;
- `requirements.txt`;
- `requirements-lock.txt` when the environment is frozen;
- small processed outputs where licensing permits;
- manuscript tables and figures;
- provenance and validation records.

## Citation and License

The software source code is licensed under the MIT License. See `LICENSE` for the full terms.

For citation metadata, see `CITATION.cff`.

Third-party software and dependencies remain subject to their respective licenses. Research figures, tables, documentation, and any other generated files are included for scholarly reproducibility; no additional rights are granted unless stated separately.

## Contact

- Author name: `Mohammad Hijazi`
- Institutional affiliation: `Lebanese American University, School of Arts and Sciences`
- Email address: `mohamad.hijazi05@lau.edu`
- Repository URL: `https://github.com/MohammadHijazi22/global-lead-paint-policy-health`
