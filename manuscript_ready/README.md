# Frozen manuscript analysis package

Created: 2026-09-02T21:54:01.256930+00:00

## Primary findings

- DALY rate: 0.8% (-18.5% to 24.7%); p=0.940; n=160.
- Death rate: 0.5% (-18.9% to 24.4%); p=0.966; n=160.

## Interpretation lock

The unadjusted DALY association attenuated substantially after adjustment. The primary and robustness confidence intervals 
include zero. Results are cross-sectional country-level associations and must not be described as causal.

## Package structure

- `tables/`: main manuscript tables in CSV form.
- `figures/`: publication-ready 300-dpi figures.
- `supplement/`: diagnostics and complete sensitivity outputs.
- `audit/`: validation checks, file hashes, and software versions.

## IHME data

The IHME source file is not included in this repository.

Authorized users should obtain the GBD 2023 extract from the IHME GBD Results Tool using the query configuration documented in:

``supplementary/ihme_verification/ihme_query_configuration.csv``

Place the downloaded file at:

``data/raw/IHME-GBD_2023_DATA.csv``

## Validation

Passed 16 of 16 automated checks. Review
`audit/validation_report.csv` before manuscript drafting.