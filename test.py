from pathlib import Path
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parent
IHME_PATH = PROJECT_ROOT / "data" / "raw" / "IHME-GBD_2023_DATA-32e3c624-1.csv"
SUPPLEMENTARY_DIR = PROJECT_ROOT / "manuscript_ready" / "supplement"
OUTPUT_PATH = SUPPLEMENTARY_DIR / "ihme_query_configuration.csv"
df = pd.read_csv(IHME_PATH)
# search_terms = [
#     "age",
#     "metric",
#     "standard",
#     "rate",
#     "measure",
# ]
# relevant_columns = [
#     column
#     for column in df.columns
#     if any(term in column.lower() for term in search_terms)
# ]
# print("Relevant columns:")
# print(relevant_columns)
# for column in relevant_columns:
#     print(f"\n--- {column} ---")
#     print(df[column].dropna().unique()[:50])
# print("#################################################################")
# print(
#     df[["age_name", "metric_name"]]
#     .drop_duplicates()
#     .sort_values(["age_name", "metric_name"])
#     .to_string(index=False)
# )
# print("#################################################################")
# unit_columns = [
#     column
#     for column in df.columns
#     if any(
#         term in column.lower()
#         for term in [
#             "unit",
#             "denominator",
#             "metric",
#             "scale",
#         ]
#     )
# ]
# print(unit_columns)
# for column in unit_columns:
#     print(f"\n--- {column} ---")
#     print(df[column].dropna().unique()[:50])
# print("#################################################################")
# configuration_terms = [
#     "risk",
#     "cause",
#     "rei",
#     "etiology",
#     "measure",
#     "metric",
#     "age",
#     "sex",
#     "location",
#     "year",
# ]
# configuration_columns = [
#     column
#     for column in df.columns
#     if any(
#         term in column.lower()
#         for term in configuration_terms
#     )
# ]
# print("Configuration columns:")
# print(configuration_columns)
# for column in configuration_columns:
#     unique_values = df[column].dropna().unique()
#     print(f"\n--- {column} ---")
#     print(f"Unique count: {len(unique_values)}")
#     print(unique_values[:100])
# print("#################################################################")
# audit_columns = [
#     column
#     for column in [
#         "rei_id",
#         "rei_name",
#         "risk_id",
#         "risk_name",
#         "cause_id",
#         "cause_name",
#         "measure_id",
#         "measure_name",
#         "metric_id",
#         "metric_name",
#         "age_id",
#         "age_name",
#         "sex_id",
#         "sex_name",
#         "year",
#     ]
#     if column in df.columns
# ]
# query_configuration = (
#     df[audit_columns]
#     .drop_duplicates()
#     .sort_values(audit_columns)
# )
# SUPPLEMENTARY_DIR.mkdir(
#     parents=True,
#     exist_ok=True,
# )
# query_configuration.to_csv(
#     OUTPUT_PATH,
#     index=False,
# )
# print("\nIHME query configuration saved to:")
# print(OUTPUT_PATH.resolve())
# print(query_configuration.to_string(index=False))
# raw_path = Path("data/raw/IHME-GBD_2023_DATA-32e3c624-1.csv")
# df = pd.read_csv(raw_path)
# print("#################################################################")
# print("All columns:")
# for column in df.columns:
#     print(column)
# possible_risk_columns = [
#     column
#     for column in df.columns
#     if any(
#         term in column.lower()
#         for term in [
#             "risk",
#             "rei",
#             "exposure",
#             "attribut",
#         ]
#     )
# ]
# print("\nPossible risk or attribution columns:")
# print(possible_risk_columns)
# for column in possible_risk_columns:
#     print(f"\n--- {column} ---")
#     print(
#         df[column]
#         .dropna()
#         .astype(str)
#         .unique()[:50]
#     )
# print("#################################################################")
# import pandas as pd
# path = "data/raw/IHME-GBD_2023_DATA-32e3c624-1.csv"
# df = pd.read_csv(path)
# print("Columns:")
# print(df.columns.tolist())
# fields = [
#     "measure_name",
#     "metric_name",
#     "rei_name",
#     "risk_name",
#     "cause_name",
#     "location_name",
#     "age_name",
#     "sex_name",
#     "year",
# ]
# for field in fields:
#     if field in df.columns:
#         print(f"\n{field}:")
#         print(df[field].dropna().unique())
print("#################################################################")
possible_uncertainty_columns = [
    column
    for column in df.columns
    if any(
        term in column.lower()
        for term in ["val", "value", "lower", "upper"]
    )
]
print(possible_uncertainty_columns)
columns_to_show = [
    column
    for column in [
        "location_name",
        "measure_name",
        "metric_name",
        "val",
        "lower",
        "upper",
    ]
    if column in df.columns
]

print(df[columns_to_show].head(20))
deaths = df[
    df["measure_name"] == "Deaths"
][[
    "location_name",
    "val",
    "lower",
    "upper",
]].rename(columns={
    "location_name": "ihme_country",
    "val": "death_rate",
    "lower": "death_rate_lower",
    "upper": "death_rate_upper",
})
dalys = df[
    df["measure_name"].str.contains(
        "DALYs",
        case=False,
        na=False,
    )
][[
    "location_name",
    "val",
    "lower",
    "upper",
]].rename(columns={
    "location_name": "ihme_country",
    "val": "daly_rate",
    "lower": "daly_rate_lower",
    "upper": "daly_rate_upper",
})
health = deaths.merge(
    dalys,
    on="ihme_country",
    how="inner",
    validate="one_to_one",
)
