from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "confirmed_transactions.csv"
)


PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


OUTPUT_FILE = (
    PROCESSED_DIR
    / "ml_dataset_v1.csv"
)


# ============================================================
# REQUIRED INPUT COLUMNS
# ============================================================

REQUIRED_COLUMNS = {
    "property_key",
    "property_id",

    "address",
    "suburb",
    "region",

    "bedrooms",
    "bathrooms",
    "land_area_m2",

    "sale_date",
    "sale_price",

    "raw_trailing_numbers",
    "numeric_values_deduplicated",
    "ambiguous_equal_pair",
    "extra_trailing_numeric_count",
    "bed_bath_parse_warning",

    "source_url",
}


# ============================================================
# VALIDATE INPUT SCHEMA
# ============================================================

def validate_columns(df):

    missing = (
        REQUIRED_COLUMNS
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "\nconfirmed_transactions.csv "
            "is missing required columns:\n"
            f"{sorted(missing)}\n\n"
            "Run the latest clean_data.py first."
        )


# ============================================================
# CONVERT BOOLEAN-LIKE VALUES
# ============================================================

def to_boolean(series):
    """
    Safely convert values such as:

        True
        False
        "True"
        "False"
        1
        0

    into Boolean values.
    """

    if pd.api.types.is_bool_dtype(
        series
    ):
        return series.fillna(False)

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
        .fillna(False)
        .astype(bool)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print("BUILDING ML DATASET V1")
    print("=" * 72)

    # ========================================================
    # CHECK INPUT FILE
    # ========================================================

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            "\nCannot find:\n"
            f"{INPUT_FILE}\n\n"
            "Run clean_data.py first."
        )

    # ========================================================
    # LOAD CONFIRMED TRANSACTIONS
    # ========================================================

    df = pd.read_csv(
        INPUT_FILE
    )

    original_row_count = len(
        df
    )

    print(
        "Confirmed transaction rows:",
        original_row_count
    )

    # ========================================================
    # VALIDATE COLUMNS
    # ========================================================

    validate_columns(
        df
    )

    # ========================================================
    # DATA TYPES
    # ========================================================

    df["sale_date"] = (
        pd.to_datetime(
            df["sale_date"],
            errors="coerce"
        )
    )

    numeric_columns = [
        "sale_price",
        "bedrooms",
        "bathrooms",
        "land_area_m2",
        "extra_trailing_numeric_count",
    ]

    for column in numeric_columns:

        df[column] = (
            pd.to_numeric(
                df[column],
                errors="coerce"
            )
        )

    # ========================================================
    # BOOLEAN PARSER FLAGS
    # ========================================================

    boolean_columns = [
        "numeric_values_deduplicated",
        "ambiguous_equal_pair",
        "bed_bath_parse_warning",
    ]

    for column in boolean_columns:

        df[column] = (
            to_boolean(
                df[column]
            )
        )

    # ========================================================
    # REMOVE INVALID TARGET ROWS
    # ========================================================

    invalid_target = (
        df["property_key"].isna()
        |
        df["sale_date"].isna()
        |
        df["sale_price"].isna()
        |
        (df["sale_price"] <= 0)
    )

    invalid_target_count = int(
        invalid_target.sum()
    )

    df = (
        df[
            ~invalid_target
        ]
        .copy()
    )

    # ========================================================
    # REMOVE DUPLICATE TRANSACTIONS
    # ========================================================

    duplicate_transactions = (
        df.duplicated(
            subset=[
                "property_key",
                "sale_date",
                "sale_price",
            ],
            keep="first"
        )
    )

    duplicate_count = int(
        duplicate_transactions.sum()
    )

    df = (
        df[
            ~duplicate_transactions
        ]
        .copy()
    )

    # ========================================================
    # IMPORTANT:
    # HANDLE AMBIGUOUS TWO-NUMBER CASE
    #
    # Example raw DOM:
    #
    #     3 3
    #
    # This could be:
    #
    #     bedroom 3 duplicated
    #
    # rather than:
    #
    #     3 bedrooms
    #     3 bathrooms
    #
    # Because this cannot be resolved reliably from the
    # search-card DOM alone, keep bedrooms but treat the
    # inferred bathroom as unknown.
    # ========================================================

    ambiguous_count = int(
        df[
            "ambiguous_equal_pair"
        ].sum()
    )

    df.loc[
        df["ambiguous_equal_pair"],
        "bathrooms"
    ] = pd.NA

    # ========================================================
    # VALIDATION FLAGS
    # ========================================================

    df["invalid_bedrooms"] = (
        df["bedrooms"].notna()
        &
        (
            (df["bedrooms"] <= 0)
            |
            (df["bedrooms"] > 30)
        )
    )

    df["invalid_bathrooms"] = (
        df["bathrooms"].notna()
        &
        (
            (df["bathrooms"] <= 0)
            |
            (df["bathrooms"] > 30)
        )
    )

    df["invalid_land_area"] = (
        df["land_area_m2"].notna()
        &
        (
            df["land_area_m2"] <= 0
        )
    )

    # ========================================================
    # INVALID PROPERTY FEATURES -> MISSING
    #
    # Do NOT remove the entire transaction if the target
    # sale price is valid.
    # ========================================================

    df.loc[
        df["invalid_bedrooms"],
        "bedrooms"
    ] = pd.NA

    df.loc[
        df["invalid_bathrooms"],
        "bathrooms"
    ] = pd.NA

    df.loc[
        df["invalid_land_area"],
        "land_area_m2"
    ] = pd.NA

    # ========================================================
    # MISSING FEATURE FLAGS
    # ========================================================

    df["missing_bedrooms"] = (
        df["bedrooms"]
        .isna()
        .astype("int8")
    )

    df["missing_bathrooms"] = (
        df["bathrooms"]
        .isna()
        .astype("int8")
    )

    df["missing_land_area"] = (
        df["land_area_m2"]
        .isna()
        .astype("int8")
    )

    df["missing_suburb"] = (
        df["suburb"]
        .isna()
        .astype("int8")
    )

    # ========================================================
    # TEMPORAL FEATURES
    # ========================================================

    df["sale_year"] = (
        df["sale_date"]
        .dt.year
    )

    df["sale_month"] = (
        df["sale_date"]
        .dt.month
    )

    df["sale_quarter"] = (
        df["sale_date"]
        .dt.quarter
    )

    # ========================================================
    # AUCKLAND ANNIVERSARY FLOOD PERIOD
    # ========================================================

    FLOOD_DATE = pd.Timestamp(
        "2023-01-27"
    )

    df["post_2023_flood"] = (
        df["sale_date"]
        >= FLOOD_DATE
    ).astype(
        "int8"
    )

    # IMPORTANT:
    #
    # This means:
    #
    #     transaction occurred after 27 January 2023
    #
    # NOT:
    #
    #     property was flooded
    #
    # Actual flood exposure will later come from
    # Auckland Council spatial data.

    # ========================================================
    # PARSER QUALITY INDICATOR
    # ========================================================

    df["parser_review_required"] = (
        df["ambiguous_equal_pair"]
        |
        df["bed_bath_parse_warning"]
        |
        df["invalid_bedrooms"]
        |
        df["invalid_bathrooms"]
        |
        df["invalid_land_area"]
    ).astype(
        "int8"
    )

    # ========================================================
    # SORT CHRONOLOGICALLY
    # ========================================================

    df = (
        df
        .sort_values(
            [
                "sale_date",
                "property_key",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # FINAL COLUMN ORDER
    # ========================================================

    final_columns = [
        # ----------------------------------------------------
        # IDENTIFIERS
        # ----------------------------------------------------
        "property_key",
        "property_id",

        # ----------------------------------------------------
        # LOCATION
        # ----------------------------------------------------
        "address",
        "suburb",
        "region",

        # ----------------------------------------------------
        # TRANSACTION / TARGET
        # ----------------------------------------------------
        "sale_date",
        "sale_price",

        # ----------------------------------------------------
        # PROPERTY FEATURES
        # ----------------------------------------------------
        "bedrooms",
        "bathrooms",
        "land_area_m2",

        # ----------------------------------------------------
        # TIME FEATURES
        # ----------------------------------------------------
        "sale_year",
        "sale_month",
        "sale_quarter",
        "post_2023_flood",

        # ----------------------------------------------------
        # MISSINGNESS
        # ----------------------------------------------------
        "missing_bedrooms",
        "missing_bathrooms",
        "missing_land_area",
        "missing_suburb",

        # ----------------------------------------------------
        # VALIDATION FLAGS
        # ----------------------------------------------------
        "invalid_bedrooms",
        "invalid_bathrooms",
        "invalid_land_area",

        # ----------------------------------------------------
        # PARSER / DATA PROVENANCE
        # ----------------------------------------------------
        "raw_trailing_numbers",
        "numeric_values_deduplicated",
        "ambiguous_equal_pair",
        "extra_trailing_numeric_count",
        "bed_bath_parse_warning",
        "parser_review_required",

        # ----------------------------------------------------
        # SOURCE
        # ----------------------------------------------------
        "source_url",
    ]

    df = (
        df[
            final_columns
        ]
        .copy()
    )

    # ========================================================
    # CREATE OUTPUT FOLDER
    # ========================================================

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # SAVE ML DATASET
    # ========================================================

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 72)
    print("ML DATASET V1 CREATED")
    print("=" * 72)

    print(
        "Original confirmed rows:",
        original_row_count
    )

    print(
        "Invalid target rows removed:",
        invalid_target_count
    )

    print(
        "Duplicate transactions removed:",
        duplicate_count
    )

    print(
        "Final ML observations:",
        len(df)
    )

    # ========================================================
    # DATASET STATISTICS
    # ========================================================

    if not df.empty:

        print()

        print(
            "Unique properties:",
            df[
                "property_key"
            ].nunique()
        )

        print(
            "Date range:",
            df[
                "sale_date"
            ].min().date(),
            "->",
            df[
                "sale_date"
            ].max().date()
        )

        print(
            "Median sale price:",
            f"${df['sale_price'].median():,.0f}"
        )

        print(
            "Minimum sale price:",
            f"${df['sale_price'].min():,.0f}"
        )

        print(
            "Maximum sale price:",
            f"${df['sale_price'].max():,.0f}"
        )

        # ====================================================
        # MISSING FEATURES
        # ====================================================

        print()
        print("-" * 72)
        print("MISSING PROPERTY FEATURES")
        print("-" * 72)

        print(
            "Missing bedrooms:",
            int(
                df[
                    "missing_bedrooms"
                ].sum()
            )
        )

        print(
            "Missing bathrooms:",
            int(
                df[
                    "missing_bathrooms"
                ].sum()
            )
        )

        print(
            "Missing land area:",
            int(
                df[
                    "missing_land_area"
                ].sum()
            )
        )

        print(
            "Missing suburb:",
            int(
                df[
                    "missing_suburb"
                ].sum()
            )
        )

        # ====================================================
        # DOM PARSER INFORMATION
        # ====================================================

        print()
        print("-" * 72)
        print("DOM / PARSER INFORMATION")
        print("-" * 72)

        print(
            "DOM-deduplicated rows:",
            int(
                df[
                    "numeric_values_deduplicated"
                ].sum()
            )
        )

        print(
            "Ambiguous equal pairs:",
            ambiguous_count
        )

        print(
            "Bed/bath parse warnings:",
            int(
                df[
                    "bed_bath_parse_warning"
                ].sum()
            )
        )

        print(
            "Rows requiring parser review:",
            int(
                df[
                    "parser_review_required"
                ].sum()
            )
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        print()
        print("-" * 72)
        print("VALIDATION WARNINGS")
        print("-" * 72)

        print(
            "Invalid bedrooms:",
            int(
                df[
                    "invalid_bedrooms"
                ].sum()
            )
        )

        print(
            "Invalid bathrooms:",
            int(
                df[
                    "invalid_bathrooms"
                ].sum()
            )
        )

        print(
            "Invalid land area:",
            int(
                df[
                    "invalid_land_area"
                ].sum()
            )
        )

    # ========================================================
    # OUTPUT
    # ========================================================

    print()
    print("-" * 72)
    print("OUTPUT")
    print("-" * 72)

    print(
        OUTPUT_FILE
    )

    print("=" * 72)


if __name__ == "__main__":
    main()