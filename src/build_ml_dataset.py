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


TRANSACTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "confirmed_transactions.csv"
)


PROPERTIES_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "properties.csv"
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
# REQUIRED COLUMNS
# ============================================================

REQUIRED_TRANSACTION_COLUMNS = {
    "property_key",
    "sale_date",
    "sale_price",
    "source_url",
}


REQUIRED_PROPERTY_COLUMNS = {
    "property_key",
    "property_id",
    "address",
    "suburb",
    "region",
    "bedrooms",
    "bathrooms",
    "land_area_m2",
    "source_url",
}


# ============================================================
# VALIDATE COLUMNS
# ============================================================

def validate_columns(
    df,
    required_columns,
    filename
):

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"\n{filename} is missing "
            f"required columns:\n"
            f"{sorted(missing)}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("BUILDING ML DATASET")
    print("=" * 70)

    # ========================================================
    # CHECK INPUT FILES
    # ========================================================

    if not TRANSACTIONS_FILE.exists():

        raise FileNotFoundError(
            "\nCannot find:\n"
            f"{TRANSACTIONS_FILE}\n\n"
            "Run clean_data.py first."
        )


    if not PROPERTIES_FILE.exists():

        raise FileNotFoundError(
            "\nCannot find:\n"
            f"{PROPERTIES_FILE}\n\n"
            "Run clean_data.py first."
        )

    # ========================================================
    # READ DATA
    # ========================================================

    transactions = pd.read_csv(
        TRANSACTIONS_FILE
    )

    properties = pd.read_csv(
        PROPERTIES_FILE
    )

    print(
        "Confirmed transaction rows:",
        len(transactions)
    )

    print(
        "Unique property rows:",
        len(properties)
    )

    # ========================================================
    # VALIDATE SCHEMA
    # ========================================================

    validate_columns(
        transactions,
        REQUIRED_TRANSACTION_COLUMNS,
        "confirmed_transactions.csv"
    )

    validate_columns(
        properties,
        REQUIRED_PROPERTY_COLUMNS,
        "properties.csv"
    )

    # ========================================================
    # CLEAN TRANSACTION TYPES
    # ========================================================

    transactions["sale_date"] = (
        pd.to_datetime(
            transactions["sale_date"],
            errors="coerce"
        )
    )

    transactions["sale_price"] = (
        pd.to_numeric(
            transactions["sale_price"],
            errors="coerce"
        )
    )

    # ========================================================
    # CLEAN PROPERTY TYPES
    # ========================================================

    for column in [
        "bedrooms",
        "bathrooms",
        "land_area_m2",
    ]:

        properties[column] = (
            pd.to_numeric(
                properties[column],
                errors="coerce"
            )
        )

    # ========================================================
    # REMOVE TRANSACTIONS WITHOUT A VALID TARGET
    # ========================================================

    transactions = transactions[
        transactions["property_key"].notna()
        &
        transactions["sale_date"].notna()
        &
        transactions["sale_price"].notna()
    ].copy()

    # Remove only impossible/non-positive sale values.
    #
    # Do NOT remove expensive or cheap houses here.
    transactions = transactions[
        transactions["sale_price"] > 0
    ].copy()

    # ========================================================
    # REMOVE DUPLICATE TRANSACTIONS
    # ========================================================

    transactions = (
        transactions
        .drop_duplicates(
            subset=[
                "property_key",
                "sale_date",
                "sale_price",
            ],
            keep="first"
        )
        .copy()
    )

    # ========================================================
    # PREPARE TRANSACTION TABLE
    # ========================================================

    transactions = (
        transactions[
            [
                "property_key",
                "sale_date",
                "sale_price",
                "source_url",
            ]
        ]
        .rename(
            columns={
                "source_url":
                    "transaction_source_url"
            }
        )
    )

    # ========================================================
    # PREPARE PROPERTY TABLE
    # ========================================================

    properties = (
        properties[
            [
                "property_key",
                "property_id",

                "address",
                "suburb",
                "region",

                "bedrooms",
                "bathrooms",
                "land_area_m2",

                "source_url",
            ]
        ]
        .rename(
            columns={
                "source_url":
                    "property_source_url"
            }
        )
    )

    # Defensive check:
    # properties.csv should contain exactly one row per key.
    duplicate_property_keys = (
        properties[
            "property_key"
        ]
        .duplicated()
        .sum()
    )

    if duplicate_property_keys > 0:

        raise ValueError(
            "\nproperties.csv contains "
            f"{duplicate_property_keys} "
            "duplicate property_key values.\n"
            "Run clean_data.py again and "
            "inspect properties.csv."
        )

    # ========================================================
    # JOIN TRANSACTION + PROPERTY DATA
    # ========================================================

    df = transactions.merge(
        properties,
        on="property_key",
        how="left",
        validate="many_to_one"
    )

    # ========================================================
    # CHECK PROPERTY MATCHING
    # ========================================================

    unmatched_properties = (
        df["property_id"].isna()
        &
        df["address"].isna()
    )

    unmatched_count = int(
        unmatched_properties.sum()
    )

    if unmatched_count > 0:

        print()
        print(
            "WARNING:"
        )

        print(
            unmatched_count,
            "transactions could not be matched "
            "to usable property information."
        )

    # ========================================================
    # TIME FEATURES
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

    flood_date = pd.Timestamp(
        "2023-01-27"
    )

    df["post_2023_flood"] = (
        df["sale_date"]
        >= flood_date
    ).astype(
        "int8"
    )

    # IMPORTANT:
    #
    # post_2023_flood does NOT mean that a property
    # was flooded.
    #
    # It only means:
    #
    #     sale occurred on/after 27 January 2023
    #
    # Later we will add actual spatial variables:
    #
    #     in_flood_plain
    #     in_flood_prone_area
    #
    # and then construct an interaction such as:
    #
    #     flood_zone_after_2023
    #

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
    # FEATURE SANITY FLAGS
    #
    # Keep suspicious observations for now.
    # Do NOT silently delete them.
    # ========================================================

    df["invalid_bedrooms"] = (
        df["bedrooms"]
        .notna()
        &
        (
            (df["bedrooms"] < 0)
            |
            (df["bedrooms"] > 30)
        )
    ).astype(
        "int8"
    )

    df["invalid_bathrooms"] = (
        df["bathrooms"]
        .notna()
        &
        (
            (df["bathrooms"] < 0)
            |
            (df["bathrooms"] > 30)
        )
    ).astype(
        "int8"
    )

    df["invalid_land_area"] = (
        df["land_area_m2"]
        .notna()
        &
        (
            df["land_area_m2"] <= 0
        )
    ).astype(
        "int8"
    )

    # ========================================================
    # FINAL TRANSACTION DEDUPLICATION
    # ========================================================

    df = (
        df
        .drop_duplicates(
            subset=[
                "property_key",
                "sale_date",
                "sale_price",
            ],
            keep="first"
        )
        .copy()
    )

    # ========================================================
    # CHRONOLOGICAL ORDER
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
    # FINAL V1 COLUMN ORDER
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
        # TARGET / TRANSACTION
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
        # TEMPORAL FEATURES
        # ----------------------------------------------------
        "sale_year",
        "sale_month",
        "sale_quarter",
        "post_2023_flood",

        # ----------------------------------------------------
        # MISSINGNESS FLAGS
        # ----------------------------------------------------
        "missing_bedrooms",
        "missing_bathrooms",
        "missing_land_area",
        "missing_suburb",

        # ----------------------------------------------------
        # QUALITY FLAGS
        # ----------------------------------------------------
        "invalid_bedrooms",
        "invalid_bathrooms",
        "invalid_land_area",

        # ----------------------------------------------------
        # DATA PROVENANCE
        # ----------------------------------------------------
        "transaction_source_url",
        "property_source_url",
    ]

    df = df[
        final_columns
    ].copy()

    # ========================================================
    # CREATE OUTPUT DIRECTORY
    # ========================================================

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # SAVE
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
    print("=" * 70)
    print("ML DATASET CREATED")
    print("=" * 70)

    print(
        "Observations:",
        len(df)
    )

    if not df.empty:

        print(
            "Unique properties:",
            df["property_key"]
            .nunique()
        )

        print(
            "Date range:",
            df["sale_date"].min().date(),
            "→",
            df["sale_date"].max().date()
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

        print()
        print("-" * 70)
        print("MISSING PROPERTY FEATURES")
        print("-" * 70)

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

        print()
        print("-" * 70)
        print("QUALITY WARNINGS")
        print("-" * 70)

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

    print()
    print("-" * 70)
    print("OUTPUT")
    print("-" * 70)

    print(
        OUTPUT_FILE
    )

    print("=" * 70)


if __name__ == "__main__":
    main()