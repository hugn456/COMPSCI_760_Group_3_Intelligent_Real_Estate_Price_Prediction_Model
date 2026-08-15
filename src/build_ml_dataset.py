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


OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


OUTPUT_FILE = (
    OUTPUT_DIR
    / "ml_dataset_v1.csv"
)


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    # --------------------------------------------
    # Dates
    # --------------------------------------------

    df["sale_date"] = (
        pd.to_datetime(
            df["sale_date"],
            errors="coerce"
        )
    )

    # --------------------------------------------
    # Remove rows without target
    # --------------------------------------------

    df = df[
        df["sale_date"].notna()
        &
        df["sale_price"].notna()
    ].copy()

    # --------------------------------------------
    # Remove impossible prices only.
    #
    # Do NOT arbitrarily remove expensive houses.
    # Outlier analysis comes later.
    # --------------------------------------------

    df = df[
        df["sale_price"] > 0
    ].copy()

    # --------------------------------------------
    # Time features
    # --------------------------------------------

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

    # --------------------------------------------
    # Auckland Anniversary flood date
    # --------------------------------------------

    flood_date = pd.Timestamp(
        "2023-01-27"
    )

    df["post_2023_flood"] = (
        df["sale_date"]
        >= flood_date
    ).astype(int)

    # --------------------------------------------
    # Deduplication
    # --------------------------------------------

    df = df.drop_duplicates(
        subset=[
            "property_id",
            "sale_date",
            "sale_price"
        ]
    )

    # --------------------------------------------
    # Chronological order
    # --------------------------------------------

    df = df.sort_values(
        "sale_date"
    ).reset_index(
        drop=True
    )

    # --------------------------------------------
    # Final v1 columns
    # --------------------------------------------

    columns = [
        "property_id",
        "address",
        "suburb",
        "region",

        "sale_date",
        "sale_price",

        "bedrooms",
        "bathrooms",
        "land_area_m2",

        "sale_year",
        "sale_month",
        "sale_quarter",
        "post_2023_flood",

        "source_url"
    ]

    df = df[
        columns
    ]

    # --------------------------------------------
    # Save
    # --------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 60)

    print(
        "ML observations:",
        len(df)
    )

    if not df.empty:

        print(
            "Date range:",
            df["sale_date"].min(),
            "→",
            df["sale_date"].max()
        )

        print(
            "Median sale price:",
            f"${df['sale_price'].median():,.0f}"
        )

    print("=" * 60)

    print()
    print(
        "Saved:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()