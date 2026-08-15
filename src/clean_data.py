from pathlib import Path
from urllib.parse import urlparse
import re

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


RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "property_cards.csv"
)


INTERIM_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
)


ALL_PROPERTIES_FILE = (
    INTERIM_DIR
    / "all_properties.csv"
)


CONFIRMED_FILE = (
    INTERIM_DIR
    / "confirmed_transactions.csv"
)


PROPERTIES_FILE = (
    INTERIM_DIR
    / "properties.csv"
)


# ============================================================
# BASIC TEXT
# ============================================================

def normalise_space(value):

    if not isinstance(
        value,
        str
    ):
        return ""

    return re.sub(
        r"\s+",
        " ",
        value
    ).strip()


# ============================================================
# PROPERTY ID
# ============================================================

def extract_property_id(url):

    if not isinstance(
        url,
        str
    ):
        return None

    path = urlparse(
        url
    ).path.rstrip("/")

    if not path:
        return None

    return path.split("/")[-1]


# ============================================================
# SALE DATE
# ============================================================

def extract_sale_date(text):

    match = re.search(
        r"Last\s+sold\s+on\s+"
        r"(\d{1,2}/\d{1,2}/\d{4})",
        text,
        re.IGNORECASE
    )

    if not match:
        return None

    return pd.to_datetime(
        match.group(1),
        format="%d/%m/%Y",
        errors="coerce"
    )


# ============================================================
# ACTUAL SALE PRICE
# ============================================================

def extract_sale_price(text):

    # Important:
    # ONLY extract amounts explicitly described
    # as "(last sale price)".

    match = re.search(
        r"\$([\d,]+)"
        r"\s*"
        r"\(last\s+sale\s+price\)",
        text,
        re.IGNORECASE
    )

    if not match:
        return None

    return int(
        match.group(1)
        .replace(",", "")
    )


# ============================================================
# LAND AREA
# ============================================================

def extract_land_area(text):

    match = re.search(
        r"([\d,.]+)\s*m²",
        text,
        re.IGNORECASE
    )

    if match:

        return float(
            match.group(1)
            .replace(",", "")
        )

    match = re.search(
        r"([\d,.]+)\s*ha",
        text,
        re.IGNORECASE
    )

    if match:

        hectares = float(
            match.group(1)
            .replace(",", "")
        )

        return (
            hectares
            * 10_000
        )

    return None


# ============================================================
# CLEAN LOCATION + ATTRIBUTE TEXT
# ============================================================

def remove_sale_metadata(text):

    clean = normalise_space(
        text
    )

    # -------------------------------------------
    # Status/date at beginning
    # -------------------------------------------

    clean = re.sub(
        r"^Recently\s+sold\s+",
        "",
        clean,
        flags=re.IGNORECASE
    )

    clean = re.sub(
        r"^Last\s+sold\s+on\s+"
        r"\d{1,2}/\d{1,2}/\d{4}\s+",
        "",
        clean,
        flags=re.IGNORECASE
    )

    # -------------------------------------------
    # Missing price message
    # -------------------------------------------

    clean = re.sub(
        r"\s*Price\s+is\s+not\s+yet\s+confirmed.*$",
        "",
        clean,
        flags=re.IGNORECASE
    )

    # -------------------------------------------
    # Confirmed price
    # -------------------------------------------

    clean = re.sub(
        r"\s*\$[\d,]+"
        r"\s*\(last\s+sale\s+price\).*$",
        "",
        clean,
        flags=re.IGNORECASE
    )

    # -------------------------------------------
    # Land area at end
    # -------------------------------------------

    clean = re.sub(
        r"\s*[\d,.]+\s*(?:m²|ha)\s*$",
        "",
        clean,
        flags=re.IGNORECASE
    )

    return normalise_space(
        clean
    )


# ============================================================
# ADDRESS / SUBURB / BED / BATH
# ============================================================

def parse_location_attributes(text):

    clean = remove_sale_metadata(
        text
    )

    tokens = clean.split()

    numbers = []

    # Property attribute values occur at the
    # end of the card text after location.
    while (
        tokens
        and len(numbers) < 4
        and re.fullmatch(
            r"\d+",
            tokens[-1]
        )
    ):

        numbers.append(
            int(tokens.pop())
        )

    numbers.reverse()

    # Some rendered DOMs can contain duplicated
    # accessibility values such as:
    #
    # 4 4 2 2
    #
    # Convert that to:
    #
    # 4 bedrooms
    # 2 bathrooms
    #
    if (
        len(numbers) == 4
        and numbers[0] == numbers[1]
        and numbers[2] == numbers[3]
    ):

        numbers = [
            numbers[0],
            numbers[2]
        ]

    bedrooms = None
    bathrooms = None

    if len(numbers) >= 1:
        bedrooms = numbers[0]

    if len(numbers) >= 2:
        bathrooms = numbers[1]

    location = " ".join(
        tokens
    ).strip()

    address = None
    suburb = None

    if "," in location:

        address, suburb = (
            location.rsplit(
                ",",
                1
            )
        )

        address = address.strip()
        suburb = suburb.strip()

    elif location:

        address = location

    return pd.Series(
        {
            "address":
                address,

            "suburb":
                suburb,

            "bedrooms":
                bedrooms,

            "bathrooms":
                bathrooms
        }
    )


# ============================================================
# MAIN
# ============================================================

def main():

    INTERIM_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not RAW_FILE.exists():

        raise FileNotFoundError(
            f"Cannot find raw file:\n"
            f"{RAW_FILE}"
        )

    df = pd.read_csv(
        RAW_FILE
    )

    print(
        "Raw property cards:",
        len(df)
    )

    # -------------------------------------------
    # Basic cleaning
    # -------------------------------------------

    df["card_text"] = (
        df["card_text"]
        .apply(normalise_space)
    )

    df["property_id"] = (
        df["source_url"]
        .apply(
            extract_property_id
        )
    )

    # -------------------------------------------
    # Parse location/features
    # -------------------------------------------

    parsed = (
        df["card_text"]
        .apply(
            parse_location_attributes
        )
    )

    df = pd.concat(
        [
            df,
            parsed
        ],
        axis=1
    )

    # -------------------------------------------
    # Property fields
    # -------------------------------------------

    df["region"] = "Auckland"

    df["land_area_m2"] = (
        df["card_text"]
        .apply(
            extract_land_area
        )
    )

    # -------------------------------------------
    # Transaction fields
    # -------------------------------------------

    df["sale_date"] = (
        df["card_text"]
        .apply(
            extract_sale_date
        )
    )

    df["sale_price"] = (
        df["card_text"]
        .apply(
            extract_sale_price
        )
    )

    df["price_confirmed"] = (
        df["sale_date"].notna()
        &
        df["sale_price"].notna()
    )

    # -------------------------------------------
    # Types
    # -------------------------------------------

    df["bedrooms"] = pd.to_numeric(
        df["bedrooms"],
        errors="coerce"
    )

    df["bathrooms"] = pd.to_numeric(
        df["bathrooms"],
        errors="coerce"
    )

    df["land_area_m2"] = (
        pd.to_numeric(
            df["land_area_m2"],
            errors="coerce"
        )
    )

    df["sale_price"] = (
        pd.to_numeric(
            df["sale_price"],
            errors="coerce"
        )
    )

    # -------------------------------------------
    # Remove duplicate observations
    # -------------------------------------------

    df = df.drop_duplicates(
        subset=[
            "property_id",
            "sale_date",
            "sale_price"
        ]
    ).copy()

    # ========================================================
    # ALL RECORDS
    # ========================================================

    all_columns = [
        "property_id",
        "address",
        "suburb",
        "region",
        "bedrooms",
        "bathrooms",
        "land_area_m2",
        "sale_date",
        "sale_price",
        "price_confirmed",
        "source_url",
        "page_number",
        "collected_at_utc",
        "card_text"
    ]

    df[
        all_columns
    ].to_csv(
        ALL_PROPERTIES_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # CONFIRMED TRANSACTIONS
    # ========================================================

    confirmed = df[
        df["price_confirmed"]
    ].copy()

    confirmed[
        all_columns
    ].to_csv(
        CONFIRMED_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # PROPERTY TABLE
    # ========================================================

    properties = (
        df[
            [
                "property_id",
                "address",
                "suburb",
                "region",
                "bedrooms",
                "bathrooms",
                "land_area_m2",
                "source_url"
            ]
        ]
        .drop_duplicates(
            subset=[
                "property_id"
            ]
        )
    )

    properties.to_csv(
        PROPERTIES_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 60)

    print(
        "Total structured records:",
        len(df)
    )

    print(
        "Confirmed transactions:",
        len(confirmed)
    )

    print(
        "Unconfirmed:",
        len(df) - len(confirmed)
    )

    print(
        "Unique properties:",
        len(properties)
    )

    print("=" * 60)

    print()
    print("Created:")

    print(
        ALL_PROPERTIES_FILE
    )

    print(
        CONFIRMED_FILE
    )

    print(
        PROPERTIES_FILE
    )


if __name__ == "__main__":
    main()