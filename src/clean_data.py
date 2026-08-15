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
# REQUIRED INPUT COLUMNS
# ============================================================

REQUIRED_COLUMNS = {
    "page_number",
    "search_page_url",
    "source_url",
    "card_text",
    "collected_at_utc",
}


# ============================================================
# TEXT CLEANING
# ============================================================

def normalise_space(value):
    """
    Convert repeated spaces, tabs and newlines into
    a single space.
    """

    if not isinstance(value, str):
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
    """
    Extract the property ID from the last part of a
    realestate.co.nz property URL.

    Example:

    https://www.realestate.co.nz/property/.../abc123

    becomes:

    abc123
    """

    if not isinstance(url, str):
        return None

    url = url.strip()

    if not url:
        return None

    path = urlparse(
        url
    ).path.rstrip("/")

    if not path:
        return None

    property_id = (
        path.split("/")[-1]
        .strip()
    )

    return property_id or None


# ============================================================
# SALE DATE
# ============================================================

def extract_sale_date(text):
    """
    Extract only a date explicitly labelled:

        Last sold on DD/MM/YYYY
    """

    if not isinstance(text, str):
        return pd.NaT

    match = re.search(
        r"\bLast\s+sold\s+on\s+"
        r"(\d{1,2}/\d{1,2}/\d{4})\b",
        text,
        re.IGNORECASE
    )

    if not match:
        return pd.NaT

    return pd.to_datetime(
        match.group(1),
        format="%d/%m/%Y",
        errors="coerce"
    )


# ============================================================
# SALE PRICE
# ============================================================

def extract_sale_price(text):
    """
    Extract ONLY an actual value explicitly labelled:

        $1,250,000 (last sale price)

    We deliberately ignore arbitrary dollar values because
    they could be:

        - CV
        - estimated value
        - nearby sale
        - other non-target values
    """

    if not isinstance(text, str):
        return None

    match = re.search(
        r"\$([\d,]+)"
        r"\s*"
        r"\(\s*last\s+sale\s+price\s*\)",
        text,
        re.IGNORECASE
    )

    if not match:
        return None

    try:

        return int(
            match.group(1)
            .replace(",", "")
        )

    except ValueError:

        return None


# ============================================================
# LAND AREA
# ============================================================

def extract_land_area(text):
    """
    Extract land area and convert it to square metres.

    Supported examples:

        332m²
        332 m²
        332m2
        1.1ha
        1.1 ha
        17.1ha
    """

    if not isinstance(text, str):
        return None

    matches = list(
        re.finditer(
            r"(\d[\d,]*(?:\.\d+)?)"
            r"\s*"
            r"(m²|m2|ha)",
            text,
            re.IGNORECASE
        )
    )

    if not matches:
        return None

    # Use the final area value shown in the result card.
    match = matches[-1]

    try:

        value = float(
            match.group(1)
            .replace(",", "")
        )

    except ValueError:

        return None

    unit = (
        match.group(2)
        .lower()
    )

    if unit == "ha":
        return value * 10_000

    return value


# ============================================================
# REMOVE NON-PROPERTY METADATA
# ============================================================

def remove_sale_metadata(text):
    """
    Convert something like:

        Last sold on 20/07/2026
        15 Picasso Drive, West Harbour
        3 2 693m²
        $1,411,000 (last sale price)

    approximately into:

        15 Picasso Drive, West Harbour 3 2

    Also handles:

        Recently sold
        148A Aviemore Drive, Highland Park
        3 332m²Price is not yet confirmed
    """

    clean = normalise_space(
        text
    )

    # --------------------------------------------------------
    # Remove possible UI labels at beginning
    # --------------------------------------------------------

    clean = re.sub(
        r"^(?:(?:Sold|Save\s+this\s+property)\s+)*",
        "",
        clean,
        flags=re.IGNORECASE
    )

    # --------------------------------------------------------
    # Recently sold
    # --------------------------------------------------------

    clean = re.sub(
        r"^Recently\s+sold\s*",
        "",
        clean,
        flags=re.IGNORECASE
    )

    # --------------------------------------------------------
    # Last sold date
    # --------------------------------------------------------

    clean = re.sub(
        r"^Last\s+sold\s+on\s+"
        r"\d{1,2}/\d{1,2}/\d{4}"
        r"\s*",
        "",
        clean,
        flags=re.IGNORECASE
    )

    # --------------------------------------------------------
    # Unconfirmed price
    #
    # Handles both:
    #
    # 332m² Price is not yet confirmed
    #
    # and
    #
    # 332m²Price is not yet confirmed
    # --------------------------------------------------------

    clean = re.sub(
        r"\s*Price\s+is\s+not\s+yet\s+confirmed.*$",
        "",
        clean,
        flags=re.IGNORECASE
    )

    # --------------------------------------------------------
    # Confirmed last sale price
    # --------------------------------------------------------

    clean = re.sub(
        r"\s*"
        r"\$[\d,]+"
        r"\s*"
        r"\(\s*last\s+sale\s+price\s*\)"
        r".*$",
        "",
        clean,
        flags=re.IGNORECASE
    )

    # --------------------------------------------------------
    # Land area
    # --------------------------------------------------------

    clean = re.sub(
        r"\s*"
        r"\d[\d,]*(?:\.\d+)?"
        r"\s*"
        r"(?:m²|m2|ha)"
        r"\s*$",
        "",
        clean,
        flags=re.IGNORECASE
    )

    return normalise_space(
        clean
    )


# ============================================================
# ADDRESS / SUBURB / BEDROOMS / BATHROOMS
# ============================================================

def parse_location_attributes(text):
    """
    Parse the remaining result-card text.

    Example:

        148A Aviemore Drive, Highland Park 3

    becomes:

        address      = 148A Aviemore Drive
        suburb       = Highland Park
        bedrooms     = 3
        bathrooms    = missing


    Example:

        6/24 Andrew Road, Howick 2 1

    becomes:

        address      = 6/24 Andrew Road
        suburb       = Howick
        bedrooms     = 2
        bathrooms    = 1


    Bedroom/bathroom values are inferred from the current
    sold-card ordering, so the parsing method is explicitly
    recorded in the output for data provenance.
    """

    clean = remove_sale_metadata(
        text
    )

    if not clean:

        return pd.Series(
            {
                "address": None,
                "suburb": None,
                "bedrooms": None,
                "bathrooms": None,
                "extra_trailing_numeric_count": 0,
            }
        )

    tokens = clean.split()

    trailing_numbers = []

    # --------------------------------------------------------
    # Remove up to four trailing standalone integers.
    #
    # Normally:
    #
    #   first  = bedrooms
    #   second = bathrooms
    #
    # Extra values are recorded rather than silently ignored.
    # --------------------------------------------------------

    while (
        tokens
        and len(trailing_numbers) < 4
        and re.fullmatch(
            r"\d+",
            tokens[-1]
        )
    ):

        trailing_numbers.append(
            int(tokens.pop())
        )

    trailing_numbers.reverse()

    bedrooms = None
    bathrooms = None

    if len(trailing_numbers) >= 1:
        bedrooms = trailing_numbers[0]

    if len(trailing_numbers) >= 2:
        bathrooms = trailing_numbers[1]

    extra_numeric_count = max(
        len(trailing_numbers) - 2,
        0
    )

    # --------------------------------------------------------
    # Remaining tokens should be location text
    # --------------------------------------------------------

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

        address = (
            address.strip()
            or None
        )

        suburb = (
            suburb.strip()
            or None
        )

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
                bathrooms,

            "extra_trailing_numeric_count":
                extra_numeric_count,
        }
    )


# ============================================================
# PROPERTY KEY
# ============================================================

def create_property_key(row):
    """
    Create a reliable deduplication key.

    Normally property_id is used.

    If property_id parsing ever fails, fall back to the
    source URL rather than treating all missing IDs as the
    same property.
    """

    property_id = row.get(
        "property_id"
    )

    if pd.notna(
        property_id
    ):

        property_id = str(
            property_id
        ).strip()

        if property_id:

            return (
                f"id:{property_id}"
            )

    source_url = row.get(
        "source_url"
    )

    if pd.notna(
        source_url
    ):

        source_url = str(
            source_url
        ).strip()

        if source_url:

            return (
                f"url:{source_url.lower()}"
            )

    address = (
        str(
            row.get(
                "address",
                ""
            )
        )
        .strip()
        .lower()
    )

    suburb = (
        str(
            row.get(
                "suburb",
                ""
            )
        )
        .strip()
        .lower()
    )

    return (
        f"fallback:"
        f"{address}|{suburb}"
    )


# ============================================================
# PROPERTY QUALITY SCORE
# ============================================================

def property_quality_score(row):
    """
    When the same property appears more than once,
    prefer the record containing the most usable fields.
    """

    score = 0

    useful_fields = [
        "address",
        "suburb",
        "bedrooms",
        "bathrooms",
        "land_area_m2",
    ]

    for field in useful_fields:

        if pd.notna(
            row.get(field)
        ):

            score += 1

    return score


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # CREATE OUTPUT DIRECTORY
    # ========================================================

    INTERIM_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # CHECK RAW DATA EXISTS
    # ========================================================

    if not RAW_FILE.exists():

        raise FileNotFoundError(
            "\nCannot find:\n"
            f"{RAW_FILE}\n\n"
            "Run collect_raw.py first."
        )

    # ========================================================
    # READ RAW PROPERTY CARDS
    # ========================================================

    df = pd.read_csv(
        RAW_FILE
    )

    print()
    print("=" * 70)
    print("CLEANING PROPERTY DATA")
    print("=" * 70)

    print(
        "Raw rows:",
        len(df)
    )

    # ========================================================
    # VALIDATE INPUT SCHEMA
    # ========================================================

    missing_columns = (
        REQUIRED_COLUMNS
        - set(df.columns)
    )

    if missing_columns:

        raise ValueError(
            "\nproperty_cards.csv is missing "
            "required columns:\n"
            f"{sorted(missing_columns)}"
        )

    # ========================================================
    # CLEAN RAW TEXT
    # ========================================================

    df["card_text"] = (
        df["card_text"]
        .apply(
            normalise_space
        )
    )

    # Remove blank cards.
    df = df[
        df["card_text"] != ""
    ].copy()

    # ========================================================
    # PROPERTY ID
    # ========================================================

    df["property_id"] = (
        df["source_url"]
        .apply(
            extract_property_id
        )
    )

    # ========================================================
    # ADDRESS / SUBURB / BEDROOM / BATHROOM
    # ========================================================

    parsed = (
        df["card_text"]
        .apply(
            parse_location_attributes
        )
    )

    df = pd.concat(
        [
            df.reset_index(
                drop=True
            ),
            parsed.reset_index(
                drop=True
            ),
        ],
        axis=1
    )

    # Record how bed/bath was derived.
    df["bed_bath_parse_method"] = (
        "sold_card_trailing_numbers"
    )

    # ========================================================
    # PROPERTY KEY
    # ========================================================

    df["property_key"] = (
        df.apply(
            create_property_key,
            axis=1
        )
    )

    # ========================================================
    # REGION
    # ========================================================

    df["region"] = "Auckland"

    # ========================================================
    # LAND AREA
    # ========================================================

    df["land_area_m2"] = (
        df["card_text"]
        .apply(
            extract_land_area
        )
    )

    # ========================================================
    # TRANSACTION FIELDS
    # ========================================================

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

    # A usable confirmed transaction needs BOTH.
    df["price_confirmed"] = (
        df["sale_date"]
        .notna()
        &
        df["sale_price"]
        .notna()
    )

    # ========================================================
    # DATA TYPES
    # ========================================================

    numeric_columns = [
        "bedrooms",
        "bathrooms",
        "land_area_m2",
        "sale_price",
    ]

    for column in numeric_columns:

        df[column] = (
            pd.to_numeric(
                df[column],
                errors="coerce"
            )
        )

    df["page_number"] = (
        pd.to_numeric(
            df["page_number"],
            errors="coerce"
        )
    )

    df["collected_at_utc"] = (
        pd.to_datetime(
            df["collected_at_utc"],
            errors="coerce",
            utc=True
        )
    )

    # ========================================================
    # MISSING-VALUE FLAGS
    # ========================================================

    df["missing_property_id"] = (
        df["property_id"]
        .isna()
    )

    df["missing_address"] = (
        df["address"]
        .isna()
    )

    df["missing_suburb"] = (
        df["suburb"]
        .isna()
    )

    df["missing_bedrooms"] = (
        df["bedrooms"]
        .isna()
    )

    df["missing_bathrooms"] = (
        df["bathrooms"]
        .isna()
    )

    df["missing_land_area"] = (
        df["land_area_m2"]
        .isna()
    )

    # ========================================================
    # SANITY / VALIDATION FLAGS
    #
    # These flags DO NOT automatically delete the row.
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
    )

    df["invalid_land_area"] = (
        df["land_area_m2"]
        .notna()
        &
        (
            df["land_area_m2"]
            <= 0
        )
    )

    df["invalid_sale_price"] = (
        df["sale_price"]
        .notna()
        &
        (
            df["sale_price"]
            <= 0
        )
    )

    df["bed_bath_parse_warning"] = (
        df[
            "extra_trailing_numeric_count"
        ]
        > 0
    )

    # ========================================================
    # REMOVE EXACT DUPLICATE OBSERVATIONS
    # ========================================================

    df = df.drop_duplicates(
        subset=[
            "source_url",
            "sale_date",
            "sale_price",
            "card_text",
        ],
        keep="first"
    ).copy()

    # ========================================================
    # OUTPUT 1
    #
    # ALL PARSED RECORDS
    # ========================================================

    all_columns = [
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
        "price_confirmed",

        "page_number",
        "search_page_url",
        "source_url",
        "collected_at_utc",

        "bed_bath_parse_method",
        "extra_trailing_numeric_count",

        "missing_property_id",
        "missing_address",
        "missing_suburb",
        "missing_bedrooms",
        "missing_bathrooms",
        "missing_land_area",

        "invalid_bedrooms",
        "invalid_bathrooms",
        "invalid_land_area",
        "invalid_sale_price",

        "bed_bath_parse_warning",

        "card_text",
    ]

    all_df = df[
        all_columns
    ].copy()

    all_df.to_csv(
        ALL_PROPERTIES_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # OUTPUT 2
    #
    # CONFIRMED ACTUAL TRANSACTIONS ONLY
    # ========================================================

    confirmed = df[
        df["price_confirmed"]
    ].copy()

    # Remove impossible targets only.
    #
    # We are NOT removing legitimate expensive/cheap
    # properties here. Outlier analysis belongs later.
    confirmed = confirmed[
        ~confirmed[
            "invalid_sale_price"
        ]
    ].copy()

    # One identical sale should appear only once.
    confirmed = confirmed.drop_duplicates(
        subset=[
            "property_key",
            "sale_date",
            "sale_price",
        ],
        keep="first"
    )

    confirmed[
        all_columns
    ].to_csv(
        CONFIRMED_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # OUTPUT 3
    #
    # UNIQUE PROPERTY TABLE
    # ========================================================

    property_candidates = (
        df.copy()
    )

    property_candidates[
        "property_quality_score"
    ] = (
        property_candidates.apply(
            property_quality_score,
            axis=1
        )
    )

    # Prefer:
    #
    # 1. record with most property attributes
    # 2. latest collection if quality is tied
    property_candidates = (
        property_candidates
        .sort_values(
            [
                "property_quality_score",
                "collected_at_utc",
            ],
            ascending=[
                False,
                False,
            ],
            na_position="last"
        )
    )

    properties = (
        property_candidates[
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
                "bed_bath_parse_method",
            ]
        ]
        .drop_duplicates(
            subset=[
                "property_key"
            ],
            keep="first"
        )
        .copy()
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
    print("=" * 70)
    print("CLEANING COMPLETE")
    print("=" * 70)

    print(
        "Raw rows:",
        len(pd.read_csv(RAW_FILE))
    )

    print(
        "Parsed records:",
        len(all_df)
    )

    print(
        "Confirmed transactions:",
        len(confirmed)
    )

    print(
        "Unconfirmed records:",
        (
            len(all_df)
            - len(
                df[
                    df["price_confirmed"]
                ]
            )
        )
    )

    print(
        "Unique properties:",
        len(properties)
    )

    print()
    print("-" * 70)
    print("MISSING VALUES")
    print("-" * 70)

    print(
        "Missing property ID:",
        int(
            all_df[
                "missing_property_id"
            ].sum()
        )
    )

    print(
        "Missing address:",
        int(
            all_df[
                "missing_address"
            ].sum()
        )
    )

    print(
        "Missing suburb:",
        int(
            all_df[
                "missing_suburb"
            ].sum()
        )
    )

    print(
        "Missing bedrooms:",
        int(
            all_df[
                "missing_bedrooms"
            ].sum()
        )
    )

    print(
        "Missing bathrooms:",
        int(
            all_df[
                "missing_bathrooms"
            ].sum()
        )
    )

    print(
        "Missing land area:",
        int(
            all_df[
                "missing_land_area"
            ].sum()
        )
    )

    print()
    print("-" * 70)
    print("VALIDATION WARNINGS")
    print("-" * 70)

    print(
        "Invalid bedrooms:",
        int(
            all_df[
                "invalid_bedrooms"
            ].sum()
        )
    )

    print(
        "Invalid bathrooms:",
        int(
            all_df[
                "invalid_bathrooms"
            ].sum()
        )
    )

    print(
        "Invalid land area:",
        int(
            all_df[
                "invalid_land_area"
            ].sum()
        )
    )

    print(
        "Invalid sale price:",
        int(
            all_df[
                "invalid_sale_price"
            ].sum()
        )
    )

    print(
        "Bed/bath parse warnings:",
        int(
            all_df[
                "bed_bath_parse_warning"
            ].sum()
        )
    )

    print()
    print("-" * 70)
    print("FILES CREATED")
    print("-" * 70)

    print(
        ALL_PROPERTIES_FILE
    )

    print(
        CONFIRMED_FILE
    )

    print(
        PROPERTIES_FILE
    )

    print("=" * 70)


if __name__ == "__main__":
    main()