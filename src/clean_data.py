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
    Replace repeated spaces, newlines and tabs
    with one normal space.
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
    Extract the property ID from the final part
    of the property URL.

    Example:

    https://www.realestate.co.nz/property/.../0gv38b8zc

    becomes:

    0gv38b8zc
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
    Extract only an explicitly labelled sale date.

    Example:

    Last sold on 08/07/2026

    becomes:

    2026-07-08
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
    Extract ONLY a price explicitly labelled:

        (last sale price)

    Example:

        $525,000 (last sale price)

    becomes:

        525000

    Arbitrary dollar values are deliberately ignored.
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
# AREA EXTRACTION
# ============================================================

def get_area_matches(text):
    """
    Return every area-like value from the card.

    Supports:

        506m²
        506 m²
        506m2
        1.2ha
        17.1 ha
    """

    if not isinstance(text, str):
        return []

    return list(
        re.finditer(
            r"(\d[\d,]*(?:\.\d+)?)"
            r"\s*"
            r"(m²|m2|ha)",
            text,
            re.IGNORECASE
        )
    )


def extract_land_area(text):
    """
    Extract the final area value and convert
    hectares to square metres.

    The rendered DOM can contain duplicates such as:

        506m² 506m²

    Taking the final value still gives:

        506
    """

    matches = get_area_matches(
        text
    )

    if not matches:
        return None

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

        return (
            value
            * 10_000
        )

    return value


# ============================================================
# REMOVE WEBSITE UI / SALE METADATA
# ============================================================

def remove_sale_metadata(text):
    """
    Example raw card:

    Sold share Share this listing star Save this property
    Last sold on 08/07/2026
    203 Parkhurst Road, Parakai
    2 2 1 1
    506m² 506m²
    $525,000 (last sale price)

    becomes approximately:

    203 Parkhurst Road, Parakai 2 2 1 1
    """

    clean = normalise_space(
        text
    )

    # ========================================================
    # CONFIRMED/HISTORICAL SALE
    # ========================================================

    sold_match = re.search(
        r"Last\s+sold\s+on\s+"
        r"\d{1,2}/\d{1,2}/\d{4}",
        clean,
        re.IGNORECASE
    )

    if sold_match:

        # Discard website UI text before the sale date.
        clean = clean[
            sold_match.end():
        ].strip()

    else:

        # ====================================================
        # RECENT / UNCONFIRMED SALE
        # ====================================================

        recent_match = re.search(
            r"Recently\s+sold",
            clean,
            re.IGNORECASE
        )

        if recent_match:

            clean = clean[
                recent_match.end():
            ].strip()

    # ========================================================
    # REMOVE "PRICE IS NOT YET CONFIRMED"
    # ========================================================

    clean = re.sub(
        r"\s*"
        r"Price\s+is\s+not\s+yet\s+confirmed"
        r".*$",
        "",
        clean,
        flags=re.IGNORECASE
    )

    # ========================================================
    # REMOVE CONFIRMED SALE PRICE
    # ========================================================

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

    # ========================================================
    # REMOVE ONE OR MORE AREA VALUES FROM THE END
    #
    # Handles:
    #
    #     506m²
    #
    # and:
    #
    #     506m² 506m²
    # ========================================================

    clean = re.sub(
        r"(?:"
        r"\s*"
        r"\d[\d,]*(?:\.\d+)?"
        r"\s*"
        r"(?:m²|m2|ha)"
        r")+"
        r"\s*$",
        "",
        clean,
        flags=re.IGNORECASE
    )

    return normalise_space(
        clean
    )


# ============================================================
# NUMERIC DOM DEDUPLICATION
# ============================================================

def deduplicate_numeric_pairs(numbers):
    """
    Deduplicate adjacent repeated pairs.

    Examples:

        [2, 2, 1, 1]
        ->
        [2, 1]

        [4, 4, 2, 2]
        ->
        [4, 2]

        [3, 3, 2, 2, 1, 1]
        ->
        [3, 2, 1]

    If the sequence is NOT made entirely from
    duplicated adjacent pairs, leave it unchanged.

    Example:

        [4, 2]
        ->
        [4, 2]
    """

    if not numbers:

        return numbers

    # Complete duplicated pairs require an even
    # number of values.
    if len(numbers) % 2 != 0:

        return numbers

    result = []

    for index in range(
        0,
        len(numbers),
        2
    ):

        first = (
            numbers[index]
        )

        second = (
            numbers[index + 1]
        )

        if first != second:

            # Not a duplicate-pair pattern.
            return numbers

        result.append(
            first
        )

    return result


# ============================================================
# ADDRESS / SUBURB / BEDROOM / BATHROOM
# ============================================================

def parse_location_attributes(text):
    """
    Parse address, suburb, bedrooms and bathrooms.

    Example raw rendered card:

        Sold ...
        Last sold on 08/07/2026
        203 Parkhurst Road, Parakai
        2 2 1 1
        506m² 506m²
        $525,000 (last sale price)

    becomes:

        address   = 203 Parkhurst Road
        suburb    = Parakai
        bedrooms  = 2
        bathrooms = 1


    IMPORTANT:

    We deduplicate ONLY when there are at least
    FOUR trailing numeric values.

    Therefore:

        2 2 1 1
        -> 2 bedrooms, 1 bathroom

    but:

        2 2
        -> 2 bedrooms, 2 bathrooms

    because two values alone are ambiguous.
    """

    clean = remove_sale_metadata(
        text
    )

    if not clean:

        return pd.Series(
            {
                "address":
                    None,

                "suburb":
                    None,

                "bedrooms":
                    None,

                "bathrooms":
                    None,

                "raw_trailing_numbers":
                    None,

                "numeric_values_deduplicated":
                    False,

                "extra_trailing_numeric_count":
                    0,

                "ambiguous_equal_pair":
                    False,
            }
        )

    tokens = clean.split()

    trailing_numbers = []

    # ========================================================
    # EXTRACT TRAILING INTEGER VALUES
    # ========================================================

    while (
        tokens
        and
        len(trailing_numbers) < 8
        and
        re.fullmatch(
            r"\d+",
            tokens[-1]
        )
    ):

        trailing_numbers.append(
            int(
                tokens.pop()
            )
        )

    # Values were collected from right to left.
    trailing_numbers.reverse()

    original_numbers = (
        trailing_numbers.copy()
    )

    numeric_values_deduplicated = (
        False
    )

    # ========================================================
    # SAFE DOM DEDUPLICATION
    # ========================================================

    if (
        len(trailing_numbers) >= 4
        and
        len(trailing_numbers) % 2 == 0
    ):

        deduplicated = (
            deduplicate_numeric_pairs(
                trailing_numbers
            )
        )

        if (
            deduplicated
            != trailing_numbers
        ):

            trailing_numbers = (
                deduplicated
            )

            numeric_values_deduplicated = (
                True
            )

    # ========================================================
    # AMBIGUOUS TWO-VALUE CASE
    #
    # Example:
    #
    #     2 2
    #
    # This might genuinely mean:
    #
    #     2 bedrooms
    #     2 bathrooms
    #
    # so DO NOT deduplicate it.
    #
    # We simply flag it for provenance/review.
    # ========================================================

    ambiguous_equal_pair = (
        len(original_numbers) == 2
        and
        original_numbers[0]
        == original_numbers[1]
    )

    # ========================================================
    # BEDROOM / BATHROOM
    # ========================================================

    bedrooms = None
    bathrooms = None

    if len(trailing_numbers) >= 1:

        bedrooms = (
            trailing_numbers[0]
        )

    if len(trailing_numbers) >= 2:

        bathrooms = (
            trailing_numbers[1]
        )

    # ========================================================
    # ANY ADDITIONAL NUMERIC VALUES
    #
    # Example:
    #
    #     3 3 2 2 1 1
    #
    # becomes:
    #
    #     3 2 1
    #
    # We use:
    #
    #     bedrooms  = 3
    #     bathrooms = 2
    #
    # and flag the remaining "1" rather than
    # guessing what it represents.
    # ========================================================

    extra_numeric_count = max(
        len(trailing_numbers) - 2,
        0
    )

    # ========================================================
    # ADDRESS / SUBURB
    # ========================================================

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

    # ========================================================
    # SAVE ORIGINAL NUMERIC VALUES FOR DEBUGGING
    # ========================================================

    raw_numbers_text = None

    if original_numbers:

        raw_numbers_text = (
            " ".join(
                str(number)
                for number
                in original_numbers
            )
        )

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

            "raw_trailing_numbers":
                raw_numbers_text,

            "numeric_values_deduplicated":
                numeric_values_deduplicated,

            "extra_trailing_numeric_count":
                extra_numeric_count,

            "ambiguous_equal_pair":
                ambiguous_equal_pair,
        }
    )


# ============================================================
# PROPERTY KEY
# ============================================================

def create_property_key(row):
    """
    Create a stable property identifier.

    Primary choice:

        property_id

    Fallback:

        source_url

    Final defensive fallback:

        address + suburb
    """

    property_id = (
        row.get(
            "property_id"
        )
    )

    if pd.notna(
        property_id
    ):

        property_id = (
            str(property_id)
            .strip()
        )

        if property_id:

            return (
                f"id:{property_id}"
            )

    source_url = (
        row.get(
            "source_url"
        )
    )

    if pd.notna(
        source_url
    ):

        source_url = (
            str(source_url)
            .strip()
        )

        if source_url:

            return (
                f"url:"
                f"{source_url.lower()}"
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
    When several observations exist for the same
    property, prefer the most complete property record.
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
    # CHECK RAW INPUT
    # ========================================================

    if not RAW_FILE.exists():

        raise FileNotFoundError(
            "\nCannot find raw data:\n"
            f"{RAW_FILE}\n\n"
            "Run collect_raw.py first."
        )

    # ========================================================
    # READ RAW PROPERTY CARDS
    # ========================================================

    df = pd.read_csv(
        RAW_FILE
    )

    raw_row_count = len(
        df
    )

    print()
    print("=" * 70)
    print("CLEANING PROPERTY DATA")
    print("=" * 70)

    print(
        "Raw rows:",
        raw_row_count
    )

    # ========================================================
    # VALIDATE RAW SCHEMA
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
    # NORMALISE RAW CARD TEXT
    # ========================================================

    df["card_text"] = (
        df["card_text"]
        .apply(
            normalise_space
        )
    )

    # Remove empty cards.
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
    # ADDRESS / SUBURB / BED / BATH
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

    # Record how these attributes were generated.
    df["bed_bath_parse_method"] = (
        "sold_card_trailing_numbers_safe_dom_deduplication"
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

    df["region"] = (
        "Auckland"
    )

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
    # TRANSACTION INFORMATION
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

    # A confirmed target requires BOTH an actual date
    # and an explicitly labelled last-sale price.
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
    # VALIDATION FLAGS
    #
    # Do NOT automatically remove these records.
    # ========================================================

    df["invalid_bedrooms"] = (
        df["bedrooms"]
        .notna()
        &
        (
            (df["bedrooms"] <= 0)
            |
            (df["bedrooms"] > 30)
        )
    )

    df["invalid_bathrooms"] = (
        df["bathrooms"]
        .notna()
        &
        (
            (df["bathrooms"] <= 0)
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

    # Additional numeric values after bedroom/bathroom
    # need review rather than guessing their meaning.
    df["bed_bath_parse_warning"] = (
        df[
            "extra_trailing_numeric_count"
        ]
        > 0
    )

    # ========================================================
    # REMOVE EXACT DUPLICATE OBSERVATIONS
    # ========================================================

    df = (
        df
        .drop_duplicates(
            subset=[
                "source_url",
                "sale_date",
                "sale_price",
                "card_text",
            ],
            keep="first"
        )
        .copy()
    )

    # ========================================================
    # COLUMN ORDER USED BY ALL_PROPERTIES AND TRANSACTIONS
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

        "raw_trailing_numbers",
        "numeric_values_deduplicated",
        "ambiguous_equal_pair",
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

    # ========================================================
    # OUTPUT 1:
    # ALL PARSED PROPERTY CARDS
    # ========================================================

    all_df = (
        df[
            all_columns
        ]
        .copy()
    )

    all_df.to_csv(
        ALL_PROPERTIES_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # OUTPUT 2:
    # CONFIRMED ACTUAL TRANSACTIONS
    # ========================================================

    confirmed = (
        df[
            df[
                "price_confirmed"
            ]
        ]
        .copy()
    )

    # Remove impossible targets only.
    confirmed = (
        confirmed[
            ~confirmed[
                "invalid_sale_price"
            ]
        ]
        .copy()
    )

    # Remove repeated copies of the same transaction.
    confirmed = (
        confirmed
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

    confirmed[
        all_columns
    ].to_csv(
        CONFIRMED_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # OUTPUT 3:
    # ONE ROW PER UNIQUE PROPERTY
    # ========================================================

    property_candidates = (
        df.copy()
    )

    property_candidates[
        "property_quality_score"
    ] = (
        property_candidates
        .apply(
            property_quality_score,
            axis=1
        )
    )

    # Prefer the most complete property record.
    # If tied, prefer the latest collection.
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
                "numeric_values_deduplicated",
                "ambiguous_equal_pair",
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
        raw_row_count
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
        int(
            (
                ~all_df[
                    "price_confirmed"
                ]
            ).sum()
        )
    )

    print(
        "Unique properties:",
        len(properties)
    )

    # ========================================================
    # MISSING FEATURES
    # ========================================================

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

    # ========================================================
    # PARSER INFORMATION
    # ========================================================

    print()
    print("-" * 70)
    print("PARSER INFORMATION")
    print("-" * 70)

    print(
        "DOM-deduplicated rows:",
        int(
            all_df[
                "numeric_values_deduplicated"
            ].sum()
        )
    )

    print(
        "Ambiguous equal pairs:",
        int(
            all_df[
                "ambiguous_equal_pair"
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

    # ========================================================
    # VALIDATION INFORMATION
    # ========================================================

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

    # ========================================================
    # FILES CREATED
    # ========================================================

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