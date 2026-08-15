from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urljoin
import re
import time

import pandas as pd
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException


# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = (
    "https://www.realestate.co.nz/"
    "residential/sold/auckland"
)

START_PAGE = 1
END_PAGE = 80

WAIT_TIMEOUT = 30

# Set according to the limits/conditions of your permission.
DELAY_SECONDS = 2

# You do NOT need 80 HTML files for normal collection.
SAVE_RAW_HTML = False


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

DATA_DIR = (
    PROJECT_ROOT
    / "data"
)

RAW_DATA_DIR = (
    DATA_DIR
    / "raw"
)

RAW_HTML_DIR = (
    DATA_DIR
    / "raw_html"
)

PROPERTY_CARDS_FILE = (
    RAW_DATA_DIR
    / "property_cards.csv"
)

COLLECTION_REPORT_FILE = (
    RAW_DATA_DIR
    / "collection_report.csv"
)


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_directories():

    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if SAVE_RAW_HTML:

        RAW_HTML_DIR.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# URL
# ============================================================

def build_page_url(page_number):

    if page_number == 1:
        return BASE_URL

    return (
        f"{BASE_URL}"
        f"?page={page_number}"
    )


# ============================================================
# BROWSER
# ============================================================

def create_driver():

    options = Options()

    # Keep Chrome visible while developing/debugging.
    options.add_argument(
        "--start-maximized"
    )

    options.add_argument(
        "--disable-notifications"
    )

    # Don't use headless until you've confirmed
    # collection works properly.
    #
    # Later you can optionally enable:
    #
    # options.add_argument("--headless=new")

    driver = webdriver.Chrome(
        options=options
    )

    driver.set_page_load_timeout(
        45
    )

    return driver


# ============================================================
# WAIT FOR PAGE
# ============================================================

def wait_for_results(driver):

    wait = WebDriverWait(
        driver,
        WAIT_TIMEOUT
    )

    # Wait for normal browser loading.
    wait.until(
        lambda browser:
        browser.execute_script(
            "return document.readyState"
        ) == "complete"
    )

    # Wait until property links appear.
    wait.until(
        lambda browser:
        len(
            browser.find_elements(
                By.CSS_SELECTOR,
                'a[href*="/property/"]'
            )
        ) > 0
    )


# ============================================================
# TEXT HELPERS
# ============================================================

def normalise_text(text):

    if not text:
        return ""

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


def is_sold_text(text):

    lower = text.lower()

    return (
        "recently sold" in lower
        or
        "last sold on" in lower
    )


# ============================================================
# FIND COMPLETE PROPERTY CARD
# ============================================================

def find_property_card(
    anchor
):

    """
    Start from a property <a> element and walk upward
    through parent elements.

    We want the smallest parent that:

        1. contains sold-property information, and
        2. contains links belonging to ONE unique property.

    This is much safer than using anchor.get_text()
    because price/date/attributes may live outside
    that individual <a>.
    """

    node = anchor

    # Don't climb indefinitely.
    for _ in range(12):

        if node is None:
            break

        text = normalise_text(
            node.get_text(
                " ",
                strip=True
            )
        )

        if is_sold_text(text):

            property_links = (
                node.select(
                    'a[href*="/property/"]'
                )
            )

            unique_urls = set()

            for link in property_links:

                href = link.get(
                    "href"
                )

                if not href:
                    continue

                url = urljoin(
                    "https://www.realestate.co.nz",
                    href
                )

                unique_urls.add(
                    url
                )

            # A property card may have multiple links:
            #
            # image → same property
            # address → same property
            #
            # That is okay.
            #
            # But if this parent contains 20 different
            # property URLs, we've climbed too high.
            if len(unique_urls) == 1:

                return node

        node = node.parent

    return None


# ============================================================
# PARSE RENDERED PAGE
# ============================================================

def parse_rendered_html(
    html,
    page_number,
    page_url
):

    soup = BeautifulSoup(
        html,
        "lxml"
    )

    property_anchors = (
        soup.select(
            'a[href*="/property/"]'
        )
    )

    records = []

    seen_urls = set()

    collection_time = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    for anchor in property_anchors:

        href = anchor.get(
            "href"
        )

        if not href:
            continue

        source_url = urljoin(
            "https://www.realestate.co.nz",
            href
        )

        # Same property may have image link,
        # address link, etc.
        if source_url in seen_urls:
            continue

        card = find_property_card(
            anchor
        )

        if card is None:
            continue

        card_text = normalise_text(
            card.get_text(
                " ",
                strip=True
            )
        )

        if not is_sold_text(
            card_text
        ):
            continue

        seen_urls.add(
            source_url
        )

        records.append(
            {
                "page_number":
                    page_number,

                "search_page_url":
                    page_url,

                "source_url":
                    source_url,

                "card_text":
                    card_text,

                "collected_at_utc":
                    collection_time,
            }
        )

    return records


# ============================================================
# VIEWING INFORMATION
# ============================================================

def get_viewing_text(
    driver
):

    try:

        body_text = (
            driver.find_element(
                By.TAG_NAME,
                "body"
            ).text
        )

    except Exception:

        return None

    match = re.search(
        r"Viewing\s+"
        r"[\d,]+-[\d,]+\s+"
        r"of\s+[\d,]+\s+results",
        body_text,
        re.IGNORECASE
    )

    if match:

        return match.group(0)

    return None


# ============================================================
# OPTIONAL RAW HTML
# ============================================================

def save_raw_html(
    html,
    page_number
):

    if not SAVE_RAW_HTML:
        return

    output_file = (
        RAW_HTML_DIR
        / (
            f"auckland_sold_page_"
            f"{page_number:04d}.html"
        )
    )

    output_file.write_text(
        html,
        encoding="utf-8"
    )


# ============================================================
# SAVE PROPERTY CHECKPOINT
# ============================================================

def save_property_records(
    records
):

    if not records:
        return

    df = pd.DataFrame(
        records
    )

    # One raw card per property URL
    # for this search collection.
    df = df.drop_duplicates(
        subset=[
            "source_url"
        ],
        keep="first"
    )

    df = df.sort_values(
        [
            "page_number",
            "source_url"
        ]
    )

    df.to_csv(
        PROPERTY_CARDS_FILE,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# SAVE COLLECTION REPORT
# ============================================================

def save_report(
    report
):

    df = pd.DataFrame(
        report
    )

    df.to_csv(
        COLLECTION_REPORT_FILE,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# MAIN COLLECTION
# ============================================================

def main():

    create_directories()

    print()
    print("=" * 75)
    print("AUCKLAND SOLD PROPERTY COLLECTION")
    print("=" * 75)

    print(
        f"Pages: "
        f"{START_PAGE} → {END_PAGE}"
    )

    print(
        "Raw HTML saving:",
        SAVE_RAW_HTML
    )

    driver = create_driver()

    all_records = []

    report = []

    previous_first_property = None

    try:

        for page_number in range(
            START_PAGE,
            END_PAGE + 1
        ):

            page_url = build_page_url(
                page_number
            )

            print()
            print("=" * 75)

            print(
                f"PAGE {page_number}"
            )

            print(
                page_url
            )

            print("=" * 75)

            try:

                # ===========================================
                # OPEN PAGE
                # ===========================================

                driver.get(
                    page_url
                )

                wait_for_results(
                    driver
                )

                # Small buffer for late JavaScript rendering.
                time.sleep(1)

            except TimeoutException:

                print(
                    "ERROR: Page timed out."
                )

                report.append(
                    {
                        "page_number":
                            page_number,

                        "status":
                            "TIMEOUT",

                        "requested_url":
                            page_url,

                        "actual_url":
                            driver.current_url,

                        "viewing_text":
                            None,

                        "property_records":
                            0,

                        "unique_total":
                            len({
                                r["source_url"]
                                for r in all_records
                            }),
                    }
                )

                save_report(
                    report
                )

                continue

            # ===============================================
            # GET RENDERED HTML
            # ===============================================

            html = driver.execute_script(
                """
                return document
                    .documentElement
                    .outerHTML;
                """
            )

            # ===============================================
            # OPTIONAL ARCHIVE
            # ===============================================

            save_raw_html(
                html,
                page_number
            )

            # ===============================================
            # PARSE COMPLETE PROPERTY CARDS
            # ===============================================

            records = (
                parse_rendered_html(
                    html,
                    page_number,
                    page_url
                )
            )

            actual_url = (
                driver.current_url
            )

            viewing_text = (
                get_viewing_text(
                    driver
                )
            )

            print(
                "Actual URL:"
            )

            print(
                actual_url
            )

            print()

            print(
                "Page range:"
            )

            print(
                viewing_text
            )

            print()

            print(
                "Property records:",
                len(records)
            )

            # ===============================================
            # PAGE DIAGNOSTICS
            # ===============================================

            first_property = None
            last_property = None

            if records:

                first_property = (
                    records[0][
                        "source_url"
                    ]
                )

                last_property = (
                    records[-1][
                        "source_url"
                    ]
                )

                print()

                print(
                    "FIRST:"
                )

                print(
                    records[0][
                        "card_text"
                    ][:200]
                )

                print()

                print(
                    "LAST:"
                )

                print(
                    records[-1][
                        "card_text"
                    ][:200]
                )

            # ===============================================
            # CHECK FOR REPEATED PAGE
            # ===============================================

            repeated_page = (
                first_property
                is not None
                and
                first_property
                == previous_first_property
            )

            if repeated_page:

                print()
                print(
                    "WARNING:"
                    " This looks like the same "
                    "page as the previous page."
                )

            previous_first_property = (
                first_property
            )

            # ===============================================
            # ADD RECORDS
            # ===============================================

            all_records.extend(
                records
            )

            # ===============================================
            # UNIQUE TOTAL
            # ===============================================

            unique_urls = {
                record["source_url"]
                for record in all_records
            }

            unique_total = len(
                unique_urls
            )

            print()

            print(
                "TOTAL UNIQUE PROPERTIES:",
                unique_total
            )

            # ===============================================
            # COLLECTION REPORT
            # ===============================================

            report.append(
                {
                    "page_number":
                        page_number,

                    "status":
                        "OK",

                    "requested_url":
                        page_url,

                    "actual_url":
                        actual_url,

                    "viewing_text":
                        viewing_text,

                    "property_records":
                        len(records),

                    "first_property":
                        first_property,

                    "last_property":
                        last_property,

                    "repeated_page":
                        repeated_page,

                    "unique_total":
                        unique_total,
                }
            )

            # ===============================================
            # CHECKPOINT AFTER EVERY PAGE
            # ===============================================

            save_property_records(
                all_records
            )

            save_report(
                report
            )

            # ===============================================
            # DELAY
            # ===============================================

            if (
                page_number
                < END_PAGE
            ):

                time.sleep(
                    DELAY_SECONDS
                )

    finally:

        driver.quit()

    # ========================================================
    # FINAL SAVE
    # ========================================================

    save_property_records(
        all_records
    )

    save_report(
        report
    )

    unique_total = len({
        record["source_url"]
        for record in all_records
    })

    print()
    print("=" * 75)
    print("COLLECTION FINISHED")
    print("=" * 75)

    print(
        "Pages requested:",
        END_PAGE - START_PAGE + 1
    )

    print(
        "Unique properties:",
        unique_total
    )

    print()

    print(
        "Property data:"
    )

    print(
        PROPERTY_CARDS_FILE
    )

    print()

    print(
        "Collection report:"
    )

    print(
        COLLECTION_REPORT_FILE
    )


if __name__ == "__main__":
    main()