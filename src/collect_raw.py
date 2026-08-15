from pathlib import Path
from datetime import datetime, timezone
import time

import pandas as pd

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait


# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = (
    "https://www.realestate.co.nz/"
    "residential/sold/auckland"
)

# --------------------------------------------
# IMPORTANT:
# Start small while testing.
#
# Page 1-2 ≈ 40 property cards.
# --------------------------------------------

START_PAGE = 1
END_PAGE = 80

WAIT_TIMEOUT = 30

# Delay between pages.
# Keep this consistent with your authorised
# rate/volume limits.
DELAY_SECONDS = 3


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


RAW_HTML_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw_html"
)


RAW_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
)


RAW_CSV_FILE = (
    RAW_DATA_DIR
    / "property_cards.csv"
)


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

def create_directories():

    RAW_HTML_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# BUILD PAGE URL
# ============================================================

def build_page_url(page_number):

    if page_number == 1:
        return BASE_URL

    return (
        f"{BASE_URL}"
        f"?page={page_number}"
    )


# ============================================================
# CREATE CHROME
# ============================================================

def create_driver():

    options = Options()

    # Keep browser visible during development.
    options.add_argument(
        "--start-maximized"
    )

    options.add_argument(
        "--disable-notifications"
    )

    # Later, after everything works,
    # you can optionally enable headless:
    #
    # options.add_argument("--headless=new")

    return webdriver.Chrome(
        options=options
    )


# ============================================================
# WAIT FOR PROPERTY CARDS
# ============================================================

def wait_for_property_cards(driver):

    wait = WebDriverWait(
        driver,
        WAIT_TIMEOUT
    )

    # Wait for the browser document.
    wait.until(
        lambda browser:
        browser.execute_script(
            "return document.readyState"
        ) == "complete"
    )

    # More important:
    # wait until property links actually exist.
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
# SAVE RENDERED HTML
# ============================================================

def save_html(
    driver,
    page_number
):

    html = driver.execute_script(
        """
        return document
            .documentElement
            .outerHTML;
        """
    )

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

    return output_file


# ============================================================
# EXTRACT RAW PROPERTY CARDS
# ============================================================

def extract_property_cards(
    driver,
    page_number,
    search_page_url
):

    elements = driver.find_elements(
        By.CSS_SELECTOR,
        'a[href*="/property/"]'
    )

    records = []

    seen_urls = set()

    collection_time = (
        datetime.now(timezone.utc)
        .isoformat()
    )

    for element in elements:

        href = element.get_attribute(
            "href"
        )

        if not href:
            continue

        # Remove duplicate links within the page.
        if href in seen_urls:
            continue

        text = " ".join(
            element.text.split()
        )

        if not text:
            continue

        lower_text = text.lower()

        # Only sold-property result cards.
        if (
            "recently sold" not in lower_text
            and
            "last sold on" not in lower_text
        ):
            continue

        seen_urls.add(
            href
        )

        records.append(
            {
                "page_number":
                    page_number,

                "search_page_url":
                    search_page_url,

                "source_url":
                    href,

                "card_text":
                    text,

                "collected_at_utc":
                    collection_time
            }
        )

    return records


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(records):

    df = pd.DataFrame(
        records
    )

    if df.empty:
        return

    df = df.drop_duplicates(
        subset=[
            "source_url",
            "card_text"
        ]
    )

    df.to_csv(
        RAW_CSV_FILE,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    create_directories()

    driver = create_driver()

    all_records = []

    try:

        for page_number in range(
            START_PAGE,
            END_PAGE + 1
        ):

            page_url = build_page_url(
                page_number
            )

            print()
            print("=" * 70)
            print(
                f"PAGE {page_number}"
            )
            print(page_url)
            print("=" * 70)

            # ----------------------------------------
            # Open page
            # ----------------------------------------

            driver.get(
                page_url
            )

            # ----------------------------------------
            # Wait for page
            # ----------------------------------------

            wait_for_property_cards(
                driver
            )

            # Small buffer for late rendering.
            time.sleep(1)

            # ----------------------------------------
            # Save raw rendered HTML
            # ----------------------------------------

            html_file = save_html(
                driver,
                page_number
            )

            print(
                "HTML saved:"
            )

            print(
                html_file
            )

            # ----------------------------------------
            # Extract cards
            # ----------------------------------------

            page_records = (
                extract_property_cards(
                    driver,
                    page_number,
                    page_url
                )
            )

            print(
                "Property cards found:",
                len(page_records)
            )

            all_records.extend(
                page_records
            )

            # ----------------------------------------
            # Save after every page.
            #
            # This means a crash on page 100 does
            # not destroy pages 1-99.
            # ----------------------------------------

            save_checkpoint(
                all_records
            )

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
    # FINAL CLEAN RAW EXPORT
    # ========================================================

    df = pd.DataFrame(
        all_records
    )

    df = df.drop_duplicates(
        subset=[
            "source_url",
            "card_text"
        ]
    )

    df.to_csv(
        RAW_CSV_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 70)
    print("RAW COLLECTION FINISHED")
    print("=" * 70)

    print(
        "Property records:",
        len(df)
    )

    print()

    print(
        "Raw CSV:"
    )

    print(
        RAW_CSV_FILE
    )


if __name__ == "__main__":
    main()