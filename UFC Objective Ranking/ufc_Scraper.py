"""
UFC Stats Scraper
=================
Visits UFCStats.com and collects fight results from every completed event.
Saves everything to fights.csv with one row per fight.

Each row contains:
    - event name, date, location
    - fighter 1 name and fighter 2 name
    - winner
    - knockdowns, significant strikes, takedowns, submissions for each fighter
    - weight class, method, round, time

HOW TO RUN:
  1. Install dependencies:  pip install requests beautifulsoup4 pandas
  2. Run the script:        python ufc_scraper.py
  3. For a testing I set max_events=5 in the run_scraper() call at the bottom
  4. For the full scrape, we will set max_events=None
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

BASE_URL  = "http://ufcstats.com"
DELAY     = 1.2

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def get_soup(url: str) -> BeautifulSoup | None:
    """
    Visit a URL and return the parsed HTML.
    Returns None if the request fails for any reason
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        time.sleep(DELAY)
        return BeautifulSoup(response.text, "html.parser")
    except Exception as e:
        log.warning(f"Failed to fetch {url}  {e}")
        return None


def get_all_event_urls() -> list[str]:

    #Visit the completed events listing page and return every event URL

    log.info("Fetching list of all completed events...")
    soup = get_soup(f"{BASE_URL}/statistics/events/completed?page=all")
    if not soup:
        return []

    links = soup.select("td.b-statistics__table-col a")
    urls  = [link["href"] for link in links if link.get("href")]
    log.info(f"Found {len(urls)} events.")
    return urls


def get_event_metadata(soup: BeautifulSoup) -> dict:

    #Pull the event name, date, and location from the top of an event page

    metadata = {}

    title_tag = soup.select_one("h2.b-content__title span")
    metadata["event"] = title_tag.get_text(strip=True) if title_tag else "Unknown"

    detail_items = soup.select("li.b-list__box-list-item")
    for item in detail_items:
        text = item.get_text(separator=" ", strip=True)
        if text.startswith("Date:"):
            metadata["date"] = text.replace("Date:", "").strip()
        elif text.startswith("Location:"):
            metadata["location"] = text.replace("Location:", "").strip()

    return metadata


def parse_event_fights(event_url: str) -> list[dict]:
    """
    Visit a single event page and parse every fight row from the results table

    The table columns in order are:
      W/L | Fighter | Fighter | Kd | Kd | Str | Str | Td | Td | Sub | Sub |
      Weight class | Method | Method detail | Round | Time

    Each row contains data for both fighters side by side
    We split them out and return one dict per fight
    """
    soup = get_soup(event_url)
    if not soup:
        return []

    metadata = get_event_metadata(soup)
    fights   = []

    table = soup.select_one("table.b-fight-details__table")
    if not table:
        log.warning(f"No fight table found at {event_url}")
        return []

    rows = table.select("tr")

    for row in rows[1:]:
        cells = row.select("td")
        if not cells:
            continue

        paragraphs = [cell.select("p") for cell in cells]

        def text(col: int, fighter: int) -> str:
            """
            Helper to safely get text from a specific column and fighter index.
            col     = column index in the table
            fighter = 0 for fighter 1, 1 for fighter 2
            """
            try:
                tags = paragraphs[col]
                if fighter < len(tags):
                    return tags[fighter].get_text(strip=True)
                return ""
            except IndexError:
                return ""

        def single(col: int) -> str:

            #Helper to get text from a column that only has one value

            try:
                tags = paragraphs[col]
                return tags[0].get_text(strip=True) if tags else ""
            except IndexError:
                return ""

        result       = single(0)
        fighter_1    = text(1, 0)
        fighter_2    = text(1, 1)

        if not fighter_1 or not fighter_2:
            continue

        winner = fighter_1 if result.lower() == "win" else (
                 "Draw / No Contest" if result.lower() in ("draw", "nc") else "Unknown"
        )

        fight = {
            "event":           metadata.get("event", ""),
            "date":            metadata.get("date", ""),
            "location":        metadata.get("location", ""),
            "fighter_1":       fighter_1,
            "fighter_2":       fighter_2,
            "winner":          winner,
            "kd_f1":           text(2, 0),
            "kd_f2":           text(2, 1),
            "sig_str_f1":      text(3, 0),
            "sig_str_f2":      text(3, 1),
            "td_f1":           text(4, 0),
            "td_f2":           text(4, 1),
            "sub_att_f1":      text(5, 0),
            "sub_att_f2":      text(5, 1),
            "weight_class":    single(6),
            "method":          text(7, 0),
            "method_detail":   text(7, 1),
            "round":           single(8),
            "time":            single(9),
        }

        fights.append(fight)

    return fights


def run_scraper(max_events: int = None) -> pd.DataFrame:
    """
    Main function that runs the full scrape pipeline

    max_events: limit to this many events (used for quick testing to make sure everything works properly)
                Set to None to scrape everything.
    """
    event_urls = get_all_event_urls()

    if max_events:
        event_urls = event_urls[:max_events]
        log.info(f"Limited to first {max_events} events.")

    all_fights = []

    for idx, event_url in enumerate(event_urls, start=1):
        log.info(f"Event {idx}/{len(event_urls)}: {event_url}")
        fights = parse_event_fights(event_url)
        all_fights.extend(fights)
        log.info(f"  Got {len(fights)} fights (running total: {len(all_fights)})")

    df = pd.DataFrame(all_fights)
    df.to_csv("fights.csv", index=False)
    log.info(f"Done! Saved {len(df)} fights to fights.csv")
    return df


if __name__ == "__main__":
    df = run_scraper(max_events=None)

    print()
    print("Preview of fights.csv:")
    print(df.to_string(max_rows=10, max_cols=10))