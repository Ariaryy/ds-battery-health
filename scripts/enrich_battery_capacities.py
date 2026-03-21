from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RAW_DATASET_PATH = ROOT / "datasets" / "user_behavior_dataset.csv"
CAPACITY_LOOKUP_PATH = ROOT / "datasets" / "device_battery_capacities.csv"
ENRICHED_DATASET_PATH = ROOT / "datasets" / "user_behavior_dataset_enriched.csv"

WIKIPEDIA_RAW_BASE = "https://en.wikipedia.org/w/index.php?title={title}&action=raw"
WIKIPEDIA_PAGE_BASE = "https://en.wikipedia.org/wiki/{title}"

DEVICE_SPECS = [
    {
        "device_model": "Google Pixel 5",
        "page_title": "Google_Pixel_5",
        "pattern": r"battery\s*=\s*4080.*?mAh|Pixel 5 has a 4080\s*mAh battery",
    },
    {
        "device_model": "OnePlus 9",
        "page_title": "OnePlus_9",
        "pattern": r"Both phones feature a 4500\s*mAh lithium polymer battery|Battery\s*=\s*4500\s*mAh",
    },
    {
        "device_model": "Xiaomi Mi 11",
        "page_title": "Xiaomi_Mi_11",
        "pattern": r"battery\s*=.*?4600mah|4600\s*mAh",
    },
    {
        "device_model": "Samsung Galaxy S21",
        "page_title": "Samsung_Galaxy_S21",
        "pattern": r"S21''': \{\{val\|4000\|u=mAh\}\}|contain non-removable 4000\s*mAh",
    },
    {
        "device_model": "iPhone 12",
        "page_title": "IPhone_12",
        "pattern": r"iPhone 12 has a .*?\((2,815|2815)\s*mAh\)|Battery\s*=\s*2815\s*mAh",
    },
]


def fetch_wikitext(page_title: str) -> str:
    url = WIKIPEDIA_RAW_BASE.format(title=page_title)
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 ds-battery-health/1.0"})
    with urlopen(request) as response:  # noqa: S310 - controlled domain constant
        wikitext = response.read().decode("utf-8")

    redirect_match = re.match(r"#REDIRECT \[\[(.+?)\]\]", wikitext, flags=re.IGNORECASE)
    if redirect_match:
        redirected_title = redirect_match.group(1).replace(" ", "_")
        return fetch_wikitext(redirected_title)
    return wikitext


def extract_capacity(wikitext: str, pattern: str) -> tuple[int, str]:
    match = re.search(pattern, wikitext, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError("Could not find battery capacity using the configured pattern.")

    matched_text = match.group(0)
    mah_match = re.search(r"(\d[\d,]*)\D{0,20}mAh|(\d[\d,]*)mah", matched_text, flags=re.IGNORECASE)
    if not mah_match:
        raise ValueError("Matched text did not contain an mAh value.")

    raw_value = next(group for group in mah_match.groups() if group is not None)
    capacity_mah = int(raw_value.replace(",", ""))
    return capacity_mah, matched_text.strip().replace("\n", " ")


def build_capacity_lookup() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in DEVICE_SPECS:
        wikitext = fetch_wikitext(spec["page_title"])
        capacity_mah, matched_text = extract_capacity(wikitext, spec["pattern"])
        rows.append(
            {
                "Device Model": spec["device_model"],
                "Battery Capacity (mAh)": capacity_mah,
                "Source Name": "Wikipedia",
                "Source URL": WIKIPEDIA_PAGE_BASE.format(title=spec["page_title"]),
                "Matched Text": matched_text,
            }
        )
    return pd.DataFrame(rows)


def enrich_dataset(raw_df: pd.DataFrame, capacity_lookup_df: pd.DataFrame) -> pd.DataFrame:
    enriched = raw_df.merge(capacity_lookup_df[["Device Model", "Battery Capacity (mAh)"]], on="Device Model", how="left")
    if enriched["Battery Capacity (mAh)"].isna().any():
        missing_models = sorted(enriched.loc[enriched["Battery Capacity (mAh)"].isna(), "Device Model"].unique().tolist())
        raise ValueError(f"Missing battery capacity for device models: {missing_models}")
    return enriched


def main() -> int:
    try:
        raw_df = pd.read_csv(RAW_DATASET_PATH)
        capacity_lookup_df = build_capacity_lookup()
        enriched_df = enrich_dataset(raw_df, capacity_lookup_df)
    except (OSError, HTTPError, URLError, ValueError, pd.errors.ParserError) as error:
        print(f"Failed to enrich battery capacities: {error}", file=sys.stderr)
        return 1

    capacity_lookup_df.to_csv(CAPACITY_LOOKUP_PATH, index=False, quoting=csv.QUOTE_MINIMAL)
    enriched_df.to_csv(ENRICHED_DATASET_PATH, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"Wrote lookup: {CAPACITY_LOOKUP_PATH}")
    print(f"Wrote enriched dataset: {ENRICHED_DATASET_PATH}")
    print(capacity_lookup_df[["Device Model", "Battery Capacity (mAh)"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
