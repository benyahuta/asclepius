from __future__ import annotations

"""
CMS data ingestion pipeline.

Downloads real Medicare claims data from the CMS Data API v1.
Primary dataset: Medicare Physician & Other Practitioners —
  Provider-level utilization and payment data (2023).

This dataset contains every procedure billed to Medicare Part B,
aggregated by provider and HCPCS code, including:
  - Number of services
  - Number of beneficiaries
  - Average submitted charge vs. average Medicare payment
  - Provider specialty and geography

We derive denial-relevant signals from the gap between submitted charges
and actual payments, and from known high-denial procedure categories.

Data dictionary:
https://data.cms.gov/resources/medicare-physician-other-practitioners-by-provider-and-service-data-dictionary
"""

import logging
from pathlib import Path

import pandas as pd
import requests

from config import (
    RAW_DIR,
    PROCESSED_DIR,
    MIN_CLAIM_VOLUME,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ── CMS Data API v1 endpoint ─────────────────────────────────────────
# Medicare Physician & Other Practitioners - by Provider and Service
# Found via https://data.cms.gov/data.json catalog
CMS_API_BASE = "https://data.cms.gov/data-api/v1/dataset"
DATASET_UUID = "0e9f2f2b-7bf9-451a-912c-e02e654dd725"


# ── Column mappings ───────────────────────────────────────────────────
# CMS column names (new API uses PascalCase) → our standardized names
COLUMN_MAP = {
    "Rndrng_NPI": "provider_npi",
    "Rndrng_Prvdr_Last_Org_Name": "provider_name",
    "Rndrng_Prvdr_First_Name": "provider_first_name",
    "Rndrng_Prvdr_State_Abrvtn": "state",
    "Rndrng_Prvdr_Type": "specialty",
    "HCPCS_Cd": "procedure_code",
    "HCPCS_Desc": "procedure_description",
    "Place_Of_Srvc": "place_of_service",
    "Tot_Benes": "total_beneficiaries",
    "Tot_Srvcs": "total_services",
    "Tot_Bene_Day_Srvcs": "total_beneficiary_day_services",
    "Avg_Sbmtd_Chrg": "avg_submitted_charge",
    "Avg_Mdcr_Alowd_Amt": "avg_allowed_amount",
    "Avg_Mdcr_Pymt_Amt": "avg_medicare_payment",
    "Avg_Mdcr_Stdzd_Amt": "avg_standardized_payment",
}


def fetch_cms_data(
    size: int = 5000,
    offset: int = 0,
    specialty_filter: str | None = None,
) -> pd.DataFrame:
    """
    Fetch data from CMS Data API v1.

    Args:
        size: Number of rows per request (max ~5000 per page)
        offset: Pagination offset
        specialty_filter: Optional specialty name to filter by

    Returns:
        DataFrame of results
    """
    url = f"{CMS_API_BASE}/{DATASET_UUID}/data"

    params = {
        "size": size,
        "offset": offset,
    }

    if specialty_filter:
        params["filter[Rndrng_Prvdr_Type]"] = specialty_filter

    logger.info(f"Fetching {size} rows (offset={offset})" +
                (f" [specialty={specialty_filter}]" if specialty_filter else ""))

    response = requests.get(url, params=params, timeout=120)
    response.raise_for_status()

    data = response.json()
    logger.info(f"Received {len(data)} rows")

    return pd.DataFrame(data)


def fetch_for_specialty(
    specialty: str,
    max_rows: int = 10000,
) -> pd.DataFrame:
    """
    Fetch all data for a single specialty, paginating as needed.
    """
    all_frames = []
    offset = 0
    page_size = 5000

    while offset < max_rows:
        chunk = fetch_cms_data(
            size=min(page_size, max_rows - offset),
            offset=offset,
            specialty_filter=specialty,
        )

        if chunk.empty:
            break

        all_frames.append(chunk)
        offset += len(chunk)

        if len(chunk) < page_size:
            break  # No more pages

    if not all_frames:
        return pd.DataFrame()

    return pd.concat(all_frames, ignore_index=True)


def fetch_multiple_specialties(
    specialties: list[str],
    max_rows_per_specialty: int = 10000,
) -> pd.DataFrame:
    """
    Fetch data for multiple specialties and combine.

    The CMS API filters by one specialty at a time, so we loop.
    """
    all_frames = []

    for i, specialty in enumerate(specialties):
        logger.info(f"[{i+1}/{len(specialties)}] Fetching: {specialty}")
        spec_df = fetch_for_specialty(specialty, max_rows=max_rows_per_specialty)
        if not spec_df.empty:
            all_frames.append(spec_df)
            logger.info(f"  → {len(spec_df)} rows")
        else:
            logger.warning(f"  → No data for {specialty}")

    if not all_frames:
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    logger.info(f"Total rows fetched across all specialties: {len(combined)}")
    return combined


def clean_and_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw CMS data and compute denial-relevant metrics.

    Key derived columns:
      - payment_gap_pct: (submitted - payment) / submitted
        High gap = payer is paying far less than billed, which correlates
        with partial denials and downcoding.
      - denial_risk_flag: Binary flag for claims with payment gap > 50%
      - adjustment_amount: Dollar amount of the gap per service
    """
    if df.empty:
        return df

    # Rename columns to standardized names
    df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})

    # Convert numeric columns
    numeric_cols = [
        "total_beneficiaries",
        "total_services",
        "total_beneficiary_day_services",
        "avg_submitted_charge",
        "avg_allowed_amount",
        "avg_medicare_payment",
        "avg_standardized_payment",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing critical fields
    required = ["procedure_code", "specialty", "avg_submitted_charge", "avg_medicare_payment"]
    existing_required = [c for c in required if c in df.columns]
    df = df.dropna(subset=existing_required)

    # Filter low-volume claims (noise)
    if "total_services" in df.columns:
        df = df[df["total_services"] >= MIN_CLAIM_VOLUME]

    # ── Derived metrics ───────────────────────────────────────────────
    if "avg_submitted_charge" in df.columns and "avg_medicare_payment" in df.columns:
        # Payment gap: how much less Medicare paid vs. what was billed
        df["adjustment_amount"] = df["avg_submitted_charge"] - df["avg_medicare_payment"]

        # Gap as percentage of submitted charge
        df["payment_gap_pct"] = (
            df["adjustment_amount"] / df["avg_submitted_charge"].replace(0, float("nan"))
        ) * 100

        # Allowed-to-payment gap: difference between what Medicare allows
        # and what it actually paid. This is a better denial proxy because
        # the allowed amount already strips out the chargemaster markup.
        # A gap here means Medicare approved less than expected, which
        # correlates with partial denials, downcoding, and adjustments.
        if "avg_allowed_amount" in df.columns:
            df["allowed_payment_gap"] = df["avg_allowed_amount"] - df["avg_medicare_payment"]
            df["allowed_gap_pct"] = (
                df["allowed_payment_gap"]
                / df["avg_allowed_amount"].replace(0, float("nan"))
            ) * 100
        else:
            df["allowed_gap_pct"] = 0.0
            df["allowed_payment_gap"] = 0.0

        # Denial risk flag: uses PERCENTILE-based threshold instead of
        # a flat 50%. Flags claims in the top 20% of payment gaps within
        # the dataset. This adapts to the actual data distribution rather
        # than producing meaningless flags when 86% of claims exceed 50%.
        gap_p80 = df["payment_gap_pct"].quantile(0.80)
        df["denial_risk_flag"] = (df["payment_gap_pct"] > gap_p80).astype(int)
        df["_gap_threshold"] = gap_p80  # store for reference

        # Estimated total revenue at risk — use allowed-to-payment gap
        # (the actual underpayment), not the chargemaster gap
        if "total_services" in df.columns:
            df["total_revenue_at_risk"] = df["allowed_payment_gap"] * df["total_services"]
            # Floor at zero — negative "risk" doesn't make sense
            df["total_revenue_at_risk"] = df["total_revenue_at_risk"].clip(lower=0)

    # Clean up place of service
    if "place_of_service" in df.columns:
        pos_map = {"F": "Facility", "O": "Office/Non-Facility"}
        df["place_of_service_label"] = df["place_of_service"].map(pos_map).fillna("Unknown")

    return df


def ingest_and_save(
    max_rows_per_specialty: int = 10000,
    specialties: list[str] | None = None,
) -> Path:
    """
    Full pipeline: fetch → clean → save to parquet.

    Args:
        max_rows_per_specialty: Max rows to pull per specialty
        specialties: List of specialties to fetch. If None, fetches
            a general sample without specialty filtering.

    Returns:
        Path to saved parquet file
    """
    dataset_key = "provider_service_2023"

    # Fetch
    if specialties:
        raw_df = fetch_multiple_specialties(
            specialties, max_rows_per_specialty=max_rows_per_specialty
        )
    else:
        raw_df = fetch_cms_data(size=5000, offset=0)

    if raw_df.empty:
        logger.warning("No data returned from CMS API.")
        return Path()

    # Save raw
    raw_path = RAW_DIR / f"{dataset_key}_raw.parquet"
    raw_df.to_parquet(raw_path, index=False)
    logger.info(f"Raw data saved: {raw_path} ({len(raw_df)} rows)")

    # Clean and transform
    clean_df = clean_and_transform(raw_df)

    # Save processed
    processed_path = PROCESSED_DIR / f"{dataset_key}_processed.parquet"
    clean_df.to_parquet(processed_path, index=False)
    logger.info(f"Processed data saved: {processed_path} ({len(clean_df)} rows)")

    return processed_path


# ── Specialty groupings (useful for Ember-relevant filtering) ─────────
# These map to the specialties Ember lists on their website
EMBER_RELEVANT_SPECIALTIES = [
    "Allergy/Immunology",
    "Cardiology",
    "Dermatology",
    "Endocrinology",
    "Gastroenterology",
    "Geriatric Medicine",
    "Hand Surgery",
    "Hematology/Oncology",
    "Nephrology",
    "Neurology",
    "Obstetrics/Gynecology",
    "Ophthalmology",
    "Orthopedic Surgery",
    "Otolaryngology",
    "Pain Management",
    "Plastic and Reconstructive Surgery",
    "Podiatry",
    "Internal Medicine",
    "Family Practice",
    "Psychiatry",
    "Pulmonary Disease",
    "Rheumatology",
    "Urology",
]


if __name__ == "__main__":
    output = ingest_and_save(
        max_rows_per_specialty=10000,
        specialties=EMBER_RELEVANT_SPECIALTIES,
    )
    print(f"Data saved to: {output}")
