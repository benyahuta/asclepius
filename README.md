# Medicare Denial Pattern Analysis Dashboard

An interactive dashboard that analyzes real CMS Medicare claims data to surface denial risk patterns across specialties, procedures, and geographies. Built to demonstrate how data-driven denial prevention works — the same approach used by AI-powered Revenue Cycle Management platforms.

## What it does

- **Specialty analysis**: Which medical specialties have the highest claim adjustment rates?
- **Procedure risk ranking**: Which CPT/HCPCS codes are most likely to trigger denials?
- **Geographic patterns**: How do denial rates vary by state (driven by regional MAC and LCD differences)?
- **Anomaly detection**: Statistical alerts when a specialty, procedure, or region's denial pattern deviates significantly from baseline
- **CARC code reference**: Plain-language explanations of every denial reason code with actionable billing guidance

## Data source

Real Medicare Provider Utilization and Payment Data from [CMS](https://data.cms.gov). No synthetic data. The dashboard derives denial risk signals from the gap between submitted charges and Medicare payments — a direct proxy for claim adjustments, downcoding, and partial denials.

## Setup

```bash
# Clone and install
cd ember-denial-dashboard
pip install -r requirements.txt

# Download CMS data (takes 1-2 minutes)
python scripts/setup_data.py

# Launch dashboard
streamlit run dashboard/app.py
```

### Options for data setup

```bash
# Fetch more data
python scripts/setup_data.py --max-rows 500000

# Fetch all specialties (not just the default set)
python scripts/setup_data.py --all

# Fetch a single specialty
python scripts/setup_data.py --specialty "Cardiology"
```

## Project structure

```
ember-denial-dashboard/
├── config.py                  # Project configuration and CMS API settings
├── requirements.txt
├── data/
│   ├── raw/                   # Raw CMS API responses (parquet)
│   └── processed/             # Cleaned and transformed data (parquet)
├── src/
│   ├── ingest.py              # CMS data download and transformation pipeline
│   ├── analysis.py            # Denial pattern analysis engine
│   ├── anomaly.py             # Statistical anomaly detection
│   └── carc_codes.py          # CARC/RARC denial reason code reference
├── dashboard/
│   ├── app.py                 # Main Streamlit application
│   └── components.py          # Reusable chart and UI components
└── scripts/
    └── setup_data.py          # One-time data setup script
```

## Key concepts

**Payment gap** = (submitted charge - Medicare payment) / submitted charge. A high gap indicates the payer is paying far less than billed, which correlates with denials, downcoding, and insufficient documentation.

**Denial risk flag** = binary flag for claims where the payment gap exceeds 50%. These are the claims most likely to represent actual denials or significant adjustments.

**CARC codes** (Claim Adjustment Reason Codes) = the standardized codes every U.S. insurer uses to explain why a claim was adjusted. The dashboard includes a curated reference of the most common codes with plain-language explanations.

## Why this matters for RCM

Revenue Cycle Management platforms like Ember AI use exactly these kinds of patterns — specialty-level denial rates, procedure-specific risk scores, geographic variation — to predict and prevent denials before claims are submitted. This dashboard demonstrates that analytical foundation using real public data.
