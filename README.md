# Medicare Denial Pattern Analysis

Analyzes 30M+ real CMS Medicare claims to predict denial risk using machine learning. Built as a proof-of-concept for AI-powered Revenue Cycle Management.

## What it does

**Data pipeline:** Fetches real Medicare Provider Utilization data (2023) from the CMS Data API across 20 specialties, cleans and transforms it, computes denial-relevant metrics, and stores it as parquet.

**Two-layer analysis:** (1) Chargemaster gap (submitted charge vs. Medicare payment) for relative specialty/procedure ranking. (2) Allowed-to-payment gap (what Medicare approved vs. what it actually paid) as a direct underpayment signal.

**ML prediction model:** Random Forest classifier trained on 80K claims, tested on 20K. Achieves ROC-AUC of 0.831 predicting high allowed-to-payment gaps from 7 engineered features (procedure category, charge magnitude, place of service, specialty, geographic risk, service volume, payment ratio). Feature importances show what drives denial risk.

**Statistical analysis:** Volume-weighted aggregation (prevents small-provider bias), percentile-based risk flagging (top 20% of gaps), z-score + IQR anomaly detection across specialties and states.

**CARC denial mapping:** Data-driven inference of likely denial categories per specialty (from procedure mix analysis), cross-referenced with published MGMA/AAPC research. Agreement scoring between the two sources.

**Interactive dashboard:** 7-tab Streamlit app with auto-generated key findings, choropleth maps showing geographic deviation, diverging bar charts, model evaluation metrics, and CSV export.

## Quick start

```bash
git clone https://github.com/YOUR_USERNAME/medical-denial-analysis.git
cd medical-denial-analysis
pip3 install -r requirements.txt
python3 scripts/setup_data.py        # Downloads CMS data (~2 min)
python3 -m streamlit run dashboard/app.py
```

## Project structure

```
├── config.py                  # CMS API config and analysis thresholds
├── requirements.txt           # streamlit, pandas, plotly, scikit-learn
├── src/
│   ├── ingest.py              # CMS Data API v1 pipeline (fetch → clean → parquet)
│   ├── analysis.py            # Volume-weighted aggregation, risk scoring, key findings
│   ├── anomaly.py             # Z-score + IQR anomaly detection
│   ├── model.py               # Random Forest denial risk classifier
│   └── carc_codes.py          # CARC codes, specialty risk mapping, data-driven inference
├── dashboard/
│   ├── app.py                 # Streamlit app (7 tabs)
│   └── components.py          # Plotly charts, KPI cards, model visualization
└── scripts/
    └── setup_data.py          # CLI data download with --max-rows, --specialty args
```

## Key technical decisions

**Volume-weighted averages** instead of mean-of-means. A provider billing 10,000 claims contributes proportionally more than one billing 11. This prevents small-practice bias in specialty-level statistics.

**Percentile-based denial flagging** (p80 threshold) instead of a flat cutoff. Adapts to the actual data distribution — 50% would flag 86% of Medicare claims as "high risk" since chargemaster gaps are structurally large.

**Allowed-to-payment gap as ML target.** The chargemaster gap is a contractual discount, not a denial. The allowed-to-payment gap captures cases where Medicare paid less than it approved — a direct signal of adjustments, partial denials, and downcoding.

**Feature engineering for the classifier:** procedure category (first char of HCPCS groups drugs/surgery/radiology/E&M), log-transformed charges and volume (normalizes skewed distributions), state-level average gap (captures MAC contractor behavior), place of service (facility vs. office billing differences).

## Limitations

This dataset does not contain actual denial codes (CARCs/RARCs) or appeal outcomes. The allowed-to-payment gap is the closest available proxy. With real remittance data (835 files), the same model architecture trains on actual denial outcomes — same features, dramatically better signal. See the "How This Connects to Ember's AI Copilot" section in the dashboard for details.

## Data source

[CMS Medicare Physician & Other Practitioners — Provider and Service](https://data.cms.gov) (2023 release). Accessed via CMS Data API v1. Public data, no authentication required.
