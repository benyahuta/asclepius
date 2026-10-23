# asclepius

Looking for patterns in Medicare payment data that point to denials and underpayments, plus a model that predicts which claims are at risk.

## What it does

- Pulls 2023 Medicare provider and service data from the CMS Data API across 20 specialties
- Compares what Medicare allowed vs. what it actually paid, as a stand-in for denials and downcoding
- Trains a random forest to predict which claims have a big allowed-to-paid gap
- Flags unusual specialties and states with z-scores and IQR
- Shows it all in a Streamlit dashboard

The public data doesn't include real denial codes, so the allowed-to-paid gap is a proxy. With real remittance (835) data the same setup could train on actual denials.

## Running it

```bash
pip3 install -r requirements.txt
python3 scripts/setup_data.py
python3 -m streamlit run dashboard/app.py
```

Data is from [CMS](https://data.cms.gov) and is public.
