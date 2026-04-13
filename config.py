"""
Project configuration.

CMS data is accessed via the CMS Data API v1 (public, no token required).
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

for d in [RAW_DIR, PROCESSED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Analysis settings ─────────────────────────────────────────────────
# Minimum claim volume to include a provider/service in analysis
MIN_CLAIM_VOLUME = 10

# Z-score threshold for anomaly detection
ANOMALY_Z_THRESHOLD = 2.0

# Top N denial reasons to display in dashboard
TOP_N_DENIAL_REASONS = 15

# Top N procedures to display
TOP_N_PROCEDURES = 20
