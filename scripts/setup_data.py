"""
One-time data setup script.

Downloads and processes CMS Medicare claims data for the dashboard.
Run this before launching the Streamlit app.

Usage:
    cd ember-denial-dashboard
    pip install -r requirements.txt
    python scripts/setup_data.py

Options:
    --max-rows N     Max rows per specialty (default: 10000)
    --all            Fetch a general sample instead of specialty-filtered
    --specialty X    Fetch only a specific specialty (e.g., "Cardiology")
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest import ingest_and_save, EMBER_RELEVANT_SPECIALTIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Download CMS data for denial dashboard")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=10000,
        help="Maximum rows per specialty to fetch (default: 10000)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Fetch a general sample without specialty filtering",
    )
    parser.add_argument(
        "--specialty",
        type=str,
        default=None,
        help="Fetch only a specific specialty",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Ember Denial Dashboard — Data Setup")
    logger.info("=" * 60)

    # Determine specialty filter
    if args.specialty:
        specialties = [args.specialty]
        logger.info(f"Fetching data for: {args.specialty}")
    elif args.all:
        specialties = None
        logger.info("Fetching general sample (no specialty filter)")
    else:
        specialties = EMBER_RELEVANT_SPECIALTIES
        logger.info(f"Fetching {len(specialties)} Ember-relevant specialties")

    logger.info(f"Max rows per specialty: {args.max_rows:,}")
    logger.info("")

    # Run ingestion
    output_path = ingest_and_save(
        max_rows_per_specialty=args.max_rows,
        specialties=specialties,
    )

    if output_path and output_path.exists():
        logger.info("")
        logger.info("=" * 60)
        logger.info("Setup complete!")
        logger.info(f"Data saved to: {output_path}")
        logger.info("")
        logger.info("Launch the dashboard with:")
        logger.info("    streamlit run dashboard/app.py")
        logger.info("=" * 60)
    else:
        logger.error("Data setup failed. Check your network connection and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
