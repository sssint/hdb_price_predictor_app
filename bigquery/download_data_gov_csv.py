"""Download the HDB resale data from data.gov.sg into a local CSV.

This is useful for manual BigQuery upload when direct API-to-BigQuery automation fails.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from data_loader import fetch_hdb_resale_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download HDB resale data.gov.sg records to CSV")
    parser.add_argument("--output", default="data/manual_upload/hdb_resale_from_data_gov.csv")
    parser.add_argument("--max-records", type=int, default=None, help="Optional limit for testing")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = fetch_hdb_resale_data(max_records=args.max_records)
    df.to_csv(output, index=False)
    print(f"Saved {len(df):,} rows to {output}")
