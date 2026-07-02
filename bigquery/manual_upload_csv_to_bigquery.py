"""Manual fallback loader: upload local CSV files to Google BigQuery.

Use this when automated data.gov.sg download or application authentication is not working.

Example:
    python bigquery/manual_upload_csv_to_bigquery.py \
      --project-id YOUR_PROJECT_ID \
      --dataset hdb_ml \
      --table resale_transactions_raw \
      --csv data/manual_upload/resale-flat-prices.csv \
      --location asia-southeast1
"""
from __future__ import annotations

import argparse
from pathlib import Path

from google.api_core.exceptions import NotFound
from google.cloud import bigquery


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a local CSV file to BigQuery")
    parser.add_argument("--project-id", required=True, help="Google Cloud project ID")
    parser.add_argument("--dataset", default="hdb_ml", help="BigQuery dataset name")
    parser.add_argument("--table", default="resale_transactions_raw", help="BigQuery table name")
    parser.add_argument("--csv", required=True, help="Path to local CSV file")
    parser.add_argument("--location", default="asia-southeast1", help="BigQuery dataset location")
    parser.add_argument("--write-mode", default="WRITE_TRUNCATE", choices=["WRITE_TRUNCATE", "WRITE_APPEND"], help="Replace or append data")
    return parser.parse_args()


def ensure_dataset(client: bigquery.Client, dataset_id: str, location: str) -> bigquery.Dataset:
    dataset_ref = bigquery.Dataset(dataset_id)
    dataset_ref.location = location
    try:
        return client.get_dataset(dataset_ref)
    except NotFound:
        return client.create_dataset(dataset_ref)


def upload_csv(project_id: str, dataset: str, table: str, csv_path: str, location: str, write_mode: str) -> None:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    client = bigquery.Client(project=project_id, location=location)
    dataset_id = f"{project_id}.{dataset}"
    ensure_dataset(client, dataset_id, location)

    table_id = f"{project_id}.{dataset}.{table}"
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        autodetect=True,
        write_disposition=write_mode,
        allow_quoted_newlines=True,
        encoding="UTF-8",
    )

    with path.open("rb") as f:
        job = client.load_table_from_file(f, table_id, job_config=job_config)
    job.result()

    loaded_table = client.get_table(table_id)
    print(f"Uploaded {loaded_table.num_rows:,} rows to {table_id}")


if __name__ == "__main__":
    args = parse_args()
    upload_csv(
        project_id=args.project_id,
        dataset=args.dataset,
        table=args.table,
        csv_path=args.csv,
        location=args.location,
        write_mode=args.write_mode,
    )
