"""Read HDB training data from BigQuery.

Used by training notebooks/scripts when you want BigQuery as the source instead of local CSV/API.
"""
from __future__ import annotations

from google.cloud import bigquery


def read_table(project_id: str, dataset: str = "hdb_ml", table: str = "resale_transactions_raw", location: str = "asia-southeast1"):
    client = bigquery.Client(project=project_id, location=location)
    query = f"SELECT * FROM `{project_id}.{dataset}.{table}`"
    return client.query(query).to_dataframe()


def read_feature_table(project_id: str, dataset: str = "hdb_ml", table: str = "hdb_feature_table", location: str = "asia-southeast1"):
    client = bigquery.Client(project=project_id, location=location)
    query = f"SELECT * FROM `{project_id}.{dataset}.{table}`"
    return client.query(query).to_dataframe()
