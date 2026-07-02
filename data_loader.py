"""Data download, cleaning and feature engineering for HDB resale prediction."""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

DATASET_ID = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"  # HDB resale prices Jan 2017 onwards
DATASTORE_URL = "https://data.gov.sg/api/action/datastore_search"
BASE_DIR = Path(__file__).resolve().parent
REF_DIR = BASE_DIR / "reference_data"


def fetch_hdb_resale_data(limit: int = 5000, max_records: Optional[int] = None) -> pd.DataFrame:
    """Fetch HDB resale-flat records from data.gov.sg with pagination."""
    records: list[dict] = []
    offset = 0
    while True:
        params = {"resource_id": DATASET_ID, "limit": limit, "offset": offset}
        response = requests.get(DATASTORE_URL, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        if not payload.get("success"):
            raise RuntimeError(f"data.gov.sg API returned unsuccessful response: {payload}")
        batch = payload["result"].get("records", [])
        if not batch:
            break
        records.extend(batch)
        offset += len(batch)
        if max_records and len(records) >= max_records:
            records = records[:max_records]
            break
        total = payload["result"].get("total")
        if total is not None and offset >= total:
            break
        time.sleep(0.1)
    return pd.DataFrame(records)


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Distance between two latitude/longitude points in kilometres."""
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def parse_remaining_lease(value: str) -> float:
    """Convert values like '61 years 04 months' or '61 years' to decimal years."""
    text = str(value).lower()
    years = 0.0
    months = 0.0
    y = pd.Series([text]).str.extract(r"(\d+)\s+years?")[0].iloc[0]
    m = pd.Series([text]).str.extract(r"(\d+)\s+months?")[0].iloc[0]
    if pd.notna(y):
        years = float(y)
    if pd.notna(m):
        months = float(m)
    return years + months / 12


def add_location_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add region, town centroid, nearest MRT, nearest school count and CBD distance."""
    df = df.copy()
    town_ref = pd.read_csv(REF_DIR / "town_reference.csv")
    mrt = pd.read_csv(REF_DIR / "mrt_stations.csv")
    schools = pd.read_csv(REF_DIR / "schools.csv")
    town_ref["town"] = town_ref["town"].str.upper().str.strip()
    df = df.merge(town_ref, on="town", how="left")

    def nearest_mrt(row) -> float:
        if pd.isna(row["latitude"]):
            return np.nan
        return min(haversine_km(row["latitude"], row["longitude"], r.latitude, r.longitude) for r in mrt.itertuples())

    def school_count(row, radius_km: float = 2.0) -> int:
        if pd.isna(row["latitude"]):
            return 0
        return sum(haversine_km(row["latitude"], row["longitude"], r.latitude, r.longitude) <= radius_km for r in schools.itertuples())

    df["distance_to_nearest_mrt_km"] = df.apply(nearest_mrt, axis=1)
    df["nearby_schools_2km"] = df.apply(school_count, axis=1)
    return df


def clean_hdb_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data and create ML-ready features."""
    df = df.copy()
    required = [
        "month", "town", "flat_type", "storey_range", "floor_area_sqm", "flat_model",
        "lease_commence_date", "remaining_lease", "resale_price",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    for col in ["town", "flat_type", "flat_model"]:
        df[col] = df[col].astype(str).str.upper().str.strip()

    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df["sale_year"] = df["month"].dt.year
    df["sale_month"] = df["month"].dt.month
    df["floor_area_sqm"] = pd.to_numeric(df["floor_area_sqm"], errors="coerce")
    df["lease_commence_date"] = pd.to_numeric(df["lease_commence_date"], errors="coerce")
    df["resale_price"] = pd.to_numeric(df["resale_price"], errors="coerce")

    storey_parts = df["storey_range"].astype(str).str.extract(r"(\d+)\s+TO\s+(\d+)")
    df["storey_mid"] = storey_parts.astype(float).mean(axis=1)
    df["remaining_lease_years"] = df["remaining_lease"].apply(parse_remaining_lease)
    df["flat_age"] = df["sale_year"] - df["lease_commence_date"]
    df["price_per_sqm"] = df["resale_price"] / df["floor_area_sqm"]

    df = add_location_features(df)

    keep = [
        "town", "region", "flat_type", "flat_model", "floor_area_sqm", "storey_mid",
        "lease_commence_date", "remaining_lease_years", "flat_age", "sale_year", "sale_month",
        "latitude", "longitude", "distance_to_cbd_km", "distance_to_nearest_mrt_km",
        "nearby_schools_2km", "price_per_sqm", "resale_price",
    ]
    return df[keep].dropna()
