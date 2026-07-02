from __future__ import annotations

from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from catboost import Pool

from data_loader import REF_DIR, haversine_km
from train_model import CAT_FEATURES, FEATURES, METRICS_PATH, MODEL_PATH, load_model, train

st.set_page_config(page_title="HDB Resale Price Predictor", page_icon="🏠", layout="wide")
st.title("🏠 HDB Resale Price Predictor")
st.caption("CatBoost model using Singapore HDB resale-flat transactions from data.gov.sg. Prediction is indicative only.")

@st.cache_resource(show_spinner=True)
def load_or_train_model():
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        train(max_records=None)
    return load_model(), joblib.load(METRICS_PATH)

@st.cache_data
def load_reference():
    town_ref = pd.read_csv(REF_DIR / "town_reference.csv")
    mrt = pd.read_csv(REF_DIR / "mrt_stations.csv")
    schools = pd.read_csv(REF_DIR / "schools.csv")
    town_ref["town"] = town_ref["town"].str.upper().str.strip()
    return town_ref, mrt, schools

model, metrics = load_or_train_model()
town_ref, mrt, schools = load_reference()

with st.expander("Model information", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", metrics["model_type"])
    c2.metric("Records", f"{metrics['records_used']:,}")
    c3.metric("MAE", f"SGD {metrics['mae_sgd']:,.0f}")
    c4.metric("R²", f"{metrics['r2']:.3f}")
    st.write(f"Training period: {metrics['min_year']}–{metrics['max_year']}")
    st.write("Features used:", ", ".join(metrics["features"]))

st.subheader("Enter flat details")
col1, col2, col3 = st.columns(3)
with col1:
    town = st.selectbox("Town", metrics["towns"], index=metrics["towns"].index("TAMPINES") if "TAMPINES" in metrics["towns"] else 0)
    flat_type = st.selectbox("Flat type", metrics["flat_types"], index=metrics["flat_types"].index("4 ROOM") if "4 ROOM" in metrics["flat_types"] else 0)
    flat_model = st.selectbox("Flat model", metrics["flat_models"], index=0)
    floor_area_sqm = st.number_input("Floor area (sqm)", 30.0, 220.0, 90.0, step=1.0)
with col2:
    storey_mid = st.slider("Approx. storey", 1, 50, 10)
    lease_commence_date = st.number_input("Lease commence year", 1960, date.today().year, 1990, step=1)
    sale_year = st.number_input("Prediction year", 2017, date.today().year + 1, date.today().year, step=1)
    sale_month = st.slider("Prediction month", 1, 12, date.today().month)
with col3:
    town_row = town_ref[town_ref["town"] == town].iloc[0]
    region = town_row["region"]
    latitude = float(town_row["latitude"])
    longitude = float(town_row["longitude"])
    distance_to_cbd_km = float(town_row["distance_to_cbd_km"])
    distance_to_nearest_mrt_km = min(haversine_km(latitude, longitude, r.latitude, r.longitude) for r in mrt.itertuples())
    nearby_schools_2km = sum(haversine_km(latitude, longitude, r.latitude, r.longitude) <= 2 for r in schools.itertuples())
    remaining_lease_years = max(0, 99 - (sale_year - lease_commence_date))
    flat_age = sale_year - lease_commence_date
    st.text_input("Region", value=region, disabled=True)
    st.number_input("Remaining lease years", value=float(remaining_lease_years), disabled=True)
    st.number_input("Nearest MRT distance (km)", value=float(round(distance_to_nearest_mrt_km, 2)), disabled=True)
    st.number_input("Nearby schools within 2km", value=int(nearby_schools_2km), disabled=True)

input_df = pd.DataFrame([{
    "town": town,
    "region": region,
    "flat_type": flat_type,
    "flat_model": flat_model,
    "floor_area_sqm": floor_area_sqm,
    "storey_mid": storey_mid,
    "lease_commence_date": lease_commence_date,
    "remaining_lease_years": remaining_lease_years,
    "flat_age": flat_age,
    "sale_year": sale_year,
    "sale_month": sale_month,
    "latitude": latitude,
    "longitude": longitude,
    "distance_to_cbd_km": distance_to_cbd_km,
    "distance_to_nearest_mrt_km": distance_to_nearest_mrt_km,
    "nearby_schools_2km": nearby_schools_2km,
}])[FEATURES]

if st.button("Predict resale price", type="primary"):
    pool = Pool(input_df, cat_features=CAT_FEATURES)
    price = np.expm1(model.predict(pool))[0]
    price_per_sqm = price / floor_area_sqm
    c1, c2 = st.columns(2)
    c1.metric("Estimated resale price", f"SGD {price:,.0f}")
    c2.metric("Estimated price per sqm", f"SGD {price_per_sqm:,.0f}")
    with st.expander("Input features sent to model"):
        st.dataframe(input_df)
    st.info("This is a machine-learning estimate, not an official valuation. Exact block, renovation, facing, transport access and negotiation can change the actual resale price.")
