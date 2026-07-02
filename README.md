# HDB Resale Price Prediction App - CatBoost + Extra Features

This is a Streamlit machine-learning application for Singapore HDB resale price prediction, ready for Google Cloud Run.

## Dataset

Main dataset: Singapore HDB resale flat prices from data.gov.sg, resource ID:

```text
d_8b84c4ee58e3cfc0ece0d773c8ca6abc
```

The app downloads the official resale transaction records from Jan 2017 onwards.

## ML model

The project uses:

```text
CatBoostRegressor
```

CatBoost is suitable because HDB resale data contains many categorical fields such as town, flat type and flat model.

## Features included

Categorical features:

```text
town
region
flat_type
flat_model
```

Numerical features:

```text
floor_area_sqm
storey_mid
lease_commence_date
remaining_lease_years
flat_age
sale_year
sale_month
latitude
longitude
distance_to_cbd_km
distance_to_nearest_mrt_km
nearby_schools_2km
```

Extra reference data is stored in:

```text
reference_data/town_reference.csv
reference_data/mrt_stations.csv
reference_data/schools.csv
```

These files provide approximate town centroids, MRT station points and school points for feature engineering.
For production quality, replace these with exact block-level geocoding from OneMap or another approved geospatial source.

## Folder structure

```text
hdb_price_streamlit_cloudrun/
├── app.py
├── data_loader.py
├── train_model.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── README.md
└── reference_data/
    ├── town_reference.csv
    ├── mrt_stations.csv
    └── schools.csv
```

## Run locally

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

The first training run downloads the data from data.gov.sg and saves:

```text
model/hdb_catboost_model.cbm
model/metrics.joblib
```

## Build Docker image

```bash
docker build -t hdb-price-app .
docker run -p 8080:8080 hdb-price-app
```

Open:

```text
http://localhost:8080
```

## Deploy to Google Cloud Run

Set variables:

```bash
PROJECT_ID=your-gcp-project-id
REGION=asia-southeast1
SERVICE_NAME=hdb-price-app
```

Build and submit:

```bash
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME
```

Deploy:

```bash
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 900
```

## Recommended production improvement

For faster Cloud Run startup, train the model before deployment and include the `model/` folder in the container image, or store the trained model in Google Cloud Storage and load it during startup.

## Notes

This project is for learning and portfolio use. It is not an official HDB valuation tool. Actual prices depend on renovation, floor level, exact block, facing, transport access, market timing and buyer-seller negotiation.


## Notebooks included

The project now includes notebooks for explanation and experimentation:

```text
notebooks/
├── 01_hdb_eda.ipynb
├── 02_train_catboost_model.ipynb
└── 03_app_prediction_demo.ipynb
```

Use notebooks for learning, EDA, screenshots, and assignment explanation. Keep the `.py` files for Cloud Run deployment because Docker starts the app with `streamlit run app.py`.

## BigQuery manual injection fallback

Use this path when automated download from data.gov.sg or Google authentication inside the app is not working.

### Option A: Download from data.gov.sg to local CSV

```bash
python bigquery/download_data_gov_csv.py \
  --output data/manual_upload/hdb_resale_from_data_gov.csv
```

For a quick test:

```bash
python bigquery/download_data_gov_csv.py \
  --output data/manual_upload/hdb_resale_sample.csv \
  --max-records 5000
```

### Option B: Manually download CSV from data.gov.sg

1. Go to data.gov.sg.
2. Search for **HDB resale flat prices Jan 2017 onwards**.
3. Download the CSV file.
4. Save it into:

```text
data/manual_upload/
```

Example:

```text
data/manual_upload/hdb_resale_from_data_gov.csv
```

### Upload the CSV to BigQuery

First authenticate locally:

```bash
gcloud auth application-default login
```

Then upload:

```bash
python bigquery/manual_upload_csv_to_bigquery.py \
  --project-id YOUR_PROJECT_ID \
  --dataset hdb_ml \
  --table resale_transactions_raw \
  --csv data/manual_upload/hdb_resale_from_data_gov.csv \
  --location asia-southeast1
```

This creates or replaces:

```text
YOUR_PROJECT_ID.hdb_ml.resale_transactions_raw
```

### Manual BigQuery Console method

If Python authentication is still not working:

1. Open Google Cloud Console.
2. Go to **BigQuery**.
3. Create dataset: `hdb_ml`.
4. Click **Create table**.
5. Source: Upload CSV.
6. Destination table: `resale_transactions_raw`.
7. Enable schema auto-detect.
8. Create table.

After this, training can read from BigQuery using:

```python
from bigquery.bigquery_reader import read_table

df = read_table(
    project_id="YOUR_PROJECT_ID",
    dataset="hdb_ml",
    table="resale_transactions_raw",
)
```
