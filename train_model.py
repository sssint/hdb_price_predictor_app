"""Train a CatBoost HDB resale price prediction model."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from data_loader import clean_hdb_data, fetch_hdb_resale_data

MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "hdb_catboost_model.cbm"
METRICS_PATH = MODEL_DIR / "metrics.joblib"

FEATURES = [
    "town", "region", "flat_type", "flat_model",
    "floor_area_sqm", "storey_mid", "lease_commence_date", "remaining_lease_years", "flat_age",
    "sale_year", "sale_month", "latitude", "longitude", "distance_to_cbd_km",
    "distance_to_nearest_mrt_km", "nearby_schools_2km",
]
TARGET = "resale_price"
CAT_FEATURES = ["town", "region", "flat_type", "flat_model"]


def train(max_records: int | None = None) -> dict:
    raw = fetch_hdb_resale_data(max_records=max_records)
    data = clean_hdb_data(raw)
    X = data[FEATURES]
    y = np.log1p(data[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_pool = Pool(X_train, y_train, cat_features=CAT_FEATURES)
    test_pool = Pool(X_test, y_test, cat_features=CAT_FEATURES)

    model = CatBoostRegressor(
        iterations=1200,
        learning_rate=0.05,
        depth=8,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        early_stopping_rounds=80,
        verbose=100,
    )
    model.fit(train_pool, eval_set=test_pool)

    pred = np.expm1(model.predict(X_test))
    actual = np.expm1(y_test)

    metrics = {
        "model_type": "CatBoostRegressor",
        "records_used": int(len(data)),
        "mae_sgd": float(mean_absolute_error(actual, pred)),
        "rmse_sgd": float(root_mean_squared_error(actual, pred)),
        "mape": float(mean_absolute_percentage_error(actual, pred)),
        "r2": float(r2_score(actual, pred)),
        "towns": sorted(data["town"].unique().tolist()),
        "regions": sorted(data["region"].unique().tolist()),
        "flat_types": sorted(data["flat_type"].unique().tolist()),
        "flat_models": sorted(data["flat_model"].unique().tolist()),
        "min_year": int(data["sale_year"].min()),
        "max_year": int(data["sale_year"].max()),
        "features": FEATURES,
        "cat_features": CAT_FEATURES,
    }
    MODEL_DIR.mkdir(exist_ok=True)
    model.save_model(str(MODEL_PATH))
    joblib.dump(metrics, METRICS_PATH)
    return metrics


def load_model() -> CatBoostRegressor:
    model = CatBoostRegressor()
    model.load_model(str(MODEL_PATH))
    return model


if __name__ == "__main__":
    print(train())
