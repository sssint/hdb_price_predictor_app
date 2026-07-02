-- Replace YOUR_PROJECT_ID before running.
CREATE SCHEMA IF NOT EXISTS `YOUR_PROJECT_ID.hdb_ml`
OPTIONS(location = "asia-southeast1");

-- Raw table can also be created by CSV autodetect during upload.
-- This optional table definition documents the expected main columns.
CREATE TABLE IF NOT EXISTS `YOUR_PROJECT_ID.hdb_ml.resale_transactions_raw` (
  month STRING,
  town STRING,
  flat_type STRING,
  block STRING,
  street_name STRING,
  storey_range STRING,
  floor_area_sqm FLOAT64,
  flat_model STRING,
  lease_commence_date INT64,
  remaining_lease STRING,
  resale_price FLOAT64
);
