# Rideshare Pricing Engine

An end-to-end pricing project for a rideshare marketplace. It trains machine learning models on NYC trip data to predict passenger fares and driver pay, validates them, checks the driver-pay model for bias across trip segments, and serves the results in an interactive Streamlit app.

> Independent personal project using public data. 

## What it does

1. Extracts and samples data from a large parquet file using DuckDB (SQL), so the full file never has to be loaded into memory.
2. Trains two Random Forest model: one for passenger fare, one for driver pay.
3. Validates the models with 5-fold cross-validation, a bias-variance experiment across tree depths, and a comparison against a mean baseline and linear regression.
4. Analyzes segment bias in the driver-pay model by looking at residuals by hour, peak vs. off-peak, and trip distance.
5. Serves an interactive app where you can simulate a ride and see predicted fare, driver payout, platform revenue, and take-rate, plus a pricing sensitivity table.

## Data

NYC Taxi & Limousine Commission (TLC) High Volume For-Hire Vehicle trip records, July 2025. The analysis keeps trips from license `HV0005` (Lyft) with positive fare and driver pay, then samples 200,000 trips.

- Source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- The training script downloads the file automatically on first run. It is large (about 500 MB), so the first run takes a few minutes.


The app needs the models from step 1, so run the training script first.

## Segment bias findings

Summarize what the residual analysis showed, in two or three lines. For example: which segments (peak vs. off-peak, trip-distance bins, specific hours) the driver-pay model over- or under-predicted, and by roughly how much.

## App features

- Sliders and inputs for trip distance, pickup hour, day of week, and pickup/dropoff zone IDs
- Predicted passenger fare, driver payout, platform revenue, and take-rate
- Automatic airport-trip detection (JFK, LaGuardia, and Newark zones)
- A pricing sensitivity table showing how platform revenue and take-rate change at alternative fare levels

## Limitations

- Trained on one month of NYC data for one operator, so it does not generalize to other cities or seasons.
- Does not model demand, surge pricing, tips, or driver supply, so it predicts typical fares and pay rather than market-clearing prices.
- The sensitivity table holds the predicted driver payout fixed while varying the fare. It shows margin arithmetic, not how riders or drivers would react to a price change.
- Location IDs are used as numeric features, and the model is less reliable in the segments flagged in the bias analysis.
- The take-rate thresholds in the app (15% and 30%) are illustrative, not industry benchmarks.

## Tech stack

Python, DuckDB, pandas, NumPy, scikit-learn, joblib, Streamlit
