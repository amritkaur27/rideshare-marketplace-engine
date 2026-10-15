import duckdb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import urllib.request
 
 
# Paths are relative to this script so the code runs on any machine
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, 'data')
file_path = os.path.join(data_dir, 'fhvhv_tripdata_2025-07.parquet')
 
# Download the dataset from the NYC TLC if not already in the data folder
if not os.path.exists(file_path):
    os.makedirs(data_dir, exist_ok=True)
    print("Data file not found. Downloading from NYC TLC (large file, may take a few minutes)...")
    urllib.request.urlretrieve(
        'https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2025-07.parquet',
        file_path
    )
 
print("Extracting and sampling data via DuckDB...")
 
con = duckdb.connect(database=':memory:')
ml_query = f"""
    SELECT 
        trip_miles,
        PULocationID,
        DOLocationID,
        hour(pickup_datetime) AS pickup_hour,
        dayofweek(pickup_datetime) AS day_of_week,
        CASE WHEN hour(pickup_datetime) IN (7,8,9,16,17,18) THEN 1 ELSE 0 END AS is_peak,
        CASE WHEN PULocationID IN (132, 138, 1) OR DOLocationID IN (132, 138, 1) THEN 1 ELSE 0 END AS is_airport,
        base_passenger_fare,
        driver_pay
    FROM '{file_path}'
    WHERE hvfhs_license_num = 'HV0005'
      AND base_passenger_fare > 0 
      AND driver_pay > 0
    USING SAMPLE 200000 (reservoir, 42);
"""
df_ml = con.execute(ml_query).df()
 
# Define Features 
X = df_ml[['trip_miles', 'PULocationID', 'DOLocationID', 'pickup_hour', 'day_of_week', 'is_peak', 'is_airport']]
y_fare = df_ml['base_passenger_fare']
y_pay = df_ml['driver_pay']
 
X_train, X_test, y_fare_train, y_fare_test, y_pay_train, y_pay_test = train_test_split(
    X, y_fare, y_pay, test_size=0.2, random_state=42
)
 
# Train Models 
print("\nTraining Final Models...")
fare_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
fare_model.fit(X_train, y_fare_train)
 
pay_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
pay_model.fit(X_train, y_pay_train)
 
 
print("\n" + "="*50)
print(" PART 2: COMPREHENSIVE MODEL VALIDATION SUITE")
print("="*50)
 
# Cross-Validation 
cv = KFold(n_splits=5, shuffle=True, random_state=42)
 
fare_cv_scores = cross_val_score(fare_model, X_train, y_fare_train, cv=cv, scoring='r2')
print(f"\n[5-Fold Cross-Validation - Passenger Fare]")
print(f"Mean R²: {fare_cv_scores.mean():.3f} (Std: {fare_cv_scores.std():.3f})")
 
pay_cv_scores = cross_val_score(pay_model, X_train, y_pay_train, cv=cv, scoring='r2')
print(f"\n[5-Fold Cross-Validation - Driver Pay]")
print(f"Mean R²: {pay_cv_scores.mean():.3f} (Std: {pay_cv_scores.std():.3f})")
 
# Bias-Variance Tradeoff
print(f"\n[Bias-Variance Tradeoff Experiment (Driver Pay)]")
depths = [3, 5, 8, 10, 15, 20, None]
depth_results = []
for d in depths:
    rf_temp = RandomForestRegressor(n_estimators=50, max_depth=d, random_state=42)
    rf_temp.fit(X_train, y_pay_train)
    tr_r2 = r2_score(y_pay_train, rf_temp.predict(X_train))
    te_r2 = r2_score(y_pay_test, rf_temp.predict(X_test))
    tr_mae = mean_absolute_error(y_pay_train, rf_temp.predict(X_train))
    te_mae = mean_absolute_error(y_pay_test, rf_temp.predict(X_test))
    depth_results.append({
        'max_depth': str(d), 
        'Train R²': tr_r2, 
        'Test R²': te_r2, 
        'Train MAE': tr_mae, 
        'Test MAE': te_mae
    })
print(pd.DataFrame(depth_results).to_string(index=False))
 
# Model Comparison
print(f"\n[Model Architecture Comparison (Driver Pay MAE)]")
dummy = DummyRegressor(strategy="mean")
dummy.fit(X_train, y_pay_train)
dummy_mae = mean_absolute_error(y_pay_test, dummy.predict(X_test))
 
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_pay_train)
lin_mae = mean_absolute_error(y_pay_test, lin_reg.predict(X_test))
 
rf_mae = mean_absolute_error(y_pay_test, pay_model.predict(X_test))
 
print(f"Mean Baseline MAE: ${dummy_mae:.2f}")
print(f"Linear Regression MAE: ${lin_mae:.2f}")
print(f"Random Forest MAE: ${rf_mae:.2f}")
 
# Segment Bias Analysis & Residuals Setup
test_results = X_test.copy()
test_results['actual_pay'] = y_pay_test
test_results['predicted_pay'] = pay_model.predict(X_test)
test_results['pay_residual'] = test_results['actual_pay'] - test_results['predicted_pay']
 
# Hourly Bias
hour_bias = test_results.groupby('pickup_hour')['pay_residual'].agg(['mean', lambda x: np.mean(np.abs(x)), 'count'])
hour_bias.columns = ['Mean Residual (Bias)', 'MAE', 'Count']
print(f"\n[Top 5 Most Underpredicted Hours]")
print(hour_bias.sort_values('Mean Residual (Bias)', ascending=False).head(5))
 
# Peak vs Non-Peak Bias
peak_bias = test_results.groupby('is_peak')['pay_residual'].agg(['mean', lambda x: np.mean(np.abs(x)), 'count'])
peak_bias.columns = ['Mean Residual (Bias)', 'MAE', 'Count']
print(f"\n[Bias Check: Peak vs Non-Peak]")
print(peak_bias)
 
# Trip Distance Bias
test_results['distance_bin'] = pd.cut(
    test_results['trip_miles'],
    bins=[0, 2, 5, 10, 20, 100],
    labels=['0-2 mi', '2-5 mi', '5-10 mi', '10-20 mi', '20+ mi']
)
distance_bias = test_results.groupby('distance_bin', observed=False)['pay_residual'].agg(['mean', lambda x: np.mean(np.abs(x)), 'count'])
distance_bias.columns = ['Mean Residual (Bias)', 'MAE', 'Count']
print(f"\n[Bias Check by Trip Distance]")
print(distance_bias)
 
# Feature Importance
print(f"\n[Feature Importance - Driver Pay]")
importances = pd.Series(pay_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importances)
 
# Final Validated Models
project_dir = os.path.join(base_dir, 'outputs')
os.makedirs(project_dir, exist_ok=True)
joblib.dump(fare_model, f'{project_dir}/fare_model.pkl')
joblib.dump(pay_model, f'{project_dir}/pay_model.pkl')
print(f"\nModels successfully validated and saved to {project_dir}")
