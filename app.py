import streamlit as st
import pandas as pd
import joblib
import os

# Set page configuration
st.set_page_config(page_title="Lyft Marketplace Pricing & Optimization Engine", layout="wide")

# Load models safely using absolute path
project_dir = '/Users/amritkaur/Desktop/Projects/Rideshare Project'
fare_model_path = os.path.join(project_dir, 'fare_model.pkl')
pay_model_path = os.path.join(project_dir, 'pay_model.pkl')

@st.cache_resource
def load_models():
    fare_model = joblib.load(fare_model_path)
    pay_model = joblib.load(pay_model_path)
    return fare_model, pay_model

fare_model, pay_model = load_models()

# App Header
st.title("Lyft Dynamic Pricing & Marketplace Optimization Engine")
st.markdown("Simulate ride parameters to predict market-clearing passenger fares, driver payouts, and platform take-rates in real time.")

# Sidebar controls for ride simulation (7 features matching training schema)
st.sidebar.header("Ride Parameters")
miles = st.sidebar.slider("Trip Distance (Miles)", 0.5, 25.0, 4.5, 0.5)
pickup_hour = st.sidebar.slider("Pickup Hour (0-23)", 0, 23, 17)

# Corrected day-of-week mapping matching DuckDB dayofweek() (0 = Sunday, 1 = Monday, etc.)
day_of_week = st.sidebar.selectbox(
    "Day of Week",
    options=[0, 1, 2, 3, 4, 5, 6],
    format_func=lambda x: ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'][x]
)

is_peak = 1 if pickup_hour in [7, 8, 9, 16, 17, 18] else 0

# Location selectors
pulocation_id = st.sidebar.number_input("Pickup Location ID (PULocationID)", min_value=1, max_value=265, value=132)
dolocation_id = st.sidebar.number_input("Dropoff Location ID (DOLocationID)", min_value=1, max_value=265, value=234)

# Automated native airport feature determination matching training logic
airport_ids = [132, 138, 1]
is_airport = 1 if (pulocation_id in airport_ids or dolocation_id in airport_ids) else 0

# Build input dataframe matching model feature schema exactly
input_data = pd.DataFrame({
    'trip_miles': [miles],
    'PULocationID': [pulocation_id],
    'DOLocationID': [dolocation_id],
    'pickup_hour': [pickup_hour],
    'day_of_week': [day_of_week],
    'is_peak': [is_peak],
    'is_airport': [is_airport]
})

# Run base predictions from model
predicted_fare = fare_model.predict(input_data)[0]
predicted_payout = pay_model.predict(input_data)[0]

# Calculate take-rate and platform revenue
platform_cut = predicted_fare - predicted_payout
take_rate_pct = (platform_cut / predicted_fare) * 100

# Main Panel Display
st.subheader("Marketplace Clearing Results")

if is_airport == 1:
    st.info("✈️ **Airport Zone Route:** Detected major airport transit zone; incorporated natively into model prediction.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Predicted Passenger Fare", f"${predicted_fare:.2f}")
col2.metric("Predicted Driver Payout", f"${predicted_payout:.2f}")
col3.metric("Platform Revenue", f"${platform_cut:.2f}")
col4.metric("Take-Rate", f"{take_rate_pct:.1f}%")

# Marketplace Health Assessment (Statistically neutral language)
st.markdown("---")
st.subheader("Marketplace Margin Assessment")

if take_rate_pct < 15:
    st.warning("⚠️ Low Take-Rate: Platform margin is below the selected operating threshold.")
elif take_rate_pct > 30:
    st.info("ℹ️ Elevated Take-Rate: Platform margin exceeds the selected operating threshold.")
else:
    st.success("✅ Target Range: Take-rate falls within the selected operating range.")

# --- Optimization Scenario Analysis Layer ---
st.markdown("---")
st.subheader("Pricing Sensitivity Analysis")
st.markdown("Test alternative passenger fare price points against the baseline model payout to evaluate sensitivity on platform revenue and take-rates.")

# Generate dynamic scenario range centered around predicted fare
base_fare_rounded = round(predicted_fare / 5) * 5
fare_options = [max(10.0, base_fare_rounded - 10.0), max(12.0, base_fare_rounded - 5.0), base_fare_rounded, base_fare_rounded + 5.0, base_fare_rounded + 10.0]
fare_options = sorted(list(set(fare_options)))

scenario_results = []
for f in fare_options:
    rev = f - predicted_payout
    tr = (rev / f) * 100 if f > 0 else 0
    scenario_results.append({
        "Passenger Fare ($)": round(f, 2),
        "Driver Payout ($)": round(predicted_payout, 2),
        "Platform Revenue ($)": round(rev, 2),
        "Take-Rate (%)": round(tr, 1)
    })

scenario_df = pd.DataFrame(scenario_results)
st.dataframe(scenario_df, use_container_width=True)